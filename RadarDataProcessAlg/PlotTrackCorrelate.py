# encoding=utf-8

import math
import os
from collections import defaultdict, OrderedDict
from random import sample

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Mp4ToGif import Video2GifConverter
from TrackInit import direct_method_with_bkg, logic_method_with_bkg, corrected_logic_method_with_bkg
from TrackInit import extrapolate_plot, Plot, PlotStates
from TrackInit import markers, colors
from TrackInit import start_gate_check, relate_gate_check


def get_predict_plot(track, cycle_time):
    """
    :param track:
    :param cycle_time:
    :return:
    """
    last_cycle = max([plot.cycle_ for plot in track.plots_])

    # ----- 外推next cycle预测点迹
    # 提取前cycle和当前cycle的点迹
    plot_pre = [plot for plot in track.plots_ if plot.cycle_ ==
                last_cycle - 1][0]
    plot_cur = [plot for plot in track.plots_ if plot.cycle_ == last_cycle][0]

    # 提取track在last_cycle的运动状态
    # x = plot_cur.x_
    # y = plot_cur.y_
    v = plot_cur.v_
    a = plot_cur.a_
    heading = plot_cur.heading_

    # 预测位移值
    s = v * cycle_time

    # 计算(直线)外推点迹(预测点)笛卡尔坐标
    x_extra, y_extra = extrapolate_plot([plot_pre.x_, plot_pre.y_], [
        plot_cur.x_, plot_cur.y_], s)

    # 构建预测点迹对象: 跟last_cycle的plot保持一致
    plot_pred = Plot(last_cycle + 1, x_extra, y_extra, v, a, heading)

    return plot_pred


def compute_cov_mat(plot_pred, can_plot_objs):
    """
    运动状态(5个):
    x: 点迹笛卡尔坐标x(m)
    y: 点迹笛卡尔坐标y(m)
    v: 点迹速度(m/s)
    a: 点迹加速度(m/s²)
    heading: 航向偏转角°
    :param plot_pred:
    :param can_plot_objs:
    :return:
    """
    # ----- 计算滤波器残差矩阵
    res_mat = np.zeros((len(can_plot_objs), 5), dtype=np.float32)
    for i, can_plot_obj in enumerate(can_plot_objs):
        x_res = can_plot_obj.x_ - plot_pred.x_
        y_res = can_plot_obj.y_ - plot_pred.y_
        v_res = can_plot_obj.v_ - plot_pred.v_
        a_res = can_plot_obj.a_ - plot_pred.a_
        heading_res = can_plot_obj.heading_ - plot_pred.heading_

        res_mat[i, 0] = x_res
        res_mat[i, 1] = y_res
        res_mat[i, 2] = v_res
        res_mat[i, 3] = a_res
        res_mat[i, 4] = heading_res

    # print(res_mat)

    # ----- 计算滤波器残差矩阵的协方差矩阵
    cov_mat = np.cov(res_mat, rowvar=False)

    if res_mat.shape[0] == 1:
        return np.zeros((1, 1), dtype=np.float32)
    else:
        return cov_mat


def is_plot_in_relate_gate(plot_obs, plot_pred, σ_s):
    """
    :param plot_obs:
    :param plot_pred:
    :param σ_s:
    :return:
    """
    shift_vector = np.array(plot_obs) - np.array([plot_pred.x_, plot_pred.y_])
    l2_dist = np.linalg.norm(shift_vector, ord=2)  # 计算实际点迹与外推点迹之间的距离
    return l2_dist < σ_s


def get_candidate_plot_objs(cycle_time, track, plot_pred, plots, σ_s):
    """
    :param cycle_time:
    :param track:
    :param plot_pred:
    :param plots:
    :param σ_s:
    :return:
    """
    last_cycle = max([plot.cycle_ for plot in track.plots_])

    # 提取前cycle和当前cycle的点迹
    plot_pre = [plot for plot in track.plots_ if plot.cycle_ ==
                last_cycle - 1][0]
    plot_cur = [plot for plot in track.plots_ if plot.cycle_ == last_cycle][0]

    # ----- 计算落入相关波门的候选点迹: 基于相关(跟踪)波门滤波
    candidate_plots = []
    for plot in plots:
        # shift_vector = np.array(plot) - np.array([plot_pred.x_, plot_pred.y_])
        # l2_dist = np.linalg.norm(shift_vector, ord=2)  # 计算实际点迹与外推点迹之间的距离
        # if l2_dist < σ_s:
        if is_plot_in_relate_gate(plot, plot_pred, σ_s):
            candidate_plots.append(plot)

    # print(candidate_plots)
    # 构建观测值(候选点迹对象)
    can_plot_objs = []
    for plot in candidate_plots:
        # ---估计观测点迹运动参数
        # 计算位移
        shift1 = np.array([plot[0], plot[1]]) - \
                 np.array([plot_cur.x_, plot_cur.y_])
        shift0 = np.array([plot_cur.x_, plot_cur.y_]) - \
                 np.array([plot_pre.x_, plot_pre.y_])

        # 计算速度
        dist0 = np.linalg.norm(shift0, ord=2)
        dist1 = np.linalg.norm(shift1, ord=2)
        v0 = dist0 / cycle_time
        v1 = dist1 / cycle_time

        # 计算加速度
        a = (v1 - v0) / cycle_time

        # 计算航向偏转角
        heading = math.degrees(
            math.acos(np.dot(shift0, shift1) / (dist0 * dist1 + 1e-6)))

        # 构建plot object
        plot_obj = Plot(last_cycle + 1, plot[0], plot[1], v1, a, heading)

        can_plot_objs.append(plot_obj)

    return can_plot_objs


def compute_ma_dist(cov_mat, can_plot_obj, plot_pred):
    """
    :param cov_mat:
    :param can_plot_obj:
    :param plot_pred:
    :return:
    """
    # 计算运动状态残差(观测向量-预测向量)向量
    res_vector = can_plot_obj - plot_pred  # Plot类对'-'进行了重载

    if cov_mat.size == 1:  # 只有1个观测点迹落入相关(跟踪)波门
        # 计算马氏距离: 此时是欧氏距离
        ma_dist = math.sqrt(res_vector.T.dot(res_vector))

    else:  # 5×5  有至少2个观测点迹落入相关(跟踪)波门
        # 计算马氏距离
        ma_dist = math.sqrt(
            np.dot(res_vector.T, np.linalg.inv(cov_mat)).dot(res_vector))

    return ma_dist


## ---------- visualization
def draw_plot_track_correspondence(cycle_time,
                                   plots_per_cycle,
                                   tracks,
                                   init_phase_plots_state_dict,
                                   correlate_phase_plots_state_dict,
                                   is_convert=False):
    """
    可视化点迹-行迹关联
    :param cycle_time:
    :param plots_per_cycle:
    :param tracks:
    :param init_phase_plots_state_dict:
    :param correlate_phase_plots_state_dict:
    :param is_convert:
    :return:
    """
    # ---------- 雷达扫描绘图参数
    pause_time = 0.05  # show a frame per 0.05s
    n_moves_per_cycle = cycle_time / pause_time  # 20 moves per cycle
    degrees_per_move = 360.0 / n_moves_per_cycle
    radians_per_move = degrees_per_move / math.pi

    # ---------- 数据汇总
    track_init_cycle = max([track.init_cycle_ for track in tracks])

    # 将两个阶段的plot对象信息字典合并
    plots_state_dict = defaultdict(list)
    for (k, v) in init_phase_plots_state_dict.items():
        plots_state_dict[k].extend(v)
    for (k, v) in correlate_phase_plots_state_dict.items():
        plots_state_dict[k].extend(correlate_phase_plots_state_dict[k])
    # print(plots_state_dict)

    # 升序排列
    plots_state_dict = sorted(plots_state_dict.items(),
                              key=lambda x: x[0], reverse=False)

    def draw_correlation(is_save=True):
        """
        :param is_save:
        :return:
        """
        # ---------- 绘图
        n_tracks = len(tracks)
        n_sample = n_tracks + len(PlotStates)

        colors_noise = sample(colors[3:], len(PlotStates))
        markers_noise = sample(markers[:3], len(PlotStates))

        colors_track = sample(colors[:3], n_tracks)
        markers_track = sample(markers[4:8], n_tracks)

        # 绘制基础地图(极坐标系)
        fig = plt.figure(figsize=[16, 9], dpi=120)
        fig.suptitle('Radar')

        ax0 = plt.subplot(121, projection="polar")
        ax0.set_theta_zero_location('N')  # 'E', 'N'
        ax0.set_theta_direction(1)  # anti-clockwise
        ax0.set_rmin(10)
        ax0.set_rmax(100000)
        ax0.set_rticks(np.arange(-50000, 50000, 3000))
        ax0.tick_params(labelsize=6)
        ax0.set_title('polar')

        # 极坐标绘制雷达扫描指针
        bar = ax0.bar(0, 50000, width=0.2, alpha=0.3, color='red', label='Radar scan')

        ax1 = plt.subplot(122)
        ax1.set_xticks(np.arange(-50000, 50000, 10000))
        ax1.set_yticks(np.arange(-50000, 50000, 10000))

        # 绘制坐标轴
        # ax1.axis()
        ax1.axhline(y=0, linestyle="-", linewidth=1.5, c="green")
        ax1.axvline(x=0, linestyle="-", linewidth=1.5, c="green")

        # 绘制坐标轴箭头
        # ax1.arrow(x=0, y=0, dx=50000, dy=0, width=1.5, fc='red', ec='blue', alpha=0.3)

        ax1.tick_params(labelsize=7)
        ax1.set_title('cartesian')

        free_noise_legended = False
        isolated_noise_legended = False
        related_legended = False

        cycle_noise_dots = []
        cycle_noise_txts = []
        for cycle, plots_state in tqdm(plots_state_dict):
            if len(cycle_noise_dots) > 0:
                for noise_dot, txt in zip(cycle_noise_dots, cycle_noise_txts):
                    noise_dot.remove()
                    txt.remove()
                cycle_noise_dots = []
                cycle_noise_txts = []

            for k, plot_obj in enumerate(plots_state):
                if plot_obj.state_ == 0:  # 自由点迹(噪声)
                    state = PlotStates[0]
                    marker = markers_noise[0]
                    color = colors_noise[0]
                    label = '$FreePlot(noise)$'
                elif plot_obj.state_ == 1:  # 相关点迹
                    state = PlotStates[1]
                    marker = markers_track[plot_obj.correlated_track_id_]
                    color = colors_track[plot_obj.correlated_track_id_]
                    label = '$Track{:d}$'.format(plot_obj.correlated_track_id_)
                elif plot_obj.state_ == 2:  # 孤立点迹(噪声)
                    state = PlotStates[2]
                    marker = markers_noise[1]
                    color = colors_noise[1]
                    label = '$IsolatedPlot(noise)$'

                # 笛卡尔坐标
                x, y = plot_obj.x_, plot_obj.y_

                # 计算极径
                r = np.sqrt(x * x + y * y)

                # 计算极角
                theta = np.arctan2(y, x)
                theta = theta if theta >= 0.0 else theta + np.pi * 2.0

                # 绘制极坐标点迹
                if state == 'Related':
                    type0 = ax0.scatter(theta, r, c=color, marker=marker, s=5)
                    if cycle == track_init_cycle:
                        txt = 'Track' + str(plot_obj.correlated_track_id_)
                        ax0.text(theta, r, txt, fontsize=10)

                    if cycle == track_init_cycle or (cycle + 1) % 10 == 0:
                        ax0.text(theta, r, str(cycle + 1), fontsize=8)
                elif state == 'Free' or state == 'Isolated':
                    type1 = ax0.scatter(theta, r, c=color, marker=marker)
                    cycle_noise_dots.append(type1)  # 记录当前扫描周期的噪声点

                    type1_txt = ax0.text(theta, r, str(cycle + 1), fontsize=8)
                    cycle_noise_txts.append(type1_txt)  # 记录当前扫描周期的噪声点标签

                # 绘制笛卡尔坐标
                ax1.scatter(x, y, c=color, marker=marker, s=5)

                if state == 'Related' and cycle == track_init_cycle:
                    txt = 'Track' + str(plot_obj.correlated_track_id_)
                    ax1.text(x, y, txt, fontsize=10)

                if state == 'Related':
                    if cycle == track_init_cycle or (cycle + 1) % 10 == 0:
                        ax1.text(x, y, str(cycle + 1), fontsize=8)
                elif state == 'Free' or state == 'Isolated':
                    ax1.text(x, y, str(cycle + 1), fontsize=8)

            if cycle == track_init_cycle:
                ax0.legend((type0, type1), (u'Track', u'Noise'), loc=2)

            # ----- 绘制雷达扫描指针
            bar.remove()
            bar = ax0.bar(cycle * radians_per_move, 50000,
                          width=0.2,
                          alpha=0.3,
                          color='red',
                          label='Radar scan')

            ## ----- 暂停: 动态展示当前雷达扫描周期
            plt.pause(pause_time)

            ## ----- 存放每一个cycle的图
            if is_save:
                frame_f_path = './{:05d}.jpg'.format(cycle)
                plt.savefig(frame_f_path)
            # print('Cycle {:d} done.'.format(cycle + 1))

    # 调用绘图
    # draw_correlation(is_save=True)

    # ---------- 格式转换: *.jpg ——> .mp4 ——> .gif
    if is_convert:
        out_video_path = './scan.mp4'
        cmd_str = 'ffmpeg -f image2 -r 6 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}' \
            .format('.', out_video_path)
        os.system(cmd_str)

        out_gif_path = './scan.gif'
        converter = Video2GifConverter(out_video_path, out_gif_path)
        converter.convert()

    # ----- 清空jpg文件
    if len([x for x in os.listdir('./') if x.endswith('.jpg')]) > 0:
        cmd_str = 'del *.jpg'
        os.system(cmd_str)

    # plt.show()

    ## 分步骤绘制算法过程
    draw_slide_window(track=tracks[0], cycle_time=cycle_time)
    is_convert = True


def draw_slide_window(track, cycle_time, padding=150, is_convert=True):
    """
    先不考虑噪声
    :param track:
    :param cycle_time:
    :param padding:
    :param is_convert:
    :return:
    """

    def get_window(arr, start, win_size):
        return arr[start: start + win_size]

    def draw_track_init(track, cycle_time, method=0, is_save=True):
        """
        :param track:
        :param cycle_time:
        :param method:
        :param is_save:
        :return:
        """
        # ----- 绘图参数
        pause_time = 1.5

        # 超参数设定
        m, n = 4, 3
        v0 = 340.0
        txt_padding = (padding // 10) + 10
        v_min = 0.1 * v0
        v_max = 2.5 * v0
        a_max = 20.0
        angle_max = 7.0
        sigma_s = 160.0

        plot_locs = [[plot.x_, plot.y_] for plot in track.plots_]
        plot_locs = np.array(plot_locs, dtype=np.float32)

        ## ---------- plotting
        fig = plt.figure(figsize=[18, 6], dpi=100)
        fig.suptitle('Radar cartesian coordinate system')
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)
        ax0.set_title('Sliding Window')
        ax1.set_title('Track initialization: direct method')

        x, y = plot_locs[:, 0], plot_locs[:, 1]
        # ax0.scatter(x, y, c='b', marker='>', s=5)
        # plt.show()

        # 开启交互模式
        # plt.ion()

        # ---------- cycle帧计数
        fr_cnt = 0

        # ---------- 滑窗过程
        win_size = 6
        for i in range(len(plot_locs) - win_size + 1):
            # 取滑窗
            window = get_window(plot_locs, i, win_size)
            window = np.array(window, dtype=np.float32)
            # print(window)
            win_x = window[:, 0]
            win_y = window[:, 1]

            # 计算合适的txt_padding
            txt_padding = (max(win_y) - min(win_y)) * 0.05

            # ----- 绘制已经出现的点迹
            ax0.scatter(win_x, win_y, c='b', marker='>', s=5)

            # ----- 处理左图
            # 计算窗口尺寸
            x_min = min(win_x) - padding
            x_max = max(win_x) + padding
            y_min = min(win_y) - padding
            y_max = max(win_y) + padding
            # x_center, y_center = int((x_min + x_max) * 0.5), int((y_min + y_max) * 0.5)

            rect0 = plt.Rectangle(xy=(x_min, y_min),
                                  width=x_max - x_min,
                                  height=y_max - y_min,
                                  edgecolor='y',
                                  fill=False,
                                  linewidth=2)
            ax0.add_patch(rect0)

            # ----- 处理右图
            scatter = ax1.scatter(win_x, win_y, c='b', marker='>', s=25)

            # 遍历窗口每隔点迹: 绘制运动状态(m/n滑窗判定)
            if method == 0:
                ax1.set_title('Track initialization: direct method')
            elif method == 1:
                ax1.set_title('Track initialization: logical method')

            n_pass = 0
            for j in range(2, len(window)):
                idx = i + j
                plot_obj_pre = track.plots_[idx - 1]
                plot_obj_cur = track.plots_[idx]
                plot_obj_nex = track.plots_[idx + 1]
                # print(plot_obj_cur)

                # ----- 绘制运动信息
                x_loc, y_loc = plot_obj_cur.x_, plot_obj_cur.y_
                veloc = plot_obj_cur.v_
                acceleration = plot_obj_cur.a_
                heading_deflection = plot_obj_cur.heading_

                # 点迹运动状态
                assert (x_loc == win_x[j] and y_loc == win_y[j])

                # txt_y_pos = y_loc - txt_padding
                # txt_x_pos = x_loc + txt_padding
                # ax1.text(txt_x_pos, txt_y_pos,
                #          str('pos: [{:d}, {:d}]'.format(int(x_loc), int(y_loc))),
                #          fontsize=10)
                # txt_y_pos -= txt_padding
                # ax1.text(txt_x_pos, txt_y_pos,
                #          str('v: {:.3f}m/s'.format(veloc)),
                #          fontsize=10)
                # txt_y_pos -= txt_padding
                # ax1.text(txt_x_pos, txt_y_pos,
                #          str('a: {:.3f}m/s²'.format(acceleration)),
                #          fontsize=10)
                # txt_y_pos -= txt_padding
                # ax1.text(txt_x_pos, txt_y_pos,
                #          str('h: {:.3f}°'.format(heading_deflection)),
                #          fontsize=10)

                if method == 0:
                    # 直接法判定
                    if veloc >= v_min and \
                            veloc <= v_max and \
                            acceleration <= a_max and \
                            heading_deflection < angle_max:

                        txt_y_pos -= txt_padding
                        ax1.text(txt_x_pos, txt_y_pos,
                                 str('pass: True'),
                                 fontsize=10)

                        n_pass += 1
                    else:
                        txt_y_pos -= txt_padding
                        ax1.text(txt_x_pos, txt_y_pos,
                                 str('pass: False'),
                                 fontsize=10)
                elif method == 1:
                    # --------- 逻辑法判定
                    # ----- 起始波门判定
                    if start_gate_check(cycle_time,
                                        [plot_obj_pre.x_, plot_obj_pre.y_],
                                        [plot_obj_cur.x_, plot_obj_cur.y_],
                                        v0):
                        # ----- 计算初始波门(环形波门的两个半径)
                        r_min = v0 * cycle_time * 0.5  # 小半径
                        r_max = v0 * cycle_time * 1.0  # 大半径

                        # 绘制起始波门(圆形/矩形): 用矩形看起来更舒服点
                        rect_max = plt.Rectangle((plot_obj_pre.x_ - r_max, plot_obj_pre.y_ - r_max),
                                                 width=r_max * 2,
                                                 height=r_max * 2,
                                                 edgecolor='c',
                                                 fill=False,
                                                 linewidth=2)
                        rect_min = plt.Rectangle((plot_obj_pre.x_ - r_min, plot_obj_pre.y_ - r_min),
                                                 width=r_min * 2,
                                                 height=r_min * 2,
                                                 edgecolor='c',
                                                 fill=False,
                                                 linewidth=2)

                        # -- 绘制起始波门标签
                        txt_start_gate = ax1.text(plot_obj_pre.x_ - r_max + txt_padding,
                                                  plot_obj_pre.y_ - r_max + txt_padding,
                                                  str('StartGate'))

                        # 绘制点迹标签
                        txt_pre_plot = ax1.text(plot_obj_pre.x_, plot_obj_pre.y_, str('PrePlot'))
                        txt_cur_plot = ax1.text(plot_obj_cur.x_, plot_obj_cur.y_, str('CurPlot'))
                        txt_nex_plot = ax1.text(plot_obj_nex.x_, plot_obj_nex.y_, str('NexPlot'))

                        # cc_max = plt.Circle((plot_obj_pre.x_, plot_obj_pre.y_), radius=r_max, color='g', fill=False)
                        # cc_min = plt.Circle((plot_obj_pre.x_, plot_obj_pre.y_), radius=r_min, color='g', fill=False)
                        # ax1.add_patch(cc_max)
                        # ax1.add_patch(cc_min)
                        ax1.add_patch(rect_max)
                        ax1.add_patch(rect_min)

                        ## ----- 显示绘制结果
                        plt.pause(pause_time)

                        ## 存图
                        if is_save:
                            fr_cnt += 1

                            frame_f_path = './{:05d}.jpg'.format(fr_cnt)
                            plt.savefig(frame_f_path)

                        rect_max.remove()
                        rect_min.remove()
                        txt_start_gate.remove()

                        # ----- 相关波门判定
                        if relate_gate_check(cycle_time,
                                             plot_obj_cur.v_,
                                             plot_obj_cur.a_,
                                             [plot_obj_pre.x_, plot_obj_pre.y_],
                                             [plot_obj_cur.x_, plot_obj_cur.y_],
                                             [plot_obj_nex.x_, plot_obj_nex.y_],
                                             sigma_s=sigma_s):
                            n_pass += 1

                            # --- 绘制相关波门的相关判定过程
                            # 预测位移值
                            s = plot_obj_cur.v_ * cycle_time + 0.5 * plot_obj_cur.a_ * cycle_time * cycle_time

                            # 计算(直线)外推点
                            x_extra, y_extra = extrapolate_plot([plot_obj_pre.x_, plot_obj_pre.y_],
                                                                [plot_obj_cur.x_, plot_obj_cur.y_],
                                                                s)

                            # -- 绘制外推预测点迹(用虚线箭头)
                            arrow_extra = ax1.arrow(plot_obj_cur.x_, plot_obj_cur.y_,
                                                    x_extra - plot_obj_cur.x_, y_extra - plot_obj_cur.y_,
                                                    width=5, ls='--')

                            # -- 绘制外推点迹
                            dot_extra = ax1.scatter(int(x_extra), int(y_extra), c='m', marker='*', s=60)

                            # ## -- pausing
                            # plt.pause(pause_time)

                            # ## 存图
                            # if is_save:
                            #     fr_cnt += 1

                            #     frame_f_path = './{:05d}.jpg'.format(fr_cnt)
                            #     plt.savefig(frame_f_path)

                            # -- 绘制外推点迹标签
                            dot_extra_txt = ax1.text(x_extra, y_extra, str('ExtraPlot'))

                            ## --- pausing
                            plt.pause(pause_time)

                            ## 存图
                            if is_save:
                                fr_cnt += 1

                                frame_f_path = './{:05d}.jpg'.format(fr_cnt)
                                plt.savefig(frame_f_path)

                            # 绘制相关波门
                            rect_relate = plt.Rectangle((x_extra - sigma_s, y_extra - sigma_s),
                                                        width=sigma_s * 2,
                                                        height=sigma_s * 2,
                                                        edgecolor='c',
                                                        fill=False,
                                                        linewidth=2)
                            ax1.add_patch(rect_relate)

                            # 绘制相关波门标签
                            txt_relate_gate = ax1.text(x_extra - sigma_s + txt_padding,
                                                       y_extra - sigma_s + txt_padding,
                                                       str('RelateGate'))

                            ## --- pausing
                            plt.pause(pause_time)

                            ## 存图
                            if is_save:
                                fr_cnt += 1

                                frame_f_path = './{:05d}.jpg'.format(fr_cnt)
                                plt.savefig(frame_f_path)

                            # 清除外推点迹
                            dot_extra.remove()
                            arrow_extra.remove()
                            rect_relate.remove()
                            dot_extra_txt.remove()
                            txt_relate_gate.remove()
                            txt_pre_plot.remove()
                            txt_cur_plot.remove()
                            txt_nex_plot.remove()

            if n_pass >= n:
                if method == 0:
                    ax1.set_title('Track starting: direct method({:d}/{:d} sliding window) succeed.'
                                  .format(n, m))
                elif method == 1:
                    ax1.set_title('Track starting: logic method({:d}/{:d} sliding window) succeed.'
                                  .format(n, m))
            else:
                if method == 0:
                    ax1.set_title('Track starting: direct method({:d}/{:d} sliding window) failed.'
                                  .format(n, m))
                elif method == 1:
                    ax1.set_title('Track starting: logic method({:d}/{:d} sliding window) failed.'
                                  .format(n, m))

            ## --- pausing
            plt.pause(pause_time)

            ## 存图
            if is_save:
                fr_cnt += 1

                frame_f_path = './{:05d}.jpg'.format(fr_cnt)
                plt.savefig(frame_f_path)

            # ---------- 后处理
            # 清除上一个window的对象
            # patch.set_visible(False)
            rect0.remove()
            scatter.remove()
            ax1.cla()

            # plt.ioff()

        # plt.show()

    # ----------

    # print(plot_locs)
    # ---------- 绘制算法过程
    draw_track_init(track, cycle_time, method=1, is_save=True)

    # ---------- 格式转换: *.jpg ——> .mp4 ——> .gif
    if is_convert:
        if method == 0:
            out_video_path = './slide_window_track_init_{}.mp4'.format('direct')
        elif method == 1:
            out_video_path = './slide_window_track_init_{}.mp4'.format('logic')
        cmd_str = 'ffmpeg -f image2 -r 1 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}' \
            .format('.', out_video_path)
        os.system(cmd_str)

        out_gif_path = out_video_path.replace('.mp4', '.gif')
        converter = Video2GifConverter(out_video_path, out_gif_path)
        converter.convert()

    # ----- 清空jpg文件
    if len([x for x in os.listdir('./') if x.endswith('.jpg')]) > 0:
        cmd_str = 'del *.jpg'
        os.system(cmd_str)


## ---------- Algorithm

# 最近邻(NN)点-航相关算法
def nn_plot_track_correlate(plots_per_cycle, cycle_time,
                            track_init_method=0,
                            σ_s=500, λ=3):
    """
    TODO: 点迹状态(plot state)记录与更新
    :param plots_per_cycle:
    :param cycle_time:
    :param track_init_method:
    :param σ_s:
    :param λ:
    :return:
    """
    n_cycles = plots_per_cycle.shape[0]
    print('Total {:d} radar cycles.'.format(n_cycles))

    # 当前扫描与前次扫面的点迹进行NN匹配: 局部贪心匹配, 计算代价矩阵
    # TODO: 进行匈牙利匹配替换NN匹配...

    if track_init_method == 0:  # 直观法
        succeed, tracks = direct_method_with_bkg(plots_per_cycle,
                                                 cycle_time,
                                                 v_min=150, v_max=500,  # < 2M
                                                 a_max=50, angle_max=10,  # 军机7°/s
                                                 m=3, n=4)
    elif track_init_method == 1:  # 逻辑法
        succeed, tracks = logic_method_with_bkg(plots_per_cycle,
                                                cycle_time,
                                                sigma_s=160,
                                                m=3, n=4)
    elif track_init_method == 2:  # 修正逻辑法
        succeed, tracks = corrected_logic_method_with_bkg(plots_per_cycle,
                                                          cycle_time,
                                                          sigma_s=160, sigma_a=10,
                                                          m=3, n=4)

    if succeed:
        M = len(tracks)
        print('{:d} tracks initialization succeeded.'.format(M))

        # ----------  构建航迹起始阶段的点迹信息字典
        init_phase_plots_state_dict = defaultdict(list)
        last_cycle = max([plot.cycle_ for track in tracks for plot in track.plots_])
        for track in tracks:
            for plot in track.plots_:
                init_phase_plots_state_dict[plot.cycle_].append(plot)

        for cycle, cycle_plots in enumerate(plots_per_cycle):
            if cycle > last_cycle:
                continue

            # 构建已经注册的plot坐标
            registered_plots_ = [[plot.x_, plot.y_] for track in tracks
                                 for plot in track.plots_ if plot.cycle_ == cycle]

            # 遍历当前周期的所有点迹
            for plot in cycle_plots:
                if plot.tolist() not in registered_plots_:
                    plot_obj = Plot(cycle, plot[0], plot[1], -1, -1, -1)
                    plot_obj.state_ = 0  # '自由点迹'
                    plot_obj.correlated_track_id_ = -1

                    init_phase_plots_state_dict[cycle].append(plot_obj)

        # ---------- 航迹起始成功后, 点航相关过程
        # 获取下一个扫描cycle编号
        last_cycle = max(
            [plot.cycle_ for track in tracks for plot in track.plots_])
        start_cycle = last_cycle + 1
        print('Start correlation from cycle {:d}...'.format(start_cycle))
        for track in tracks:
            track.start_correlate_cycle_ = start_cycle

        # ---------- 主循环: 遍历接下来的所有cycles
        terminate_list = []
        correlate_phase_plots_state_dict = OrderedDict()

        for i in range(start_cycle, n_cycles):
            # 遍历下次扫描出现的所有点迹
            cycle_plots = plots_per_cycle[i]
            # print(cycle_plots)

            # 构建当前cycle的plot对象
            cycle_plot_objs = []

            # -----计算马氏距离代价矩阵
            N = cycle_plots.shape[0]
            # cost_mat = np.zeros((M, N), dtype=np.float32)  # 用于点-航匹配

            for track in tracks:
                # print('Processing track {:d}.'.format(track.id_))
                if track.id_ in terminate_list:
                    track.state_ = 4  # 'Terminated'
                    continue

                # 构建预测点迹对象: 跟last_cycle的plot保持一致
                plot_pred = get_predict_plot(track, cycle_time)

                # 计算候选观测点迹
                can_plot_objs = get_candidate_plot_objs(cycle_time, track, plot_pred, cycle_plots, σ_s)
                if len(can_plot_objs) == 0:  # 如果没有候选点迹落入该track的相关(跟踪)波门
                    print("Track {:d} has zero observation plots within it's relating gate."
                          .format(track.id_))
                    track.quality_counter_ -= 1

                    if track.quality_counter_ <= 0:
                        print('Track {:d} is to be terminated.'.format(track.id_))
                        terminate_list.append(track.id_)

                    continue

                # 计算残差的协方差矩阵
                cov_mat = compute_cov_mat(plot_pred, can_plot_objs)

                # --- 计算所有候选点迹的马氏距离
                ma_dists = []
                for can_plot_obj in can_plot_objs:
                    ma_dist = compute_ma_dist(cov_mat, can_plot_obj, plot_pred)
                    ma_dists.append(ma_dist)

                # ----- 点迹-航迹相关判定法则
                # ----- 判定法则
                if len(can_plot_objs) == 1:
                    # 取观测值
                    obs_plot_obj = can_plot_objs[0]

                    # 计算该点迹与其他航迹的距离
                    other_tracks = [track_o for track_o in tracks if track_o.id_ != track.id_]
                    other_ma_dists = []
                    for track_o in other_tracks:  #
                        # 构建预测点迹对象: 跟last_cycle的plot保持一致
                        plot_pred_o = get_predict_plot(track_o, cycle_time)

                        # 计算候选观测点迹
                        can_plot_objs_o = get_candidate_plot_objs(cycle_time, track_o, plot_pred_o, cycle_plots, σ_s)

                        if len(can_plot_objs_o) == 0:  # 如果没有候选点迹落入该track的相关(跟踪)波门
                            other_ma_dists.append(np.inf)
                        else:
                            # 计算残差的协方差矩阵
                            cov_mat_o = compute_cov_mat(plot_pred_o, can_plot_objs_o)

                            # --- 计算马氏距离
                            ma_dist_o = compute_ma_dist(cov_mat_o, obs_plot_obj, plot_pred_o)
                            other_ma_dists.append(ma_dist_o)

                            # # 判断该点迹是否同时落入其他航迹
                            # if is_plot_in_relate_gate([obs_plot_obj.x_, obs_plot_obj.y_], plot_pred_o, σ_s):
                            #     pass
                    # print(other_ma_dists)

                    if ma_dist <= λ * min(other_ma_dists):
                        # 点迹-航迹直接相关
                        track.add_plot(obs_plot_obj)

                        # 更新点迹属性
                        obs_plot_obj.state_ = 1  # 'Related'
                        obs_plot_obj.correlated_track_id_ = track.id_

                    else:
                        # 更新点迹属性
                        obs_plot_obj.state_ = 2  # 'Isolated'
                        obs_plot_obj.correlated_track_id_ = -1

                    # 添加当前cycle的点迹对象
                    cycle_plot_objs.append(obs_plot_obj)

                elif len(can_plot_objs) > 1:
                    # NN点航关联
                    min_ma_dist = min(ma_dists)
                    min_idx = ma_dists.index(min_ma_dist)
                    obs_plot_obj = can_plot_objs[min_idx]
                    track.add_plot(obs_plot_obj)

                    # 更新点迹属性
                    obs_plot_obj.state_ = 1  # 'Related'
                    obs_plot_obj.correlated_track_id_ = track.id_

                    # 添加当前cycle的点迹对象
                    cycle_plot_objs.append(obs_plot_obj)

            # ---------- 处理剩余的点迹
            registered_plots_ = [[plot_obj.x_, plot_obj.y_]
                                 for plot_obj in cycle_plot_objs]
            # print(registered_plots_)
            for plot in cycle_plots:
                if plot.tolist() not in registered_plots_:
                    plot_obj = Plot(start_cycle, plot[0], plot[1], -1, -1, -1)
                    plot_obj.state_ = 0  # '自由点迹'
                    plot_obj.correlated_track_id_ = -1

                    cycle_plot_objs.append(plot_obj)

            # 注册点-航关联信息
            correlate_phase_plots_state_dict[i] = cycle_plot_objs

            # ----------

            # logging
            for track in tracks:
                print('Track {:d} has {:d} plots correlated @cycle{:d}.'.format(track.id_, len(track.plots_), i))
            print('Cycle {:d} correlation done.\n'.format(i + 1))

        # ---------- 对完成所有cycle点迹-航迹关联的航迹进行动态可视化
        print('Start visualization...')
        draw_plot_track_correspondence(cycle_time,
                                       plots_per_cycle,
                                       tracks,
                                       init_phase_plots_state_dict,
                                       correlate_phase_plots_state_dict,
                                       is_convert=True)

    else:
        print('Track initialization failed.')


def test_nn_plot_track_correlate(plots_f_path):
    """
    :param plots_f_path:
    :return:
    """
    # 加载tracks文件
    if not os.path.isfile(plots_f_path):
        print('[Err]: invalid file path.')
        return
    if plots_f_path.endswith('.npy'):
        plots_per_cycle = np.load(plots_f_path, allow_pickle=True)
    elif plots_f_path.endswith('.txt'):
        pass
    else:
        print('[Err]: invalid tracks file format.')
        return

    # 雷达扫描周期(s)
    cycle_time = int(plots_f_path.split('_')[-1].split('.')[0][:-1])

    # ---------- 点航相关
    nn_plot_track_correlate(plots_per_cycle, cycle_time, track_init_method=2)
    # ----------


if __name__ == '__main__':
    test_nn_plot_track_correlate(plots_f_path='./plots_in_each_cycle_1s.npy')
