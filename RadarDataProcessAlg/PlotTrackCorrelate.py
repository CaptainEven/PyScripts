# encoding=utf-8

import math
import os
from collections import defaultdict, OrderedDict
from random import sample

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from Mp4ToGif import Video2GifConverter
from TrackInit import colors
from TrackInit import direct_method_with_bkg, logic_method_with_bkg, corrected_logic_method_with_bkg
from TrackInit import extrapolate_plot, Plot, PlotStates
from TrackInit import start_gate_check, relate_gate_check

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


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
        val = np.dot(res_vector.T, np.linalg.inv(cov_mat)).dot(res_vector)
        val = val if val >= 0.0 else -val
        ma_dist = math.sqrt(val)

    return ma_dist


def pseudo_xy_in_polar(r, theta):
    """
    0(360) point to North
    :param r:
    :param theta:
    :return:
    """
    if theta >= 0 and theta < np.pi * 0.5:  # 第二象限
        angle = np.pi * 0.5 - theta
        x = -r * math.cos(angle)
        y = r * math.sin(angle)
    elif theta >= np.pi * 0.5 and theta < np.pi:  # 第三象限
        angle = theta - np.pi * 0.5
        x = -r * math.cos(angle)
        y = -r * math.sin(angle)
    elif theta >= np.pi and theta < np.pi * 1.5:  # 第四象限
        angle = np.pi * 1.5 - theta
        x = r * math.cos(angle)
        y = -r * math.sin(angle)
    else:  # 第一象限
        angle = theta - np.pi * 1.5
        x = r * math.cos(angle)
        y = r * math.sin(angle)

    return x, y


def cart_to_polar(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    # 计算极径
    r = np.sqrt(x * x + y * y)

    # 计算极角
    theta = np.arctan2(y, x)
    theta = theta if theta >= 0.0 else theta + np.pi * 2.0

    return r, theta


## ---------- visualization
def draw_plot_track_correspondence(cycle_time,
                                   plots_per_cycle,
                                   tracks,
                                   init_phase_plots_state_dict,
                                   correlate_phase_plots_state_dict,
                                   is_save=True,
                                   is_convert=False):
    """
    可视化点迹-行迹关联
    :param cycle_time:
    :param plots_per_cycle:
    :param tracks:
    :param init_phase_plots_state_dict:
    :param correlate_phase_plots_state_dict:
    :param is_save:
    :param is_convert:
    :return:
    """
    # ---------- 雷达扫描绘图参数
    pause_time = 0.01  # show a frame per 0.05s
    n_moves_per_cycle = cycle_time / pause_time  # 20 moves per cycle
    degrees_per_move = 360.0 / n_moves_per_cycle
    radians_per_move = degrees_per_move / math.pi

    # ---------- 数据汇总
    track_init_cycle = max([track.init_cycle_ for track in tracks])

    # 将两个阶段的plot对象信息字典合并
    plot_objs_per_cycle = defaultdict(list)
    for (k, v) in init_phase_plots_state_dict.items():
        plot_objs_per_cycle[k].extend(v)
    for (k, v) in correlate_phase_plots_state_dict.items():
        plot_objs_per_cycle[k].extend(correlate_phase_plots_state_dict[k])
    # print(plots_state_dict)

    # 升序排列
    plot_objs_per_cycle = sorted(plot_objs_per_cycle.items(),
                                 key=lambda x: x[0], reverse=False)

    def draw_correlation(is_save=True):
        """
        :param is_save:
        :return:
        """
        v0 = 340
        min_ratio = 0.5
        max_ratio = 1.5
        fr_cnt = 0  # 存帧计数

        # ---------- 绘图
        marker_size = 20
        txt_size = 15
        col_w = 0.205

        n_tracks = len(tracks)
        n_sample = n_tracks + len(PlotStates)

        colors_noise = 'y'
        markers_noise = '*'
        colors_track = sample(colors[:len(tracks)], n_tracks)
        markers_track = 'o'  # sample(markers[4:8], n_tracks)

        font_dict = {'family': 'SimHei',
                     'style': 'normal',
                     'weight': 'normal',
                     'color': 'yellow',
                     'size': txt_size
                     }
        
        # bkg = plt.imread("./china.png")

        # 绘制基础地图(极坐标系)
        fig = plt.figure(figsize=[18, 9], dpi=100)
        fig.patch.set_facecolor('black')
        # fig.patch.set_alpha(0.95)
        fig.suptitle('雷达扫描实时数据', color='white')
        gs = GridSpec(2, 2, figure=fig)

        ax0 = plt.subplot(gs[:, 0], projection="polar")
        # ax0 = plt.subplot(121, projection="polar")
        ax0.patch.set_facecolor('#000000')
        ax0.patch.set_alpha(0.9)
        ax0.set_theta_zero_location('N')  # 'E', 'N'
        ax0.set_theta_direction(1)  # anti-clockwise
        ax0.set_rmin(10)
        ax0.set_rmax(3500)
        ax0.set_rticks(np.arange(-3500, 3500, 500))
        ax0.tick_params(labelsize=12, colors='gold')
        ax0.set_title('雷达极坐标', color='white')
        # ax0.imshow(bkg)

        # 极坐标绘制雷达扫描指针
        bar = ax0.bar(0, 3000, width=0.35, alpha=0.3, color='green', label='Radar scan')

        ax1 = plt.subplot(gs[0, 1])
        ax1.axis('tight')
        ax1.axis('off')
        ax1.set_title('航迹0关联点迹', color='white')
        ax2 = plt.subplot(gs[1, 1])
        ax2.axis('tight')
        ax2.axis('off')
        ax2.set_title('航迹1关联点迹', color='white')
        axes_trs = [ax1, ax2]
        col_labels = ['点迹编号', '方位角(°)', '距离(m)', '速度(m/s)', '加速度(m/s2)', '航角偏转(°)']

        # 构建track的点迹状态数组
        track_stats = np.full((len(tracks), len(plot_objs_per_cycle), len(col_labels)), 0.0, dtype=np.float32)
        # for i, track in enumerate(tracks):
        #     for j, plot in enumerate(track.plots_):
        #         plot.cart_to_polar()
        #         track_stats[i][j][0] = plot.theta_
        #         track_stats[i][j][1] = plot.r_
        #         track_stats[i][j][2] = plot.v_
        #         track_stats[i][j][3] = plot.a_
        #         track_stats[i][j][4] = plot.heading_
        # print(track_stats)

        # 测试表格
        stat_table_tr0 = ax1.table(cellText=track_stats[0], colLabels=col_labels,
                                   loc='center', colWidths=[col_w, col_w, col_w, col_w, col_w, col_w])
        stat_table_tr1 = ax1.table(cellText=track_stats[1], colLabels=col_labels,
                                   loc='center', colWidths=[col_w, col_w, col_w, col_w, col_w, col_w])

        # ax1.subplot(122)
        # ax1.set_xticks(np.arange(-5000, 5000, 1000))
        # ax1.set_yticks(np.arange(-5000, 5000, 1000))

        # # 绘制坐标轴
        # # ax1.axis()
        # ax1.axhline(y=0, linestyle="-", linewidth=1.8, c="green")
        # ax1.axvline(x=0, linestyle="-", linewidth=1.8, c="green")

        # # 绘制坐标轴箭头
        # # ax1.arrow(x=0, y=0, dx=50000, dy=0, width=1.5, fc='red', ec='blue', alpha=0.3)

        # ax1.tick_params(labelsize=7)
        # ax1.set_title('cartesian')

        free_noise_legended = False
        isolated_noise_legended = False
        related_legended = False

        cycle_noise_dots = []
        cycle_noise_txts = []

        cycle_dots_extra = []  # 记录一个cycle的外推点
        cycle_dot_txts_extra = []
        cycle_ccs_relate = []  # 记录一个cycle的相关波门

        # 两层for循环遍历每一个cycle的每一个点迹(关联点迹或噪声点迹)
        for cycle, cycle_plot_objs in tqdm(plot_objs_per_cycle):
            if cycle > 5:
                break

            # if len(cycle_noise_dots) > 0:
            #     for noise_dot, txt in zip(cycle_noise_dots, cycle_noise_txts):
            #         noise_dot.remove()
            #         txt.remove()
            #     cycle_noise_dots = []
            #     cycle_noise_txts = []
            if len(cycle_dots_extra) > 0 and len(cycle_dots_extra) == len(cycle_ccs_relate):
                for dot, dot_txt, circle in zip(cycle_dots_extra, cycle_dot_txts_extra, cycle_ccs_relate):
                    dot.remove()
                    dot_txt.remove()
                    circle.remove()

                cycle_dots_extra = []  # 记录一个cycle的外推点
                cycle_dot_txts_extra = []
                cycle_ccs_relate = []  # 记录一个cycle的相关波门

            for k, plot_obj in enumerate(cycle_plot_objs):
                if plot_obj.state_ == 0:  # 自由点迹(噪声)
                    state = PlotStates[0]
                    marker = markers_noise[0]
                    color = colors_noise
                    label = '$FreePlot(noise)$'
                elif plot_obj.state_ == 1:  # 相关点迹
                    state = PlotStates[1]
                    marker = markers_track  # markers_track[plot_obj.correlated_track_id_]
                    color = colors_track[plot_obj.correlated_track_id_]
                    label = '$Track{:d}$'.format(plot_obj.correlated_track_id_)
                elif plot_obj.state_ == 2:  # 孤立点迹(噪声)
                    state = PlotStates[2]
                    marker = markers_noise[1]
                    color = colors_noise
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
                    if plot_obj.correlated_track_id_ == 0:
                        type0 = ax0.scatter(theta, r, c=color, marker=marker, s=marker_size)
                    elif plot_obj.correlated_track_id_ == 1:
                        type1 = ax0.scatter(theta, r, c=color, marker=marker, s=marker_size)

                    # 绘制航迹标签
                    if cycle == 0:
                        txt = '航迹' + str(plot_obj.correlated_track_id_)
                        ax0.text(theta, r, txt,
                                 fontsize=font_dict['size'],
                                 color=font_dict['color'])

                    # if cycle == track_init_cycle or (cycle + 1) % 10 == 0:
                    #     ax0.text(theta, r, str(cycle + 1),
                    #              fontsize=font_dict['size'],
                    #              color=font_dict['color'])
                    ax0.text(theta, r, str(plot_obj.plot_id_),
                             fontsize=font_dict['size'],
                             color=font_dict['color'])
                elif state == 'Free' or state == 'Isolated':
                    if r < -3000 or r > 3000:
                        continue

                    type2 = ax0.scatter(theta, r, c=color, marker=marker, s=marker_size)
                    cycle_noise_dots.append(type1)  # 记录当前扫描周期的噪声点

                    # type1_txt = ax0.text(theta, r, str(cycle + 1),
                    #                      fontsize=font_dict['size'],
                    #                      color=font_dict['color'])
                    # cycle_noise_txts.append(type1_txt)  # 记录当前扫描周期的噪声点标签

                ## 绘制笛卡尔坐标
                # ax1.scatter(x, y, c=color, marker=marker, s=marker_size)

                if state == 'Related' and cycle == 0:
                    txt = 'Track' + str(plot_obj.correlated_track_id_)
                    # ax1.text(x, y, txt,
                    #          fontsize=font_dict['size'],
                    #          color=font_dict['color'])

                    # 更新表格数据
                    # stat_table_tr0.remove()

                if state == 'Related':
                    # stat_table_tr0.remove()
                    # stat_table_tr1.remove()

                    plot_obj.cart_to_polar()
                    i = plot_obj.correlated_track_id_
                    j = plot_obj.cycle_
                    track_stats[i][j][0] = plot_obj.plot_id_
                    track_stats[i][j][1] = plot_obj.degrees_
                    track_stats[i][j][2] = plot_obj.r_
                    track_stats[i][j][3] = plot_obj.v_
                    track_stats[i][j][4] = plot_obj.a_
                    track_stats[i][j][5] = plot_obj.heading_
                    stat_table_tr0 = axes_trs[i].table(cellText=track_stats[i], colLabels=col_labels,
                                                       loc='center',
                                                       colWidths=[col_w, col_w, col_w, col_w, col_w, col_w])

                    ## 绘制点-航相关过程
                    if cycle > 0:
                        cur_id = tracks[i].plots_.index(plot_obj)
                        cur_plot = plot_obj
                        pre_plot = tracks[i].plots_[cur_id - 1]

                        # 判断是否存在下一个相关点迹
                        if cur_id < len(tracks[i].plots_) - 1:
                            nex_plot = tracks[i].plots_[cur_id + 1]

                        ## 计算初始波门(环形波门的两个半径)
                        r_min = v0 * cycle_time * min_ratio  # 小半径
                        r_max = v0 * cycle_time * max_ratio  # 大半径

                        # 获取极坐标下的x,y坐标
                        x_pre_, y_pre_ = pseudo_xy_in_polar(pre_plot.r_, pre_plot.theta_)
                        cc_max = plt.Circle((x_pre_, y_pre_),
                                            radius=r_max, color='g', fill=False, transform=ax0.transData._b)
                        cc_min = plt.Circle((x_pre_, y_pre_),
                                            radius=r_min, color='g', fill=False, transform=ax0.transData._b)
                        ax0.add_patch(cc_min)
                        ax0.add_patch(cc_max)

                        ## 暂停显示中间步骤
                        plt.pause(pause_time)
                        if is_save:
                            frame_f_path = './{:05d}.png'.format(fr_cnt)
                            plt.savefig(frame_f_path, facecolor='black')
                            fr_cnt += 1

                        cc_min.remove()
                        cc_max.remove()

                        ## ----- 绘制外推点坐标和箭头
                        # 计算外推点坐标
                        # 预测位移值
                        s = cur_plot.v_ * cycle_time + 0.5 * cur_plot.a_ * cycle_time * cycle_time

                        # 计算(直线)外推点
                        x_extra, y_extra = extrapolate_plot([pre_plot.x_, pre_plot.y_], [cur_plot.x_, cur_plot.y_], s)
                        r_extra, t_extra = cart_to_polar(x_extra, y_extra)
                        x_extra_, y_extra_ = pseudo_xy_in_polar(r_extra, t_extra)
                        cur_plot.cart_to_polar()
                        x_cur_, y_cur_ = pseudo_xy_in_polar(cur_plot.r_, cur_plot.theta_)

                        # --- 绘制外推箭头连线
                        arrow_extra = ax0.arrow(x_cur_, y_cur_,
                                                x_extra_ - x_cur_, y_extra_ - y_cur_,
                                                width=2,
                                                ls='--',
                                                color='yellow',
                                                transform=ax0.transData._b)

                        ## 暂停显示中间步骤
                        plt.pause(pause_time)
                        if is_save:
                            frame_f_path = './{:05d}.png'.format(fr_cnt)
                            plt.savefig(frame_f_path, facecolor='black')
                            fr_cnt += 1

                        arrow_extra.remove()

                        # --- 绘制外推点迹
                        dot_extra = ax0.scatter(t_extra, r_extra, c='white', marker='D', s=marker_size)
                        dot_extra_txt = ax0.text(t_extra, r_extra, '外推点和相关波门',
                                                 fontsize=font_dict['size'],
                                                 color=font_dict['color'])

                        ## 暂停显示中间步骤
                        plt.pause(pause_time)
                        if is_save:
                            frame_f_path = './{:05d}.png'.format(fr_cnt)
                            plt.savefig(frame_f_path, facecolor='black')
                            fr_cnt += 1

                        # --- 绘制相关波门
                        if cur_id < len(tracks[i].plots_) - 1:
                            nex_plot.cart_to_polar()
                            x_nex_, y_nex_ = pseudo_xy_in_polar(nex_plot.r_, nex_plot.theta_)

                            cc_relate = plt.Circle((x_nex_, y_nex_),
                                                   radius=r_min, color='g', fill=False, transform=ax0.transData._b)
                            ax0.add_patch(cc_relate)

                        ## 暂停显示中间步骤
                        plt.pause(pause_time)
                        if is_save:
                            frame_f_path = './{:05d}.png'.format(fr_cnt)
                            plt.savefig(frame_f_path, facecolor='black')
                            fr_cnt += 1

                        # dot_extra.remove()
                        # cc_relate.remove()
                        cycle_dots_extra.append(dot_extra)
                        cycle_dot_txts_extra.append(dot_extra_txt)
                        cycle_ccs_relate.append(cc_relate)

                # ----- 绘制雷达扫描指针
                bar.remove()
                bar = ax0.bar(radians_per_move * cycle, 3000,
                              width=0.35,
                              alpha=0.3,
                              color='yellow',
                              label='Radar scan')
                # if state == 'Related':
                #     if cycle == track_init_cycle or (cycle + 1) % 10 == 0:
                #         ax1.text(x, y, str(cycle + 1),
                #                  fontsize=font_dict['size'],
                #                  color=font_dict['color'])
                # elif state == 'Free' or state == 'Isolated':
                #     ax1.text(x, y, str(cycle + 1),
                #              fontsize=font_dict['size'],
                #              color=font_dict['color'])

            if cycle == track_init_cycle:
                ax0.legend((type0, type1, type2), ('关联点迹', '关联点迹', '噪声'), loc=2)

            # # ----- 绘制雷达扫描指针
            # bar.remove()
            # bar = ax0.bar(cycle * radians_per_move, 3000,
            #               width=0.35,
            #               alpha=0.3,
            #               color='yellow',
            #               label='Radar scan')

            # # 测试表格
            # the_table = ax1.table(cellText=track_stats[0], colLabels=col_labels,
            #                       loc='center', colWidths=[0.1, 0.1, 0.1, 0.1, 0.1])

            ## ----- 暂停: 动态展示当前雷达扫描周期
            plt.pause(pause_time)

            ## ----- 存放每一个cycle的图
            if is_save:
                frame_f_path = './{:05d}.png'.format(fr_cnt)
                plt.savefig(frame_f_path, facecolor='black')
                fr_cnt += 1
            # print('Cycle {:d} done.'.format(cycle + 1))

    # 调用绘图
    draw_correlation(is_save=True)

    # ---------- 格式转换: *.jpg ——> .mp4 ——> .gif
    if is_convert:
        jpg_f_list = [x for x in os.listdir('./') if x.endswith('.png')]
        if len(jpg_f_list) > 0:
            out_video_path = './scan.mp4'
            cmd_str = 'ffmpeg -f image2 -r 2 -i {}/%05d.png -b 5000k -c:v mpeg4 {}' \
                .format('.', out_video_path)
            os.system(cmd_str)

            out_gif_path = './scan.gif'
            converter = Video2GifConverter(out_video_path, out_gif_path)
            converter.convert()

    # ----- 清空jpg文件
    if len([x for x in os.listdir('./') if x.endswith('.jpg')]) > 0:
        cmd_str = 'del *.png'
        os.system(cmd_str)

    # plt.show()

    # ## 分步骤绘制算法过程
    # draw_slide_window(track=tracks[0], cycle_time=cycle_time, init_method=1)
    # is_convert = True

    ## 绘制点-航相关
    # draw_plot_track_relate(plots_state_dict, tracks)


def get_window(arr, start, win_size):
    return arr[start: start + win_size]


##### ----- 绘制点航相关算法过程
def draw_plot_track_relate(plots_objs_per_cycle, tracks, σ_s=160):
    """
    :param plots_state_dict_per_cycle:
    :param tracks:
    :param σ_s:
    :return:
    """
    pause_time = 5e-1

    track_init_cycle = max([track.init_cycle_ for track in tracks])
    print('Track init cycle: {:d}.'.format(track_init_cycle))

    # ---------- 绘图
    marker_size = 12
    txt_size = 10

    n_tracks = len(tracks)
    n_sample = n_tracks + len(PlotStates)

    colors_noise = 'gray'
    markers_noise = '*'
    colors_track = sample(colors[:len(tracks)], n_tracks)
    markers_track = 'o'  # sample(markers[4:8], n_tracks)

    # 绘制基础地图(极坐标系)
    fig = plt.figure(figsize=[18, 9], dpi=100)
    fig.suptitle('Radar')
    gs = GridSpec(2, 2, figure=fig)

    ax0 = plt.subplot(gs[:, 0], projection="polar")
    ax0.set_theta_zero_location('N')  # 'E', 'N'
    ax0.set_theta_direction(1)  # anti-clockwise
    ax0.set_rmin(10)
    ax0.set_rmax(100000)
    ax0.set_rticks(np.arange(-50000, 50000, 3000))
    ax0.tick_params(labelsize=6)
    ax0.set_title('polar')

    # 极坐标绘制雷达扫描指针
    bar = ax0.bar(0, 50000, width=0.35, alpha=0.3, color='red', label='Radar scan')

    # ax1 = plt.subplot(122)
    # ax1.set_xticks(np.arange(-30000, 30000, 10000))
    # ax1.set_yticks(np.arange(-30000, 30000, 10000))

    # # 绘制坐标轴
    # # ax1.axis()
    # ax1.axhline(y=0, linestyle="-", linewidth=1.8, c="green")
    # ax1.axvline(x=0, linestyle="-", linewidth=1.8, c="green")

    # 绘制坐标轴箭头
    # ax1.arrow(x=0, y=0, dx=50000, dy=0, width=1.5, fc='red', ec='blue', alpha=0.3)

    # ax1.tick_params(labelsize=7)
    # ax1.set_title('cartesian')

    free_noise_legended = False
    isolated_noise_legended = False
    related_legended = False

    cycle_noise_dots = []
    cycle_noise_txts = []

    track2_plot_marker = 'd'
    track2_plot_color = 'r'
    track3_plot_marker = 'o'
    track3_plot_color = 'b'
    noise_plot_marker = '*'
    noise_plot_color = 'g'

    ## ----- 取滑窗
    win_size = 6
    for i in tqdm(range(len(plots_objs_per_cycle) - win_size + 1)):
        # 取窗口
        window = get_window(plots_objs_per_cycle, i, win_size)
        # print(window)

        if window[0][0] < 12:
            min_x_tr2 = min([x.x_ for cycle, X in window for x in X if x.state_ == 1 and x.correlated_track_id_ == 2])
            max_x_tr2 = max([x.x_ for cycle, X in window for x in X if x.state_ == 1 and x.correlated_track_id_ == 2])
            min_y_tr2 = min([x.y_ for cycle, X in window for x in X if x.state_ == 1 and x.correlated_track_id_ == 2])
            max_y_tr2 = max([x.y_ for cycle, X in window for x in X if x.state_ == 1 and x.correlated_track_id_ == 2])

            min_x_tr3 = min([x.x_ for cycle, X in window for x in X if x.state_ == 1 and x.correlated_track_id_ == 3])
            max_x_tr3 = max([x.x_ for cycle, X in window for x in X if x.state_ == 1 and x.correlated_track_id_ == 3])
            min_y_tr3 = min([x.y_ for cycle, X in window for x in X if x.state_ == 1 and x.correlated_track_id_ == 3])
            max_y_tr3 = max([x.y_ for cycle, X in window for x in X if x.state_ == 1 and x.correlated_track_id_ == 3])

            ax_tr2 = plt.subplot(gs[0, 1])
            ax_tr2.set_xticks(np.arange(min_x_tr2, max_x_tr2, 10000))
            ax_tr2.set_yticks(np.arange(min_y_tr2, max_y_tr2, 10000))
            ax_tr2.tick_params(labelsize=7)
            ax_tr2.set_title('cartesian')

            ax_tr3 = plt.subplot(gs[1, 1])
            ax_tr3.set_xticks(np.arange(min_x_tr3, max_x_tr3, 10000))
            ax_tr3.set_yticks(np.arange(min_y_tr3, max_y_tr3, 10000))
            ax_tr3.tick_params(labelsize=7)
            ax_tr3.set_title('cartesian')

            # 绘制坐标轴箭头
            # ax1.arrow(x=0, y=0, dx=50000, dy=0, width=1.5, fc='red', ec='blue', alpha=0.3)

            ax_tr2.tick_params(labelsize=7)
            ax_tr2.set_title('cartesian')

            # 取我们关心的track对应的点迹和噪声点迹
            track2_plot_objs = []
            track3_plot_objs = []
            noise_plots = []
            for cycle, cycle_plot_objs in window:
                for plot_obj in cycle_plot_objs:
                    if plot_obj.state_ == 1:
                        if plot_obj.correlated_track_id_ == 2:  # 我们感兴趣的相关点迹
                            track2_plot_objs.append(plot_obj)

                            # 绘制点迹
                            ax_tr2.scatter(plot_obj.x_, plot_obj.y_,
                                           c=track2_plot_color, marker=track2_plot_marker, s=25)
                        elif plot_obj.correlated_track_id_ == 3:
                            track3_plot_objs.append(plot_obj)

                            # 绘制点迹
                            ax_tr3.scatter(plot_obj.x_, plot_obj.y_,
                                           c=track3_plot_color, marker=track3_plot_marker, s=25)
                    elif plot_obj.state_ == 0 or plot_obj.state_ == 2:  # 噪声点迹
                        noise_plots.append(plot_obj)

                        # 绘制噪声点迹
                        if plot_obj.x_ > min_x_tr2 and plot_obj.x_ < max_x_tr2 \
                                and plot_obj.y_ > min_y_tr2 and plot_obj.y_ < max_y_tr2:
                            ax_tr2.scatter(plot_obj.x_, plot_obj.y_,
                                           c=noise_plot_color, marker=noise_plot_marker, s=25)

                        elif plot_obj.x_ > min_x_tr3 and plot_obj.x_ < max_x_tr3 \
                                and plot_obj.y_ > min_y_tr3 and plot_obj.y_ < max_y_tr3:
                            ax_tr3.scatter(plot_obj.x_, plot_obj.y_,
                                           c=noise_plot_color, marker=noise_plot_marker, s=25)

            ## ----- 暂停: 动态展示当前雷达扫描周期
            plt.pause(pause_time)

            ax_tr2.cla()
            ax_tr3.cla()
        else:
            pass
            # 右图将track2, track3绘制到同一个子图里


def draw_slide_window(track, cycle_time, padding=150, is_convert=True, init_method=1):
    """
    先不考虑噪声
    :param track:
    :param cycle_time:
    :param padding:
    :param is_convert:
    :param init_method:
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
        pause_time = 1e-8

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
        for i in tqdm(range(len(plot_locs) - win_size + 1)):
            for line in ax0.lines:
                line.remove()
            for line in ax1.lines:
                line.remove()

            # 取滑窗
            window = get_window(plot_locs, i, win_size)
            window = np.array(window, dtype=np.float32)
            # print(window)
            win_x = window[:, 0]
            win_y = window[:, 1]

            # 计算合适的txt_padding
            txt_padding = (max(win_y) - min(win_y)) * 0.055

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

            # --------- 窗口内遍历
            if method == 0:
                win_edge = len(window)
            elif method == 1 or method == 2:
                win_edge = len(window) - 1

            n_pass = 0
            for j in range(2, win_edge):

                idx = i + j
                plot_obj_pre = track.plots_[idx - 1]
                plot_obj_cur = track.plots_[idx]
                if j == 2:
                    plot_obj_prepre = track.plots_[idx - 2]

                if method == 1 or method == 2:
                    plot_obj_nex = track.plots_[idx + 1]

                # ----- 绘制运动信息
                x_loc, y_loc = plot_obj_cur.x_, plot_obj_cur.y_
                veloc = plot_obj_cur.v_
                acceleration = plot_obj_cur.a_
                heading_deflection = plot_obj_cur.heading_

                # 点迹运动状态
                assert (x_loc == win_x[j] and y_loc == win_y[j])

                if method == 0:
                    # 直接法判定
                    txt_x_pos = x_loc
                    txt_y_pos = y_loc - txt_padding
                    # txt_y_pos = y_loc - txt_padding
                    # txt_x_pos = x_loc + txt_padding
                    # ax1.text(txt_x_pos, txt_y_pos,
                    #          str('pos: [{:d}, {:d}]'.format(int(x_loc), int(y_loc))),
                    #          fontsize=10)
                    # txt_y_pos -= txt_padding

                    ax1.text(txt_x_pos, txt_y_pos,
                             str('v: {:.3f}m/s'.format(veloc)),
                             fontsize=10)

                    txt_y_pos -= txt_padding
                    ax1.text(txt_x_pos, txt_y_pos,
                             str('a: {:.3f}m/s²'.format(acceleration)),
                             fontsize=10)

                    txt_y_pos -= txt_padding
                    ax1.text(txt_x_pos, txt_y_pos,
                             str('h: {:.3f}°'.format(heading_deflection)),
                             fontsize=10)

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
                                                    width=3, ls='--')

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

                                # --- 绘制点迹连线
                            if j == 2:
                                line_prepre_to_pre_0 = mlines.Line2D([plot_obj_prepre.x_, plot_obj_pre.x_],
                                                                     [plot_obj_prepre.y_, plot_obj_pre.y_])
                                ax0.add_line(line_prepre_to_pre_0)

                                line_prepre_to_pre_1 = mlines.Line2D([plot_obj_prepre.x_, plot_obj_pre.x_],
                                                                     [plot_obj_prepre.y_, plot_obj_pre.y_])
                                ax1.add_line(line_prepre_to_pre_1)

                                ## --- pausing
                                plt.pause(pause_time)

                                ## 存图
                                if is_save:
                                    fr_cnt += 1

                                    frame_f_path = './{:05d}.jpg'.format(fr_cnt)
                                    plt.savefig(frame_f_path)

                            line_pre_to_cur_0 = mlines.Line2D([plot_obj_pre.x_, plot_obj_cur.x_],
                                                              [plot_obj_pre.y_, plot_obj_cur.y_])
                            # line_cur_to_nex_0 = mlines.Line2D([plot_obj_cur.x_, plot_obj_nex.x_], [plot_obj_cur.y_, plot_obj_nex.y_])
                            ax0.add_line(line_pre_to_cur_0)
                            # ax0.add_line(line_cur_to_nex_0)

                            line_pre_to_cur_1 = mlines.Line2D([plot_obj_pre.x_, plot_obj_cur.x_],
                                                              [plot_obj_pre.y_, plot_obj_cur.y_])
                            # line_cur_to_nex_1 = mlines.Line2D([plot_obj_cur.x_, plot_obj_nex.x_], [plot_obj_cur.y_, plot_obj_nex.y_])
                            ax1.add_line(line_pre_to_cur_1)
                            # ax1.add_line(line_cur_to_nex_1)

                            ## --- pausing
                            plt.pause(pause_time)

                            ## 存图
                            if is_save:
                                fr_cnt += 1

                                frame_f_path = './{:05d}.jpg'.format(fr_cnt)
                                plt.savefig(frame_f_path)

                            if j == win_edge - 1:
                                line_cur_to_nex_0 = mlines.Line2D([plot_obj_cur.x_, plot_obj_nex.x_],
                                                                  [plot_obj_cur.y_, plot_obj_nex.y_])
                                ax0.add_line(line_cur_to_nex_0)
                                line_cur_to_nex_1 = mlines.Line2D([plot_obj_cur.x_, plot_obj_nex.x_],
                                                                  [plot_obj_cur.y_, plot_obj_nex.y_])
                                ax1.add_line(line_cur_to_nex_1)

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

            if n_pass >= n:  # m/n法则
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

            for line in ax0.lines:
                line.remove()
            for line in ax1.lines:
                line.remove()

            # plt.ioff()

        # plt.show()

    # ----------

    # print(plot_locs)
    # ---------- 绘制算法过程
    draw_track_init(track, cycle_time, method=init_method, is_save=True)

    # ---------- 格式转换: *.jpg ——> .mp4 ——> .gif
    if is_convert:
        if init_method == 0:
            out_video_path = './slide_window_track_init_{}.mp4'.format('direct')
        elif init_method == 1:
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
        last_cycle = max([plot.cycle_ for track in tracks for plot in track.plots_])
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

                        # 更新关联点迹在航迹中的编号
                        obs_plot_obj.plot_id_ = track.plots_.index(obs_plot_obj)

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
                    # ----- NN点航关联
                    min_ma_dist = min(ma_dists)
                    min_idx = ma_dists.index(min_ma_dist)
                    obs_plot_obj = can_plot_objs[min_idx]
                    track.add_plot(obs_plot_obj)

                    # 更新关联点迹在航迹中的编号
                    obs_plot_obj.plot_id_ = track.plots_.index(obs_plot_obj)

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
                print('Track {:d} has {:d} plots correlated @cycle{:d}.'.format(track.id_, len(track.plots_), i + 1))
            print('Cycle {:d} correlation done.\n'.format(i + 1))

        # ---------- 对完成所有cycle点迹-航迹关联的航迹进行动态可视化
        print('Start visualization...')
        draw_plot_track_correspondence(cycle_time,
                                       plots_per_cycle,
                                       tracks,
                                       init_phase_plots_state_dict,
                                       correlate_phase_plots_state_dict,
                                       is_save=True,
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
    cycle_time = int(plots_f_path.split('_')[-2].split('.')[0][:-1])
    print('Cycle time: {:d}s'.format(cycle_time))

    # ---------- 点航相关
    nn_plot_track_correlate(plots_per_cycle, cycle_time, track_init_method=2)
    # ----------


if __name__ == '__main__':
    # test_nn_plot_track_correlate(plots_f_path='./plots_in_each_cycle_1s.npy')

    ## ./2021_03_17_11_04_12_plots_in_each_cycle_1s_10cycle
    test_nn_plot_track_correlate(plots_f_path='./2021_03_17_11_04_12_plots_in_each_cycle_1s_10cycle.npy')
