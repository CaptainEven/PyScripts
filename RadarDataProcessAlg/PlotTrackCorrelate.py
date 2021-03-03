# encoding=utf-8

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm import tqdm
from collections import defaultdict, OrderedDict

from TrackInit import markers, colors
from TrackInit import extrapolate_plot, Track, Plot, PlotStates, relate_gate_check
from TrackInit import direct_method_with_bkg, logic_method_with_bkg, corrected_logic_method_with_bkg


def get_predict_plot(track, cycle_time):
    """
    :param track:
    :param cycle_time:
    :return:
    """
    last_cycle = max([plot.cycle_ for plot in track.plots_])

    # ----- 外推next cycle预测点迹
    # 提取前cycle和当前cycle的点迹
    plot_pre = [plot for plot in track.plots_ if plot.cycle_ == last_cycle - 1][0]
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
    x_extra, y_extra = extrapolate_plot([plot_pre.x_, plot_pre.y_], [plot_cur.x_, plot_cur.y_], s)

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
    plot_pre = [plot for plot in track.plots_ if plot.cycle_ == last_cycle - 1][0]
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
        shift1 = np.array([plot[0], plot[1]]) - np.array([plot_cur.x_, plot_cur.y_])
        shift0 = np.array([plot_cur.x_, plot_cur.y_]) - np.array([plot_pre.x_, plot_pre.y_])

        # 计算速度
        dist0 = np.linalg.norm(shift0, ord=2)
        dist1 = np.linalg.norm(shift1, ord=2)
        v0 = dist0 / cycle_time
        v1 = dist1 / cycle_time

        # 计算加速度
        a = (v1 - v0) / cycle_time

        # 计算航向偏转角
        heading = math.degrees(math.acos(np.dot(shift0, shift1) / (dist0 * dist1 + 1e-6)))

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
        ma_dist = math.sqrt(np.dot(res_vector.T, np.linalg.inv(cov_mat)).dot(res_vector))

    return ma_dist


## ---------- visualization
def draw_plot_track_correspondence(plots_per_cycle, tracks,
                                   init_phase_plots_state_dict, correlate_phase_plots_state_dict):
    """
    可视化点迹-行迹关联
    :param plots_per_cycle:
    :param tracks:
    :param init_phase_plots_state_dict:
    :param correlate_phase_plots_state_dict:
    :return:
    """
    # ---------- 数据汇总
    track_init_cycle = max([track.init_cycle_ for track in tracks])

    ## 按照cycle升序排序
    # init_phase_plots_state_dict = sorted(init_phase_plots_state_dict.items(),
    #                                      key=lambda x: x[0], reverse=False)
    # correlate_phase_plots_state_dict = sorted(correlate_phase_plots_state_dict.items(),
    #                                           key=lambda x: x[0], reverse=False)

    # 将两个阶段的plot对象信息字典合并
    plots_state_dict = defaultdict(list)
    for (k, v) in init_phase_plots_state_dict.items():
        plots_state_dict[k].extend(v)
    for (k, v) in correlate_phase_plots_state_dict.items():
        plots_state_dict[k].extend(correlate_phase_plots_state_dict[k])
    # print(plots_state_dict)

    plots_state_dict = sorted(plots_state_dict.items(), key=lambda x: x[0], reverse=False)
    # for cycle, (plots, (c_id, states)) in enumerate(zip(plots_per_cycle, plots_state_dict)):
    #     assert plots.shape[0] == len(states)
    #     flags = [True for x in plots.tolist() if x in [[plot.x_, plot.y_] for plot in states]]
    #     if False in flags:
    #         print(flags)

    # ---------- 绘图
    n_tracks = len(tracks)
    n_sample = n_tracks + len(PlotStates)

    colors_noise = sample(colors[3:], len(PlotStates))
    markers_noise = sample(markers[:3], len(PlotStates))

    colors_track = sample(colors[:3], n_tracks)
    markers_track = sample(markers[4:8], n_tracks)

    # 绘制基础地图(极坐标系)
    fig = plt.figure(figsize=[16, 8])
    fig.suptitle('Radar')

    ax0 = plt.subplot(121, projection="polar")
    ax0.set_theta_zero_location('E')
    ax0.set_theta_direction(1)  # anti-clockwise
    ax0.set_rmin(10)
    ax0.set_rmax(100000)
    ax0.set_rticks(np.arange(-50000, 50000, 3000))
    ax0.set_title('polar')

    ax1 = plt.subplot(122)
    ax1.set_xticks(np.arange(-50000, 50000, 10000))
    ax1.set_yticks(np.arange(-50000, 50000, 10000))
    ax1.set_title('cartesian')

    free_noise_legended = False
    isolated_noise_legended = False
    related_legended = False
    
    for cycle, plots_state in tqdm(plots_state_dict):
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
                type0 = ax0.scatter(theta, r, c=color, marker=marker)
                if cycle == track_init_cycle or (cycle + 1) % 10 == 0:
                    ax0.text(theta, r, str(cycle + 1))
            elif state == 'Free' or state == 'Isolated':
                type1 = ax0.scatter(theta, r, c=color, marker=marker)
                ax0.text(theta, r, str(cycle + 1))

            # 绘制笛卡尔坐标
            ax1.scatter(x, y, c=color, marker=marker)           

            if state == 'Related' and cycle == track_init_cycle:
                txt = 'Track' + str(plot_obj.correlated_track_id_)
                ax1.text(x, y, txt)

            if state == 'Related':
                if cycle == track_init_cycle or (cycle + 1) % 10 == 0:
                    ax1.text(x, y, str(cycle + 1))
            elif state == 'Free' or state == 'Isolated':
                ax1.text(x, y, str(cycle + 1))
            
            plt.pause(0.1)
        
        if cycle == track_init_cycle:
            ax0.legend((type0, type1), (u'Track', u'Noise'), loc=2)
        # print('Cycle {:d} done.'.format(cycle + 1))

    # plt.legend(loc="upper left")
    plt.show()


# ----------

## 最近邻(NN)点-航相关算法
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

        # ## ---------- 可视化成功起始的航迹
        # for track in tracks:
        #     # print(track)
        #     cycles = [plot.cycle_ for plot in track.plots_]
        #     xs = [plot.x_ for plot in track.plots_]
        #     ys = [plot.y_ for plot in track.plots_]
        #
        #     plots = [[x, y] for x, y in zip(xs, ys)]
        #     plot_plots(plots, cycles)

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
                    print("Track {:d} has zero observation plots within it's relating gate.".format(track.id_))
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
            registered_plots_ = [[plot_obj.x_, plot_obj.y_] for plot_obj in cycle_plot_objs]
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
            print('Cycle {:d} correlation done.\n'.format(i))

        # ---------- 对完成所有cycle点迹-航迹关联的航迹进行动态可视化
        print('Start visualization...')
        draw_plot_track_correspondence(plots_per_cycle, tracks,
                                       init_phase_plots_state_dict, correlate_phase_plots_state_dict)

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

    cycle_time = int(plots_f_path.split('_')[-1].split('.')[0][:-1])

    # ---------- 点航相关
    nn_plot_track_correlate(plots_per_cycle, cycle_time, track_init_method=2)
    # ----------


if __name__ == '__main__':
    test_nn_plot_track_correlate(plots_f_path='./plots_in_each_cycle_1s.npy')
