# encoding=utf-8

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from random import sample

from TrackInit import extrapolate_plot, Track, Plot, relate_gate_check
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

    print(res_mat)

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
        heading = math.degrees(math.acos(np.dot(shift0, shift1) / (dist0 * dist1)))

        # 构建plot object
        plot_obj = Plot(last_cycle + 1, plot[0], plot[1], v1, a, heading)

        can_plot_objs.append(plot_obj)

    return can_plot_objs


## 最近邻(NN)点-航相关算法
def nn_plot_track_correlate(plots_per_cycle, cycle_time,
                            track_init_method=0,
                            σ_s=160, λ=3):
    """
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

        # ---------- TODO: 后续点航相关过程
        # 获取下一个扫描cycle编号
        last_cycle = max([plot.cycle_ for track in tracks for plot in track.plots_])
        start_cycle = last_cycle + 1
        print('Start correlation from cycle {:d}.'.format(start_cycle))

        # ---------- 主循环: 遍历接下来的所有cycles
        for i in range(start_cycle, n_cycles):
            # 遍历下次扫描出现的所有点迹
            plots = plots_per_cycle[last_cycle + 1]
            print(plots)

            # -----计算马氏距离代价矩阵
            N = plots.shape[0]
            cost_mat = np.zeros((M, N), dtype=np.float32)

            for track in tracks:
                print('Processing track {:d}.'.format(track.id_))

                # 构建预测点迹对象: 跟last_cycle的plot保持一致
                plot_pred = get_predict_plot(track, cycle_time)

                # 计算候选观测点迹
                can_plot_objs = get_candidate_plot_objs(cycle_time, track, plot_pred, plots, σ_s)
                if len(can_plot_objs) == 0:  # 如果没有候选点迹落入该track的相关(跟踪)波门
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
                    the_plot_obj = can_plot_objs[0]

                    # 计算该点迹与其他航迹的距离
                    other_tracks = [track_o for track_o in tracks if track_o.id_ != track.id_]
                    other_ma_dists = []
                    for track_o in other_tracks:  #
                        # 构建预测点迹对象: 跟last_cycle的plot保持一致
                        plot_pred_o = get_predict_plot(track_o, cycle_time)

                        # 计算候选观测点迹
                        can_plot_objs_o = get_candidate_plot_objs(cycle_time, track_o, plot_pred_o, plots, σ_s)

                        if len(can_plot_objs_o) == 0:  # 如果没有候选点迹落入该track的相关(跟踪)波门
                            other_ma_dists.append(np.inf)
                        else:
                            # 计算残差的协方差矩阵
                            cov_mat_o = compute_cov_mat(plot_pred_o, can_plot_objs_o)

                            # --- 计算马氏距离
                            ma_dist_o = compute_ma_dist(cov_mat_o, the_plot_obj, plot_pred_o)
                            other_ma_dists.append(ma_dist_o)

                            # # 判断该点迹是否同时落入其他航迹
                            # if is_plot_in_relate_gate([the_plot_obj.x_, the_plot_obj.y_], plot_pred_o, σ_s):
                            #     pass
                    # print(other_ma_dists)

                    if ma_dist <= λ * min(other_ma_dists):
                        # 点迹-航迹直接相关
                        track.add_plot(the_plot_obj)


                elif len(can_plot_objs) > 1:
                    # NN点航关联
                    min_ma_dist = min(ma_dists)
                    min_idx = ma_dists.index(min_ma_dist)
                    the_plot_obj = can_plot_objs[min_idx]
                    track.add_plot(the_plot_obj)
        # ----------

    else:
        print('Track initialization failed.')


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
        # 计算马氏距离
        ma_dist = math.sqrt(res_vector.T.dot(res_vector))

    else:  # 5×5  有至少2个观测点迹落入相关(跟踪)波门
        # 计算马氏距离
        ma_dist = math.sqrt(np.dot(res_vector.T, cov_mat).dot(res_vector))

    return ma_dist


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
    nn_plot_track_correlate(plots_per_cycle, cycle_time=cycle_time, track_init_method=2)
    # ----------


if __name__ == '__main__':
    test_nn_plot_track_correlate(plots_f_path='./plots_in_each_cycle_1s.npy')
