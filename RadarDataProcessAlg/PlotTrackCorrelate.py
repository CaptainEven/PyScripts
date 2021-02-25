# encoding=utf-8

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from random import sample

from TrackInit import extrapolate_plot, Track, Plot, relate_gate_check
from TrackInit import direct_method_with_bkg, logic_method_with_bkg, corrected_logic_method_with_bkg


def compute_cov_mat(cycle_time, track, plots, sigma_s=160):
    """
    运动状态(5个):
    x: 点迹笛卡尔坐标x(m)
    y: 点迹笛卡尔坐标y(m)
    v: 点迹速度(m/s)
    a: 点迹加速度(m/s²)
    heading: 航向偏转角°
    :param cycle_time:
    :param track:
    :param plots:
    :param sigma_s:
    :return:
    """
    last_cycle = max([plot.cycle_ for plot in track.plots_])

    # ----- 外推next cycle预测点迹
    v = [plot.v_ for plot in track.plots_ if plot.cycle_ == last_cycle][0]

    # 提取前cycle和当前cycle的点迹
    plot_pre = [plot for plot in track.plots_ if plot.cycle_ == last_cycle - 1][0]
    plot_cur = [plot for plot in track.plots_ if plot.cycle_ == last_cycle][0]

    # 提取track在last_cycle的运动状态
    x, y = plot_cur.x_, plot_cur.y_
    v = plot_cur.v_
    a = plot_cur.a_
    heading = plot_cur.heading_

    # 预测位移值
    s = v * cycle_time

    # 计算(直线)外推点迹(预测点)笛卡尔坐标
    x_extra, y_extra = extrapolate_plot([plot_pre.x_, plot_pre.y_], [plot_cur.x_, plot_cur.y_], s)

    # 构建预测点迹对象: 跟last_cycle的plot保持一致
    plot_pred = Plot(last_cycle + 1, x_extra, y_extra, v, a, heading)
    print(plot_pred)

    # ----- 计算落入相关波门的候选点迹
    # 计算实际点迹与外推点迹之间的距离
    candidate_plots = []
    for plot in plots:
        shift_vector = np.array(plot) - np.array([plot_pred.x_, plot_pred.y_])
        l2_dist = np.linalg.norm(shift_vector, ord=2)
        if l2_dist < sigma_s:
            candidate_plots.append(plot)
    print(candidate_plots)

    # 构建观测值(候选点迹对象)
    can_plot_objs = []
    for plot in candidate_plots:
        # ---估计观测点迹运动参数
        # 计算位移
        shift_1 = np.array([plot[0], plot[1]]) - np.array([plot_cur.x_, plot_cur.y_])
        pass

    # ----- 计算滤波器残差矩阵
    res_mat = np.zeros((len(candidate_plots), 5), dtype=np.float32)
    for can_plot in candidate_plots:
        pass


def compute_mahalanobis_dist(cycle_time, track, plot, cov_mat):
    """
    计算某个暂时点迹与航迹之间的马氏距离(基于运动状态的统计距离)
    运动状态: loc, v, a, heading
    :param cycle_time:
    :param track:
    :param plot:
    :param cov_mat:
    :return:
    """


## 最近邻(NN)点-航相关算法
def nn_plot_track_correlate(plots_per_cycle, cycle_time, track_init_method=0):
    """
    :param plots_per_cycle:
    :param cycle_time:
    :param track_init_method:
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

            # 计算滤波器残差协方差矩阵
            for track in tracks:
                cov_mat = compute_cov_mat(cycle_time, track, plots)
        # ----------

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
    nn_plot_track_correlate(plots_per_cycle, cycle_time=cycle_time, track_init_method=2)
    # ----------


if __name__ == '__main__':
    test_nn_plot_track_correlate(plots_f_path='./plots_in_each_cycle_1s.npy')
