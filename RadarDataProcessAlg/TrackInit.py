# encoding=utf-8

import os
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from collections import defaultdict
from scipy.stats import poisson
from scipy.optimize import linear_sum_assignment as linear_assignment

# ---------- Parameters ----------
markers = [
    '.',
    '+',
    '^',
    'v',
    '>',
    '<',
    's',
    'p',
    '*',
    'h',
    'H',
    'd',
    'D',
    'x',
    '|',
    '_'
]

colors = [
    'b',
    'g',
    'r',
    'c',
    'm',
    'y'
]

# TrackStates = {
#     'Temporaty': 0,
#     'Reliable':  1,
#     'Fixed':     2,
#     'Potential': 3
# }

TrackStates = {
    0: 'Potential',  # 可能航迹
    1: 'Temporaty',  # 暂时航迹
    2: 'Reliable',  # 可靠航迹
    3: 'Fixed',  # 固定航迹
}


# ---------- Data structure ----------
class Plot(object):
    def __init__(self, cycle, x, y, v, a, heading):
        """
        :param cycle: 雷达扫描周期(s)
        :param x: 点迹笛卡尔坐标x(m)
        :param y: 点迹笛卡尔坐标y(m)
        :param v: 点迹速度(m/s)
        :param a: 点迹加速度(m/s²)
        :param heading: °
        """
        self.cycle_ = cycle
        self.x_ = x
        self.y_ = y
        self.v_ = v
        self.a_ = a
        self.heading_ = heading


class Track(object):
    def __init__(self):
        self.state_ = 0  # 航迹状态
        self.plots_ = []
        self.init_cycle = -1

    def add_plot(self, plot):
        self.plots_.append(plot)


# ---------- Data(plots/tracks) generation ----------
def move_in_a_cycle(x0, y0, v0, a, direction, t=3):
    """
    @param x0: 起始坐标x
    @param y0: 起始坐标y
    @param v: 起始速度
    @param a: 加速度
    @param direction: 笛卡尔坐标系运动方向
    @param t: 扫描周期(s)
    """
    if direction < 0.0 or direction > 360.0:
        print('[Err]: invalid heading direction.')
        return None

    # 计算位移值
    s = v0 * t + 0.5 * a * t * t
    if direction > 0 and direction <= 90:  # 第1象限
        # 计算角度
        degree = direction
        radian = math.radians(degree)
        radian = radian if radian >= 0 else radian + math.pi
        degree = math.degrees(radian)

        # 计算新坐标
        x_delta = math.sin(radian) * s
        y_delta = math.cos(radian) * s
        x = x0 + x_delta
        y = y0 + y_delta
    elif direction > 90 and direction <= 180:  # 第2象限
        # 计算角度
        degree = 180 - direction
        radian = math.radians(degree)
        radian = radian if radian >= 0 else radian + math.pi
        degree = math.degrees(radian)

        # 计算新坐标
        x_delta = - math.sin(radian) * s
        y_delta = math.cos(radian) * s
        x = x0 + x_delta
        y = y0 + y_delta
    elif direction > 180 and direction <= 270:  # 第3象限
        # 计算角度
        degree = 270 - direction
        radian = math.radians(degree)
        radian = radian if radian >= 0 else radian + math.pi
        degree = math.degrees(radian)

        # 计算新坐标
        x_delta = - math.sin(radian) * s
        y_delta = - math.cos(radian) * s
        x = x0 + x_delta
        y = y0 + y_delta
    else:  # 第4象限
        # 计算角度
        degree = 360 - direction
        radian = math.radians(degree)
        radian = radian if radian >= 0 else radian + math.pi
        degree = math.degrees(radian)

        # 计算新坐标
        x_delta = math.sin(radian) * s
        y_delta = - math.cos(radian) * s
        x = x0 + x_delta
        y = y0 + y_delta

    return x, y


def gen_track_cv_ca(N=20, v0=340.0, a=0.0, direction=225, cycle_time=1.0):
    """
    按照雷达扫描周期生成一条航迹数据: 匀速运动(CV)模型&匀加速模型(CA)
    @param N: number of can cycle
    @param v: 初始速度m/s
    @param a: 加速度m/s²
    @param direction: 笛卡尔坐标系运动方向
    @param cycle_time: 三秒周期时间(s)
    TODO: 
        1. 加入随机的加速度扰动、方向扰动...生成更逼真的点迹数据
        2. 将生成的点迹/航迹数据持久化, 方便后续算法实现和调试
    """
    # 随机起始角度
    # degree = np.random.rand() * 360

    # 随即起始坐标
    x0 = np.random.randint(-40000, 40000)
    y0 = np.random.randint(-40000, 40000)
    while 1:
        if ((x0 > 32000 and x0 < 33000) or (x0 > -33000 and x0 < -32000)) \
                and ((y0 > 32000 and y0 < 33000) or (y0 > -33000 and y0 < -32000)):
            break
        else:
            # x0 = 30000
            # y0 = 35000
            x0 = np.random.randint(-40000, 40000)
            y0 = np.random.randint(-40000, 40000)
    print('X0Y0: [{:d}, {:d}]'.format(x0, y0))

    # 根据起始坐标(判断其所在笛卡尔坐标象限)设置direction
    # 模拟敌机来袭从远处飞往近处
    if x0 >= 0.0 and y0 >= 0.0:  # 第一象限
        direction = np.random.randint(200, 250)  # 飞往第三象限
    elif x0 >= 0.0 and y0 < 0.0:  # 第四象限
        direction = np.random.randint(110, 160)  # 飞往第二象限
    elif x0 < 0.0 and y0 >= 0.0:  # 第二象限
        direction = np.random.randint(290, 340)  # 飞往第四象限
    elif x0 < 0.0 and y0 < 0.0:  # 第三象限                        #
        direction = np.random.randint(20, 70)  # 飞往第一象限

    # 运动模型: 匀(加)速直线
    # 生成航迹
    track = []
    for i in range(N):
        # ---------- 每个周期都加入一定的随机扰动(噪声)
        # 为航向增加随机噪声扰动
        direction_noise = ((1 - (-1)) * np.random.random() + (-1)) * 10.0
        direction += direction_noise
        direction = direction if direction >= 0.0 else direction + 360.0
        direction = direction if direction <= 360.0 else direction - 360.0

        # 为加速度增加随机噪声扰动: 即航速扰动
        a_noise = ((1 - (-1)) * np.random.random() + (-1)) * 100
        a += a_noise

        # logging...
        print('Iter {:d} | heading direction: {:.3f}°'.format(i, direction))

        # 一个扫描周期目标状态改变量
        ret = move_in_a_cycle(x0, y0, v0, a, direction, cycle_time)
        if ret == None:
            continue

        # TODO: 加入噪声点迹false positive: 杂波背景

        x, y = ret
        x, y = int(x), int(y)

        # 保存当前扫描周期的笛卡尔坐标
        track.append([x, y])

        # 更新当前笛卡尔坐标
        x0, y0 = x, y

    return track


# 同时生成几个航迹(track)


def gen_tracks(M=3, N=60, v0=340, a=10, cycle_time=1):
    """
    :param M:  航迹数
    :param N:  雷达扫描周期数
    :param v0: 初始速度
    :param a:  加速度
    :param cycle_time: 雷达每周期扫描时间
    :return:
    """
    tracks = []
    for i in range(M):
        # 随机主航向角度(°)
        direction = np.random.rand() * 360

        # ---------- 生成一条航迹
        track = gen_track_cv_ca(N=N, v0=v0, a=a, direction=direction, cycle_time=1)
        # ----------

        tracks.append(track)
        # print(track, '\n')
    tracks = np.array(tracks)

    # 为生成的航迹添加随机背景杂波
    plots_in_each_cycle = []
    max_noise_num = 10
    for i in range(N):  # 遍历每一个雷达扫描周期
        probs = poisson.pmf(np.arange(max_noise_num + 1), 0.99)  # 泊松分布概率
        probs = probs[1:]
        rand_prob = np.random.random()
        probs = probs * rand_prob  # 随机概率
        # print(probs)

        # 按照概率生成杂波点迹
        noise_plots = []
        for j in range(max_noise_num):
            if probs[j] > 0.1:
                x = ((1 - (-1)) * np.random.random() + (-1)) * 35000.0
                y = ((1 - (-1)) * np.random.random() + (-1)) * 35000.0
                print(x, y)
                noise_plots.append([x, y])

        plots = tracks[:, i, :]
        if len(noise_plots) > 0:
            noise_plots = np.array(noise_plots).reshape((-1, 2))
            plot_list = np.concatenate((plots, noise_plots), axis=0)
            plots_in_each_cycle.append(plot_list)
        else:
            plots_in_each_cycle.append(plots)
    plots_in_each_cycle = np.array(plots_in_each_cycle)
    print(plots_in_each_cycle)

    # ---------- 序列化航迹数据到磁盘
    # tracks = np.array(tracks)
    # print(tracks)

    # ----- 存为npy文件
    # 保存原始tracks文件
    npz_save_path = './tracks_{:d}s'.format(cycle_time)
    np.save(npz_save_path, tracks)
    print('{:s} saved.'.format(npz_save_path))

    # 保存含有杂波背景的航迹文件
    npz_save_path = './plots_in_each_cycle_{:d}s'.format(cycle_time)
    np.save(npz_save_path, plots_in_each_cycle)
    print('{:s} saved.'.format(npz_save_path))

    # ----- 存为txt文件

    return np.array(tracks)


# ---------- Algorithms
"""
航迹起始滑窗法的 m/n逻辑:
如果这 N 次扫描中有某
M 个观测值满足以下条件，那么启发式规则法就认定应起始一条航迹
"""


def slide_window(track, n=4, start_cycle=1):
    """
    :param track:
    :param n:
    :param start_cycle:
    :return:
    """
    window = [track[i] for i in range(start_cycle - 2, start_cycle + n)]

    return window


def get_v(plots_2, cycle_time):
    """
    根据连续2个点迹计算后一个点迹的速度
    :param plots_2:
    :param cycle_time:
    :return:
    """
    plot0, plot1 = plots_2
    x0, y0 = plot0
    x1, y1 = plot1

    # 计算位移向量
    shift_vector = np.array([x1, y1]) - np.array([x0, y0])

    # 计算位移值
    s = np.linalg.norm(shift_vector, ord=2)

    return s / cycle_time


def get_v_a_angle(plots_3, cycle_time):
    """
    通过连续3个点迹计算最后一个点迹的速度、加速度、航向偏转角
    :param plots_3:
    :param cycle_time:
    :return:
    """
    plot0, plot1, plot2 = plots_3
    x0, y0 = plot0
    x1, y1 = plot1
    x2, y2 = plot2

    # 计算位移向量
    s0 = np.array([x1, y1]) - np.array([x0, y0])
    s1 = np.array([x2, y2]) - np.array([x1, y1])

    # 计算位移数值
    dist0 = np.linalg.norm(s0, ord=2)
    dist1 = np.linalg.norm(s1, ord=2)

    # 计算速度
    v0 = dist0 / cycle_time
    v1 = dist1 / cycle_time

    # 计算加速度
    a = (v1 - v0) / cycle_time

    # ----- 计算航向偏转角
    # 计算角度(夹角余弦): 返回反余弦弧度值
    dot_val = np.dot(s0, s1)
    cos_sim = dot_val / (dist0 * dist1)
    cos_sim = cos_sim if cos_sim <= 1.0 else 1.0
    cos_sim = cos_sim if cos_sim >= -1.0 else -1.0
    radian = math.acos(cos_sim)

    return v1, a, radian


def matching_plots_nn(plots_0, plots_1, K):
    """
    :param plots_0:
    :param plots_1:
    :param K:
    :return:
    """
    M, N = plots_0.shape[0], plots_1.shape[0]

    mapping = {}
    cost_mat = np.zeros((M, N), dtype=np.float32)
    for i, plot_0 in enumerate(plots_0):
        for j, plot_1 in enumerate(plots_1):
            shift_vector = plot_0 - plot_1
            l2_dist = np.linalg.norm(shift_vector, ord=2)
            cost_mat[i][j] = l2_dist

    # 取topK: cost最小
    # k_smallest = heapq.nsmallest(K, cost_mat.ravel().tolist())
    inds = np.argpartition(cost_mat.ravel(), K)[:K]
    inds_i = inds // N
    inds_j = inds % N
    # k_smallest = cost_mat[inds_i, inds_j]

    for i, j in zip(inds_i, inds_j):  # i∈range(M), j∈range(N)
        mapping[i] = j

    return mapping


def direct_method_with_bkg(plots_per_cycle, cycle_time, v_min, v_max, a_max, angle_max, m=3, n=4):
    """
    :param plots_per_cycle:
    :param cycle_time:
    :param v_min:
    :param v_max:
    :param a_max:
    :param angle_max:
    :param m:
    :param n:
    :return:
    """
    N = plots_per_cycle.shape[0]  # number of cycles

    tracks = []  # ret

    # 取滑动窗口
    succeed = False
    for i in range(0, N - n):  # cycle i
        if succeed:
            break

        # 取滑窗(连续6个cycle)
        window = slide_window(plots_per_cycle, n, start_cycle=i)

        # ----------对窗口中进行m/n统计
        # 构建mapping链
        K = min([cycle_plots.shape[0] for cycle_plots in window])  # 最小公共点迹数
        mappings = defaultdict(dict)
        for j in range(len(window) - 1, 0, -1):
            # ----- 构建相邻cycle的mapping
            mapping = matching_plots_nn(window[j], window[j - 1], K)
            # -----

            if len(set(mapping.values())) != len(set(mapping.keys())):
                break
            else:
                mappings[j] = mapping

        if len(mappings) < m + 2:  # 至少有m个cycle有效数据
            continue  # 滑动到下一个window

        # 对mapping结果进行排序(按照key降序排列)
        mappings = sorted(mappings.items(), key=lambda x: x[0], reverse=True)
        # print(mappings)

        # 构建暂时航迹
        for k in range(K):  # 遍历每个暂时航迹
            # ----- 航迹状态记录
            # 窗口检出数计数: 每个暂时航迹单独计数
            n_pass = 0

            # 窗口运动状态记录: 每个航迹单独记录(速度, 加速度, 航向偏转角)
            window_states = defaultdict(dict)
            # -----

            # 构建暂时航迹组成的点迹(plot)
            plot_ids = []
            id = -1

            # 提取倒序第一个有效cycle的plot id
            keys = mappings[0][1].keys()
            keys = sorted(keys, reverse=False)  # 按照当前window最大的有效cycle的点迹序号升序排列
            id = keys[k]
            plot_ids.append(id)

            # 按照mapping链递推其余cycle的plot id
            for (c, mapping) in mappings:  # mapping已经按照cycle倒序排列过了
                id = mapping[id]  # 倒推映射链plot id
                plot_ids.append(id)

            # print(ids)  # ids是按照cycle倒排的
            # 根据ids链接构建plot链: 暂时航迹
            cycle_ids = [c for (c, mapping) in mappings]  # 按照cycle编号倒排
            cycle_ids.extend([mappings[-1][0] - 1])

            assert len(cycle_ids) == len(plot_ids)

            plots = [window[cycle][plot_id] for cycle, plot_id in zip(cycle_ids, plot_ids)]

            # 可视化验证
            # plot_plots(plots, cycle_ids)

            # print(plots)
            plots_to_test = plots[:-2]
            cycle_ids_to_test = cycle_ids[:-2]
            # plot_plots(plots_to_test, cycle_ids_to_test)

            # window内逐一门限测试
            # for l, (cycle_id, plot) in enumerate(zip(cycle_ids_to_test, plots_to_test)):
            for l in range(len(plots) - 2):
                cycle_id = cycle_ids[l]

                # 构建连续三个cycle的plots
                plots_3 = [plots[l + 2], plots[l + 1], plots[l]]
                # plot_plots(plots_3)

                # 估算当前点迹的运动状态(速度, 加速度, 偏航角度)
                v, a, angle_in_radians = get_v_a_angle(plots_3, cycle_time)
                angle_in_degrees = math.degrees(angle_in_radians)
                angle_in_degrees = angle_in_degrees if angle_in_degrees >= 0.0 else angle_in_degrees + 360.0
                angle_in_degrees = angle_in_degrees if angle_in_degrees <= 360.0 else angle_in_degrees - 360.0

                # 门限判定
                if v >= v_min and \
                        v <= v_max and \
                        a <= a_max and \
                        angle_in_degrees < angle_max:
                    # 更新n_pass
                    n_pass += 1

                    # window运动状态记录
                    state_dict = {
                        'cycle': cycle_id,
                        'x': plots[l][0],
                        'y': plots[l][1],
                        'v': v,
                        'a': a,
                        'angle_in_degrees': angle_in_degrees
                    }
                    window_states[cycle_id] = state_dict

                else:  # 记录航迹起始失败原因: logging
                    is_v_pass = v >= v_min and v <= v_max
                    is_a_pass = a <= a_max
                    is_angle_pass = angle_in_degrees <= angle_max
                    if not is_v_pass:
                        if v < v_min:
                            print('Track {:d} init failed @cycle{:d}, velocity threshold: {:.3f} < {:.3f}m/s'
                                  .format(k, i, float(v), v_min))
                        elif v > v_max:
                            print('Track {:d} init failed @cycle{:d}, velocity threshold: {:.3f} > {:.3f}m/s'
                                  .format(k, i, float(v), v_max))
                    if not is_a_pass:
                        print('Track {:d} init failed @cycle{:d}, acceleration threshold: {:.3f} > {:.3f}m/s²'
                              .format(k, i, a, float(a_max)))
                    if not is_angle_pass:
                        print('Track {:d} init failed @cycle{:d}, heading angle threshold: {:.3f} > {:.3f}°'
                              .format(k, i, angle_in_degrees, angle_max))

            # 判定是否当前航迹初始化成功
            if n_pass >= m:
                print('Track {:d} inited successfully @cycle {:d}.'.format(k, i))

                # TODO: 初始化航迹对象
                track = Track()
                track.state_ = 2  # 航迹状态: 可靠航迹
                track.init_cycle = i  # 航迹起始cycle
                window_states = sorted(window_states.items(), key=lambda x: x[0], reverse=False)  # 升序重排
                for k, v in window_states:
                    print(k, v)
                    plot = Plot(v['cycle'], v['x'], v['y'], v['v'], v['a'], v['angle_in_degrees'])
                    track.add_plot(plot)
                tracks.append(track)

                # 航迹起始成功标识
                succeed = True

                # 清空窗口状态
                window_states = defaultdict(dict)

                # 跳出当前航迹检测, 到下一个暂时航迹
                continue
        # ----------

    # if tracks != []:
    #     print(tracks)
    return succeed, tracks


# Page 57
def direct_method(track, cycle_time, v_min, v_max, a_max, angle_max, m=3, n=4):
    """
    直观航迹起始算法:
    1. 速度门限判断
    2. 加速度门限判断
    3. 角度门限判断
    :param track:
    :param cycle_time: 雷达扫描一周耗时(s)
    :param v_min:
    :param v_max:
    :param a_max:
    :param angle_max:  一次雷达扫描周期内最大允许航向角度偏转(°)
    :param m:
    :param n:
    :return:
    """
    start_cycle = -1
    N = track.shape[0]

    # 取滑动窗口
    succeed = False
    for i in range(2, N - n):
        # 取滑窗
        window = slide_window(track, n, i)

        # 判定
        n_pass = 0
        for j, plot in enumerate(window):
            if j >= 2 and j < len(window) - 1:  # 从第三个点迹开始求v, a, angle
                # 获取连续3个点迹
                plots_3 = window[j - 2: j + 1]  # 3 plots: [j-2, j-1, j]

                # 估算当前点迹的运动状态
                v, a, angle_in_radians = get_v_a_angle(plots_3, cycle_time)

                # 航向偏移角度估算
                angle_in_degrees = math.degrees(angle_in_radians)
                angle_in_degrees = angle_in_degrees if angle_in_degrees >= 0.0 else angle_in_degrees + 360.0
                angle_in_degrees = angle_in_degrees if angle_in_degrees <= 360.0 else angle_in_degrees - 360.0

                print('Cycle{:d} | velocity: {:.3f}m/s | acceleration: {:.3f}m/s² | heading angle: {:.3f}°'
                      .format(i, v, a, angle_in_degrees))

                # 门限判定
                if v >= v_min and \
                        v <= v_max and \
                        a <= a_max and \
                        angle_in_degrees < angle_max:
                    n_pass += 1

                else:  # 记录航迹起始失败原因
                    is_v_pass = v >= v_min and v <= v_max
                    is_a_pass = a <= a_max
                    is_angle_pass = angle_in_degrees <= angle_max

                    if not is_v_pass:
                        if v < v_min:
                            print('Track init failed @cycle{:d}, velocity threshold: {:.3f} < {:.3f}m/s'
                                  .format(i, float(v), v_min))
                        elif v > v_max:
                            print('Track init failed @cycle{:d}, velocity threshold: {:.3f} > {:.3f}m/s'
                                  .format(i, float(v), v_max))
                    if not is_a_pass:
                        print('Track init failed @cycle{:d}, acceleration threshold: {:.3f} > {:.3f}m/s²'
                              .format(i, a, float(a_max)))
                    if not is_angle_pass:
                        print('Track init failed @cycle{:d}, heading angle threshold: {:.3f} > {:.3f}°'
                              .format(i, angle_in_degrees, angle_max))
            else:
                continue

        # 判定航迹是否起始成功
        if n_pass >= m:
            succeed = True

        if succeed:
            start_cycle = i
            break
        else:
            continue  # 下一个滑窗

    return succeed, start_cycle


def start_gate_check(cycle_time, plot_pre, plot_cur, v0, min_ratio=0.1, max_ratio=2.5):
    """
    用速度法建立起始波门
    :param cycle_time:
    :param plot_pre:
    :param plot_cur:
    :param v0:
    :return:
    """
    # ----- 计算初始波门(环形波门的两个半径)
    r_min = v0 * cycle_time * min_ratio  # 小半径
    r_max = v0 * cycle_time * max_ratio  # 大半径

    # 距离计算
    x_pre, y_pre = plot_pre
    x_cur, y_cur = plot_cur
    shift_vector = np.array([x_cur, y_cur]) - np.array([x_pre, y_pre])
    dist = np.linalg.norm(shift_vector, ord=2)

    return dist >= r_min and dist <= r_max


def extrapolate_plot(plot_pre, plot_cur, s):
    """
    :param plot_pre: 前一个点迹
    :param plot_cur: 当前点迹
    :param s: 位移值
    :return: 直线外推(预测)点迹
    """
    # 以前一个点迹为笛卡尔坐标原点建笛卡尔坐标系
    # 判定当前点迹所在的象限
    x_pre, y_pre = plot_pre
    x_cur, y_cur = plot_cur

    if x_cur >= x_pre and y_cur >= y_pre:  # 第一象限
        # 计算与x轴夹角
        radian = math.atan2((y_cur - y_pre), (x_cur - x_pre))
        if radian >= 0.0 and radian <= math.pi * 0.5:
            x_extra = x_cur + s * math.cos(radian)
            y_extra = y_cur + s * math.sin(radian)
        else:
            print('[Err]: current heading direction angle computed wrong!')
            return None

    elif x_cur < x_pre and y_cur >= y_pre:  # 第二象限
        radian = math.atan2((y_cur - y_pre), (x_pre - x_cur))
        if radian >= 0.0 and radian <= math.pi * 0.5:
            x_extra = x_cur - s * math.cos(radian)
            y_extra = y_cur + s * math.sin(radian)
        else:
            print('[Err]: current heading direction angle computed wrong!')
            return None

    elif x_cur < x_pre and y_cur < y_pre:  # 第三象限
        radian = math.atan2((y_pre - y_cur), (x_pre - x_cur))
        if radian >= 0.0 and radian <= math.pi * 0.5:
            x_extra = x_cur - s * math.cos(radian)
            y_extra = y_cur - s * math.sin(radian)
        else:
            print('[Err]: current heading direction angle computed wrong!')
            return None

    elif x_cur >= x_pre and y_cur < y_pre:  # 第四象限
        radian = math.atan2((y_pre - y_cur), (x_cur - x_pre))
        if radian >= 0.0 and radian <= math.pi * 0.5:
            x_extra = x_cur + s * math.cos(radian)
            y_extra = y_cur - s * math.sin(radian)
        else:
            print('[Err]: current heading direction angle computed wrong!')
            return None

    return x_extra, y_extra


# 数据互联是通过相关波门实现的
def relate_gate_check(cycle_time, v, plot_pre, plot_cur, plot_next, sigma):
    """
    Page46
    最简单的圆(环)形相关波门:
    对每个暂时航迹进行外推，以外推点为中心，建立后续相关波门
    :param cycle_time:
    :param v:
    :param plot_pre:
    :param plot_cur:
    :param plot_next:
    :param sigma:
    :return:
    """
    # 取当前测试序列第三次扫描点迹的笛卡尔坐标
    x_nex, y_nex = plot_next

    # 预测位移值
    s = v * cycle_time

    # 计算(直线)外推点
    x_extra, y_extra = extrapolate_plot(plot_pre, plot_cur, s)

    # 计算实际点迹与外推点迹之间的距离
    dist = math.sqrt((x_nex - x_extra) * (x_nex - x_extra) +
                     (y_nex - y_extra) * (y_nex - y_extra))

    return dist <= sigma


def corrected_relate_gate_check(cycle_time, v, plot_pre, plot_cur, plot_next, s_sigma, a_sigma):
    """
    :param cycle_time:
    :param v:
    :param plot_pre:
    :param plot_cur:
    :param plot_next:
    :param s_sigma: 位移sigma
    :param a_sigma: 航向角度sigma(°)
    :return:
    """
    # 取当前测试序列第三次扫描点迹的笛卡尔坐标
    x_pre, y_pre = plot_pre
    x_cur, y_cur = plot_cur
    x_nex, y_nex = plot_next

    # 预测位移值
    s = v * cycle_time

    # 计算(直线)外推点
    x_extra, y_extra = extrapolate_plot(plot_pre, plot_cur, s)

    # 计算实际点迹与外推点迹之间的距离
    dist = math.sqrt((x_nex - x_extra) * (x_nex - x_extra) +
                     (y_nex - y_extra) * (y_nex - y_extra))

    # 修正判定
    if dist <= s_sigma:
        # ----- 第三次扫描角度判定: 计算余弦夹角
        # 计算位移向量
        s_12 = np.array([x_cur, y_cur]) - np.array([x_pre, y_pre])
        s_23 = np.array([x_nex, y_nex]) - np.array([x_cur, y_cur])
        l2norm_12 = np.linalg.norm(s_12, ord=2)
        l2norm_23 = np.linalg.norm(s_23, ord=2)

        # 计算角度(夹角余弦): 返回反余弦弧度值
        radian = math.acos(np.dot(s_12, s_23) / (l2norm_12 * l2norm_23))
        degree = math.degrees(radian)
        if degree <= a_sigma:
            return True, 0
        else:
            return False, 1

        # TODO: 如果第三次扫描不满足条件, 继续判定第四次扫描
    else:  # 不满足基于距离的相关波门
        return False, 2


def logic_method_with_bkg(plots_per_cycle, cycle_time, sigma=160, m=3, n=4):
    """
    :param plots_per_cycle:
    :param cycle_time:
    :param sigma:
    :param m:
    :param n:
    :return:
    """
    N = plots_per_cycle.shape[0]  # number of cycles

    tracks = []  # ret

    # 取滑动窗口
    succeed = False
    for i in range(0, N - n):  # cycle i
        if succeed:
            break

        # 取滑窗(连续6个cycle)
        window = slide_window(plots_per_cycle, n, start_cycle=i)

        #


def logic_method(track, cycle_time, sigma=160, m=3, n=4):
    """
    逻辑法
    :param track:
    :param cycle_time:
    :param sigma:
    :param m:
    :param n:
    :return:
    """
    # TODO: 动态更新sigma

    start_cycle = -1
    N = track.shape[0]

    # 窗口滑动
    succeed = False
    for i in range(1, N - n):
        # 取滑窗
        window = slide_window(track, n, i)

        # 判定
        n_pass = 0
        for j, plot in enumerate(window):
            if j >= 1:  # 从第三个点迹开始求v, a, angle
                # 获取连续3个点迹
                # plots_3 = window[j-2: j+1]  # 3 plots: [j-2, j-1, j]
                plots_2 = window[j-1: j+1]  # 2 plots:[j-1, j]

                # 估算当前点迹的运动状态
                # v, a, angle_in_radians = get_v_a_angle(plots_3, cycle_time)
                v = get_v(plots_2, cycle_time)

                # # 航向偏移角度估算
                # angle_in_degrees = math.degrees(angle_in_radians)
                # angle_in_degrees = angle_in_degrees if angle_in_degrees >= 0.0 else angle_in_degrees + 360.0
                # angle_in_degrees = angle_in_degrees if angle_in_degrees <= 360.0 else angle_in_degrees - 360.0

                # print('Cycle{:d} | velocity: {:.3f}m/s | acceleration: {:.3f}m/s² | heading angle: {:.3f}°'
                #       .format(i, v, a, angle_in_degrees))

                # ----- 判定逻辑
                if j >= 1 and j < len(window) - 1:  # 从第3次扫描开始逻辑判定: j==2的点迹作为航迹头
                    # 初始波门判定: j是当前判定序列的第二次扫描
                    if start_gate_check(cycle_time, window[j - 1], window[j], v0=340):

                        # --- 对通过初始波门判定的航迹建立暂时航迹, 继续判断相关波门
                        # page71-72
                        if relate_gate_check(cycle_time, v, window[j - 1], window[j], window[j + 1], sigma=sigma):
                            n_pass += 1
                        else:
                            print('Track init failed @cycle{:d}, object(plot) is not in relating gate.'.format(i))
                    else:
                        print('Track init failed @cycle{:d} @window{:d}, object(plot) is not in the starting gate.'
                              .format(i, j))
            else:
                continue

        # 判定航迹是否起始成功
        if n_pass >= m:
            succeed = True

        if succeed:
            start_cycle = i
            break
        else:
            continue  # 下一个滑窗

    return succeed, start_cycle


# ----- 修正逻辑法 -----
def corrected_logic_method(track, cycle_time, s_sigma=160, a_sigma=10, m=3, n=4):
    """
    :param track:
    :param cycle_time:
    :param s_sigma: 位移sigma(m)
    :param a_sigma: 航向角度sigma(°)
    :param m:
    :param n:
    :return:
    """
    start_cycle = -1
    N = track.shape[0]

    # 窗口滑动
    succeed = False
    for i in range(1, N - n):
        # 取滑窗
        window = slide_window(track, n, i)

        # 判定
        n_pass = 0
        for j, plot in enumerate(window):
            if j >= 1:  # 从第三个点迹开始求v, a, angle
                # 获取连续3个点迹
                # plots_3 = window[j-2: j+1]  # 3 plots: [j-2, j-1, j]
                plots_2 = window[j-1: j+1]  # 2 plots: [j-1, j]

                # # 估算当前点迹的运动状态
                # v, a, angle_in_radians = get_v_a_angle(plots_3, cycle_time)
                v = get_v(plots_2, cycle_time)

                # # 航向偏移角度估算
                # angle_in_degrees = math.degrees(angle_in_radians)
                # angle_in_degrees = angle_in_degrees if angle_in_degrees >= 0.0 else angle_in_degrees + 360.0
                # angle_in_degrees = angle_in_degrees if angle_in_degrees <= 360.0 else angle_in_degrees - 360.0

                # print('Cycle{:d} | velocity: {:.3f}m/s | acceleration: {:.3f}m/s² | heading angle: {:.3f}°'
                #       .format(i, v, a, angle_in_degrees))

                # ----- 判定逻辑
                if j >= 1 and j < len(window) - 1:  # 从第4次扫描开始逻辑判定: j==3的点迹作为航迹头
                    # 初始波门判定: j是当前判定序列的第二次扫描
                    if start_gate_check(cycle_time, window[j-1], window[j], v0=340):

                        # --- 对通过初始波门判定的航迹建立暂时航迹, 继续判断相关波门
                        # page71-72
                        is_pass, ret = corrected_relate_gate_check(cycle_time, v,
                                                                   window[j-1], window[j], window[j+1],
                                                                   s_sigma, a_sigma)
                        if is_pass:
                            n_pass += 1
                        else:
                            if ret == 2:
                                print('Track init failed @cycle{:d} @window{:d}, corrected relating gate: out of shift sigma.'
                                      .format(i, j))
                            elif ret == 1:
                                print('Track init failed @cycle{:d} @window{:d}, corrected relating gate: out of angle sigma.'
                                        .format(i, j))
                    else:
                        print('Track init failed @cycle{:d} @window{:d}, object(plot) is not in the starting gate.'
                              .format(i, j))
            else:
                continue

        # 判定航迹是否起始成功
        if n_pass >= m:
            succeed = True

        if succeed:
            start_cycle = i
            break  # 航迹起始成功, 建立了稳定的航迹, 后续需要进行航机保持判断(点航相关)
        else:
            continue  # 下一个滑窗

    return succeed, start_cycle


def test_track_init_methods_with_bkg(plots_f_path, cycle_time, method):
    """
    :param plots_f_path:
    :param cycle_time: s
    :param method:
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

    N = plots_per_cycle.shape[0]
    print('Total {:d} radar cycles.'.format(N))

    # 当前扫描与前次扫面的点迹进行NN匹配: 局部贪心匹配, 计算代价矩阵
    # TODO: 进行匈牙利匹配

    if method == 0:  # 直观法
        succeed, tracks = direct_method_with_bkg(plots_per_cycle,
                                                 cycle_time,
                                                 v_min=150, v_max=500,  # < 2M
                                                 a_max=50, angle_max=10,  # 军机7°/s
                                                 m=3, n=4)
    if succeed:
        M = len(tracks)
        print('{:d} tracks initialization succeeded.'.format(M))

        # ---------- 可视化成功起始的航迹
        for track in tracks:
            # print(track)
            cycles = [plot.cycle_ for plot in track.plots_]
            xs = [plot.x_ for plot in track.plots_]
            ys = [plot.y_ for plot in track.plots_]

            plots = [[x, y] for x, y in zip(xs, ys)]
            plot_plots(plots, cycles)

        # ---------- TODO: 后续点航相关过程
        pass

    else:
        print('Track initialization failed.')


def test_track_init_methods(track_f_path, cycle_time, method):
    """
    测试直观法, 逻辑法
    :param track_f_path:
    :param cycle_time:
    :param method: 0: 直接法, 1: 逻辑法, 2: 修正逻辑法
    :return:
    """
    # 加载tracks文件
    if not os.path.isfile(track_f_path):
        print('[Err]: invalid file path.')
        return
    if track_f_path.endswith('.npy'):
        tracks = np.load(track_f_path, allow_pickle=True)
    elif track_f_path.endswith('.txt'):
        pass
    else:
        print('[Err]: invalid tracks file format.')
        return

    # 遍历每一个track: 进行航迹起始判定
    for i, track in enumerate(tracks):
        if method == 0:  # 直观法
            succeed, start_cycle = direct_method(track,
                                                 cycle_time,
                                                 v_min=200, v_max=400,  # 2M
                                                 a_max=15, angle_max=7,  # 军机7°/s
                                                 m=3, n=4)
        elif method == 1:  # 逻辑法
            succeed, start_cycle = logic_method(track,
                                                cycle_time,
                                                sigma=160,
                                                m=3, n=4)
        elif method == 2:  # 修正逻辑法
            succeed, start_cycle = corrected_logic_method(track,
                                                          cycle_time,
                                                          s_sigma=160, a_sigma=7,
                                                          m=3, n=4)

        if succeed:
            print('Track {:d} initialization succeeded @cycle {:d}.'
                  .format(i, start_cycle))

            # ----- 初始化航迹

            # 后续点航相关过程...
        else:
            print('Track {:d} initialization failed.'.format(i))


# ---------- Plot ----------


def plot_polar_map(track):
    """
    绘制雷达地图(极坐标)
    :param track:
    :return:
    """
    # 绘制基础地图(极坐标系)
    fig = plt.figure(figsize=[8, 8])
    ax0 = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection="polar")
    ax0.set_rmin(100)
    ax0.set_rmax(10000)
    ax0.set_rticks(np.arange(0, 10000, 1000))

    # 按照扫描周期显示
    # plt.ion()
    for pos in track:
        # 笛卡尔坐标
        x, y = pos

        # 计算极径
        r = math.sqrt(x * x + y * y)

        # 计算极角
        theta = math.atan2(y, x)

        # 绘制点迹
        ax0.scatter(theta, r, c='b', marker='o')  # 散点图
        ax0.text(theta, r, str((x, y)))  # 为每个点绘制标签
        plt.pause(0.5)

    # 绘图展示
    plt.show()


def plot_cartesian_map(track):
    """
    绘制雷达地图(笛卡尔坐标)
    :param track:
    :return:
    """
    # 绘制基础地图(笛卡尔坐标系)
    fig = plt.figure(figsize=[8, 8])
    ax0 = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax0.set_xticks(np.arange(-10000, 10000, 1000))
    ax0.set_yticks(np.arange(-10000, 10000, 1000))

    # 按照扫描周期显示
    # plt.ion()
    for i, pos in enumerate(track):
        # 笛卡尔坐标
        x, y = pos

        # 绘制点迹
        ax0.scatter(x, y, c='b', marker='o')  # 散点图
        # ax0.text(x, y, str((x, y)))  # 为每个点绘制坐标标签
        ax0.text(x, y, str(i))  # 为每个点绘制坐标标签
        plt.pause(0.5)

    # 绘图展示
    plt.show()


def plot_polar_cartesian_map(track):
    """
    绘制雷达地图(极坐标&笛卡尔坐标)
    :param track:
    :return:
    """
    # 绘制基础地图(极坐标系)
    fig = plt.figure(figsize=[16, 8])
    fig.suptitle('Radar')

    ax0 = plt.subplot(121, projection="polar")
    ax0.set_rmin(100)
    ax0.set_rmax(10000)
    ax0.set_rticks(np.arange(0, 100000, 1000))
    ax0.set_title('polar')

    ax1 = plt.subplot(122)
    ax1.set_xticks(np.arange(-10000, 50000, 1000))
    ax1.set_yticks(np.arange(-10000, 50000, 1000))
    ax1.set_title('cartesian')

    # 按照扫描周期显示
    # plt.ion()
    for i, pos in enumerate(track):
        # 笛卡尔坐标
        x, y = pos

        # 计算极径
        r = math.sqrt(x * x + y * y)

        # 计算极角
        theta = math.atan2(y, x)

        # 绘制点迹
        ax0.scatter(theta, r, c='b', marker='o')  # 散点图
        ax0.text(theta, r, str(i))  # 为每个点绘制序号/笛卡尔坐标标签

        ax1.scatter(x, y, c='b', marker='o')  # 散点图
        ax1.text(x, y, str(i))  # 为每个点绘制坐标标签
        plt.pause(0.5)

    # 绘图展示
    plt.show()


def plot_plots(plots, cycles=[]):
    """
    :param plots:
    :return:
    """
    plots = np.array(plots)
    if plots.shape[0] == 0:
        print('[Err]: empty plots.')
        return

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

    x = plots[:, 0]
    y = plots[:, 1]

    # 计算极径
    r = np.sqrt(x * x + y * y)

    # 计算极角
    theta = np.arctan2(y, x)
    neg_inds = np.where(theta < 0.0)
    theta[neg_inds] += np.pi * 2.0

    # 绘制极坐标点迹
    color, marker = sample(colors, 1)[0], sample(markers, 1)[0]
    ax0.scatter(theta, r, c=color, marker=marker)
    for j in range(theta.shape[0]):
        if cycles == []:
            ax0.text(theta[j], r[j], str(j))
        else:
            ax0.text(theta[j], r[j], str(cycles[j]))
    ax1.scatter(x, y, c=color, marker=marker)
    for j in range(x.shape[0]):
        if cycles == []:
            ax1.text(x[j], y[j], str(j))
        else:
            ax1.text(x[j], y[j], str(cycles[j]))

    plt.show()


def plot_plots_in_each_cycle(plots_f_path):
    """
    可视化含有杂波背景的点迹数据
    :param plots_f_path:
    :return:
    """
    # 加载tracks文件
    if not os.path.isfile(plots_f_path):
        print('[Err]: invalid file path.')
        return
    if plots_f_path.endswith('.npy'):
        tracks = np.load(plots_f_path, allow_pickle=True)
    elif plots_f_path.endswith('.txt'):
        pass
    else:
        print('[Err]: invalid tracks file format.')
        return

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

    # ----- 同一扫描周期之内同时绘制所有点迹
    N = tracks.shape[0]  # 含有杂波背景的数据, 此时N未知
    for i in range(N):  # 遍历每个雷达扫描周期
        color = sample(colors, 1)[0]
        marker = sample(markers, 1)[0]

        cycle_plots = tracks[i]  # 当前cycle的所有点迹坐标
        # print(cycle_plots)

        # 笛卡尔坐标
        x, y = cycle_plots[:, 0], cycle_plots[:, 1]

        # 计算极径
        r = np.sqrt(x * x + y * y)

        # 计算极角
        theta = np.arctan2(y, x)
        neg_inds = np.where(theta < 0.0)
        theta[neg_inds] += np.pi * 2.0

        # 绘制极坐标点迹
        ax0.scatter(theta, r, c=color, marker=marker)
        for j in range(theta.shape[0]):
            ax0.text(theta[j], r[j], str(i))

        # 绘制笛卡尔坐标点迹
        ax1.scatter(x, y, c=color, marker=marker)
        for j in range(x.shape[0]):
            ax1.text(x[j], y[j], str(i))

        plt.pause(0.5)

    # 绘图展示
    plt.show()


def plot_tracks(track_f_path):
    """
    :param track_f_path:
    :return:
    """
    # 加载tracks文件
    if not os.path.isfile(track_f_path):
        print('[Err]: invalid file path.')
        return
    if track_f_path.endswith('.npy'):
        tracks = np.load(track_f_path, allow_pickle=True)
    elif track_f_path.endswith('.txt'):
        pass
    else:
        print('[Err]: invalid tracks file format.')
        return

    # 绘制基础地图(极坐标系)
    fig = plt.figure(figsize=[16, 8])
    fig.suptitle('Radar')

    ax0 = plt.subplot(121, projection="polar")
    ax0.set_theta_zero_location('E')
    ax0.set_theta_direction(-1)  # anti-clockwise
    ax0.set_rmin(10)
    ax0.set_rmax(100000)
    ax0.set_rticks(np.arange(-50000, 50000, 3000))
    ax0.set_title('polar')

    ax1 = plt.subplot(122)
    ax1.set_xticks(np.arange(-50000, 50000, 10000))
    ax1.set_yticks(np.arange(-50000, 50000, 10000))
    ax1.set_title('cartesian')

    # # ----- 逐一绘制每个track
    # for track in tracks:
    #     color = sample(colors, 1)[0]
    #     marker = sample(markers, 1)[0]
    #     for i, pos in enumerate(track):
    #         # 笛卡尔坐标
    #         x, y = pos

    #         # 计算极径
    #         r = math.sqrt(x * x + y * y)

    #         # 计算极角
    #         theta = math.atan2(y, x)
    #         # if theta < 0.0:
    #         #     print('pause here.')
    #         theta = theta if theta >= 0 else theta + math.pi*2.0

    #         # 绘制点迹
    #         ax0.scatter(theta, r, c=color, marker=marker)  # 散点图
    #         ax0.text(theta, r, str(i))  # 为每个点绘制序号/笛卡尔坐标标签

    #         ax1.scatter(x, y, c=color, marker=marker)  # 散点图
    #         ax1.text(x, y, str(i))  # 为每个点绘制坐标标签
    #         plt.pause(0.5)

    # ----- 同一扫描周期之内同时绘制所有点迹
    M, N, _ = tracks.shape
    cycle_colors = sample(colors, M)
    cycle_markers = sample(markers, M)
    for i in range(N):  # 遍历每个雷达扫描周期
        cycle_plots = tracks[:, i, :]  # 当前cycle的所有点迹坐标
        # print(cycle_plots)

        # 笛卡尔坐标
        x, y = cycle_plots[:, 0], cycle_plots[:, 1]

        # 计算极径
        r = np.sqrt(x * x + y * y)

        # 计算极角
        theta = np.arctan2(y, x)
        neg_inds = np.where(theta < 0.0)
        theta[neg_inds] += np.pi * 2.0

        # 绘制点迹组成的航迹
        for j in range(M):
            # 在第一个雷达扫描周期绘制航迹(编号)标签
            if i == 0:
                ax0.text(theta[j], r[j], 'Track {:d}'.format(j))
                ax1.text(x[j], y[j], 'Track {:d}'.format(j))

            # 绘制极坐标点迹
            ax0.scatter(theta[j], r[j], c=cycle_colors[j],
                        marker=cycle_markers[j])

            # 为每个点绘制序号/笛卡尔坐标标签
            if i != 0 and i % 10 == 0:
                ax0.text(theta[j], r[j], str(i))

            # 绘制笛卡尔坐标点迹
            ax1.scatter(x[j], y[j], c=cycle_colors[j], marker=cycle_markers[j])

            # 为每个点绘制点迹标号标签
            if i != 0 and i % 10 == 0:
                ax1.text(x[j], y[j], str(i))

        plt.pause(0.5)

    # 绘图展示
    plt.show()


if __name__ == '__main__':
    # tracks = gen_tracks(M=3, N=60, v0=340, a=20, cycle_time=1)
    # plot_tracks('./tracks_2_1s.npy')

    test_track_init_methods('../tracks_2_1s.npy', cycle_time=1, method=2)

    # plot_plots_in_each_cycle('./RadarDataProcessAlg/plots_in_each_cycle_1s.npy')
    # test_track_init_methods_with_bkg('./plots_in_each_cycle_1s.npy',
    #                                  cycle_time=1,
    #                                  method=0)

    # track = gen_track_cv_ca(N=60, v0=340, a=20, cycle_time=1)
    # plot_polar_cartesian_map(track)

    # plot_cartesian_map(track)
    # plot_polar_map(track)

    #
