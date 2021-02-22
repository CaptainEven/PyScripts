# encoding=utf-8

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from random import sample


# ---------- Parameters ----------
# TrackStates = {
#     'Temporaty': 0,
#     'Reliable':  1,
#     'Fixed':     2,
#     'Potential': 3
# }

TrackStates = {
    0: 'Potential',   # 可能航迹
    1: 'Temporaty',  # 暂时航迹
    2: 'Reliable',   # 可靠航迹
    3: 'Fixed',      # 固定航迹
}


# ---------- Generate Data ----------

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
    if x0 >= 0.0 and y0 >= 0.0:    # 第一象限
        direction = np.random.randint(200, 250)  # 飞往第三象限
    elif x0 >= 0.0 and y0 < 0.0:   # 第四象限
        direction = np.random.randint(110, 160)  # 飞往第二象限
    elif x0 < 0.0 and y0 >= 0.0:   # 第二象限
        direction = np.random.randint(290, 340)  # 飞往第四象限
    elif x0 < 0.0 and y0 < 0.0:    # 第三象限                        #
        direction = np.random.randint(20, 70)    # 飞往第一象限

    # 运动模型: 匀(加)速直线
    # 生成航迹
    track = []
    for i in range(N):
        # ---------- 每个周期都加入一定的随机扰动(噪声)
        # 为航向增加随机噪声扰动
        direction_noise = ((1-(-1)) * np.random.random() + (-1)) * 10
        direction += direction_noise
        direction = direction if direction >= 0.0 else direction + 360.0

        # 为加速度增加随机噪声扰动: 即航速扰动
        a_noise = ((1-(-1)) * np.random.random() + (-1)) * 100
        a += a_noise

        # logging...
        print('Iter {:d} | heading direction: {:.3f}° | acceleration: {:.3f}m/s²'
              .format(i, direction))

        # 一个扫描周期目标状态改变量
        ret = move_in_a_cycle(x0, y0, v0, a, direction, cycle_time)
        if ret == None:
            continue

        # TODO: 加入噪声点迹false positive: 杂波背景

        x, y = ret
        x, y = int(x), int(y)
        if x < -10000 + 1 or y < -10000 + 1 or x > 10000 - 1 or y > 10000 - 1:
            # continue
            # break

            # 保存当前扫描周期的笛卡尔坐标
            track.append([x, y])

            # 更新当前笛卡尔坐标
            x0, y0 = x, y
        else:
            # print(x, y)

            # 保存当前扫描周期的笛卡尔坐标
            track.append([x, y])

            # 更新当前笛卡尔坐标
            x0, y0 = x, y

    return track

# 同时生成几个航迹(track)


def gen_tracks(M=3, N=60, v0=340, a=20, cycle_time=1):
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
        track = gen_track_cv_ca(
            N=N, v0=v0, a=a, direction=direction, cycle_time=1)
        # ----------

        tracks.append(track)
        # print(track, '\n')

    # ---------- 序列化航迹数据到磁盘
    tracks = np.array(tracks)
    # print(tracks)

    # ----- 存为npy文件
    npz_save_path = './tracks'
    np.save(npz_save_path, tracks)
    print('{:s} saved.'.format(npz_save_path))

    # ----- 存为txt文件

    return np.array(tracks)


# ---------- Algorithm
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

    # 计算位移数值
    dist0 = math.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))
    dist1 = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

    # 计算速度
    v0 = dist0 / cycle_time
    v1 = dist1 / cycle_time

    # 计算加速度
    a = (v1 - v0) / cycle_time

    # ----- 计算航向偏转角
    # 计算位移向量
    s0 = np.array([x1, y1]) - np.array([x0, y0])
    s1 = np.array([x2, y2]) - np.array([x1, y1])

    # 计算角度(夹角余弦): 返回反余弦弧度值
    radian = math.acos(np.dot(s0, s1) / (dist0*dist1))

    return v1, a, radian


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
    for i in range(2, N-n):
        # 取滑窗
        window = slide_window(track, n, i)

        # 判定
        n_pass = 0
        for j, plot in enumerate(window):
            if j >= 2:  # 从第三个点迹开始求v, a, angle
                # 获取连续3个点迹
                plots_3 = window[j-2: j+1]  # 3 plots: [j-2, j-1, j]

                # 估算当前点迹的运动状态
                v, a, angle_in_radians = get_v_a_angle(plots_3, cycle_time)

                # 航向偏移角度估算
                angle_in_degrees = math.degrees(angle_in_radians)
                angle_in_degrees = angle_in_degrees if angle_in_degrees >= 0.0 else angle_in_degrees + 360.0

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
    dist = math.sqrt((x_cur - x_pre) * (x_cur - x_pre)
                     + (y_cur - y_pre) * (y_cur - y_pre))

    return dist >= r_min and dist <= r_max


def extrapolat_plot(plot_pre, plot_cur, s):
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

    if x_cur >= x_pre and y_cur >= y_pre:     # 第一象限
        # 计算与x轴夹角
        radian = math.atan2((y_cur - y_pre), (x_cur - x_pre))
        if radian >= 0.0 and radian <= math.pi * 0.5:
            x_extra = x_cur + s * math.cos(radian)
            y_extra = y_cur + s * math.sin(radian)

    elif x_cur < x_pre and y_cur >= y_pre:    # 第二象限
        radian  = math.atan2((y_cur - y_pre), (x_pre - x_cur))
        if radian >= 0.0 and radian <= math.pi * 0.5:
            x_extra = x_cur - s * math.cos(radian)
            y_extra = y_cur + s * math.sin(radian)

    elif x_cur < x_pre and y_cur < y_pre:     # 第三象限
        radian = math.atan2((y_pre - y_cur), (x_pre - x_cur))
        if radian >= 0.0 and radian <= math.pi * 0.5:
            x_extra = x_cur - s * math.cos(radian)
            y_extra = y_cur - s * math.sin(radian)

    elif x_cur >= x_pre and y_cur < y_pre:     # 第四象限
        radian = math.atan2((y_pre - y_cur), (x_cur - x_pre))
        if radian >= 0.0 and radian <= math.pi * 0.5:
            x_extra = x_cur + s * math.cos(radian)
            y_extra = y_cur - s * math.sin(radian)

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
    x_extra, y_extra = extrapolat_plot(plot_pre, plot_cur, s)

    # 计算实际点迹与外推点迹之间的距离
    dist = math.sqrt((x_nex - x_extra)*(x_nex - x_extra) + (y_nex - y_extra)*(y_nex - y_extra))

    return dist <= sigma


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
    for i in range(2, N-n-1):
        # 取滑窗
        window = slide_window(track, n+1, i)

        # 判定
        n_pass = 0
        for j, plot in enumerate(window):
            if j >= 2:  # 从第三个点迹开始求v, a, angle
                # 获取连续3个点迹
                plots_3 = window[j-2: j+1]  # 3 plots: [j-2, j-1, j]

                # 估算当前点迹的运动状态
                v, a, angle_in_radians = get_v_a_angle(plots_3, cycle_time)

                # # 航向偏移角度估算
                # angle_in_degrees = math.degrees(angle_in_radians)
                # angle_in_degrees = angle_in_degrees if angle_in_degrees >= 0.0 else angle_in_degrees + 360.0

                # ----- 判定逻辑...
                if j >= 3 and j < len(window) - 1:  # 从第4次扫描开始逻辑判定: j==3的点迹作为航迹头
                    # 初始波门判定: j是当前判定序列的第二次扫描
                    if start_gate_check(cycle_time, window[j-1], window[j], v0=340):

                        # --- 对通过初始波门判定的航迹建立暂时航迹, 继续判断相关波门
                        # page71-72
                        if relate_gate_check(cycle_time, v, window[j-1], window[j], window[j+1], sigma=sigma):
                            n_pass += 1
                        else:
                            print('Track init failed @cycle{:d}, object(plot) is not in relating gate.'.format(i))
                    else:
                        print('Track init failed @cycle{:d}, object(plot) is not in the starting gate.'
                        .format(i))  
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


def test_track_init_methods(track_f_path, cycle_time, method):
    """
    测试直观法, 逻辑法
    :param track_f_path:
    :param cycle_time:
    :param method:
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

    for i, track in enumerate(tracks):
        if method == 0:  # 直观法
            succeed, start_cycle = direct_method(track,
                                                 cycle_time=cycle_time,
                                                 v_min=200, v_max=400,   # 2M
                                                 a_max=15, angle_max=7,  # 军机7°/s
                                                 m=3, n=4)
        elif method == 1:  # 逻辑法
            succeed, start_cycle = logic_method(track, cycle_time,
                                                sigma=160,
                                                m=3, n=4)

        if succeed:
            print('Track {:d} initialization succeeded @cycle {:d}.'
                  .format(i, start_cycle))
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


def plot_tracks(track_f_path):
    """
    :param track_f_path:
    :return:
    """
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
        theta[neg_inds] += np.pi*2.0

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
    test_track_init_methods('./tracks_2_1s.npy', cycle_time=1, method=1)

    # track = gen_track_cv_ca(N=60, v0=340, a=20, cycle_time=1)
    # plot_polar_cartesian_map(track)

    # plot_cartesian_map(track)
    # plot_polar_map(track)

    #