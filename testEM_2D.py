# _*_coding:utf-8

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def logger(func):
    def inner(*args, **kwargs):
        print('Arguments were: %s, %s' % (args, kwargs))
        return func(*args, **kwargs)
    return inner


param_dict = {}
param_dict['Mu_1'] = np.array([0, 0])
param_dict['Sigma_1'] = np.array([[1, 0], [0, 1]])
param_dict['Mu_2'] = np.array([0, 0])
param_dict['Sigma_2'] = np.array([[1, 0], [0, 1]])
param_dict['Pi_weight'] = 0.5
param_dict['Gamma_list'] = []


def set_param(mu_1, sigma_1, mu_2, sigma_2, pi_weight):
    param_dict['Mu_1'] = mu_1
    param_dict['Mu_1'].shape = (2, 1)
    param_dict['Sigma_1'] = sigma_1
    param_dict['Mu_2'] = mu_2
    param_dict['Mu_2'].shape = (2, 1)
    param_dict['Sigma_2'] = sigma_2
    param_dict['Pi_weight'] = pi_weight


def PDF(data, Mu, sigma):
    '''
    二元高斯分布概率密度函数
    @param data: 一个二位数据点(ndarray)
    @param Mu: 均值(ndarray)
    @param sigma: 协方差矩阵(ndarray)
    @retur: 该数据点的概率密度值
    '''
    sigma_sqrt = math.sqrt(np.linalg.det(sigma))  # 协方差矩阵绝对值的1/2次方
    sigma_inv = np.linalg.inv(sigma)  # 协方差矩阵的逆
    data.shape = (2, 1)
    Mu.shape = (2, 1)

    minus_mu = data - Mu
    minus_mu_trans = np.transpose(minus_mu)

    res = (1.0 / (2.0 * math.pi * sigma_sqrt)) \
        * math.exp((-0.5) * (np.dot(np.dot(minus_mu_trans, sigma_inv), minus_mu)))

    return res


def E_step(Data):
    '''
    E-step: compute responsibilities
    计算本轮的Gamma_list
    @param Data: 一系列二维数据点 
    @return: Gamma_list
    '''

    # 协方差矩阵
    sigma_1 = param_dict['Sigma_1']
    sigma_2 = param_dict['Sigma_2']

    # 权重
    pw = param_dict['Pi_weight']

    # 均值
    mu_1 = param_dict['Mu_1']
    mu_2 = param_dict['Mu_2']

    param_dict['Gamma_list'] = []

    for point in Data:
        gamma_i = pw * PDF(point, mu_2, sigma_2) \
            / (pw * PDF(point, mu_2, sigma_2) + (1.0 - pw) * PDF(point, mu_1, sigma_1))
        param_dict['Gamma_list'].append(gamma_i)


def M_step(Data):
    '''
    M_step: compute weighted means and variance
    更新均值与协方差矩阵
    此例中， gamma_i对应Mu_2, Var_2
    (1 - gamma_i)对应Mu_1, Var_1
    @param Data: 一系列二维数据点 
    '''
    N_1, N_2 = 0.0, 0.0
    for gamma in param_dict['Gamma_list']:
        N_1 += 1.0 - gamma
        N_2 += gamma

    # 更新miu(μ)
    new_mu_1 = np.array([0, 0])
    new_mu_2 = np.array([0, 0])
    for (i, gamma) in enumerate(param_dict['Gamma_list']):
        new_mu_1 += + Data[i] * (1.0 - gamma) / N_1
        new_mu_2 += + Data[i] * gamma / N_2

    # numpy对一维向量无法转置，必须指定shape
    new_mu_1.shape = (2, 1)
    new_mu_2.shape = (2, 1)

    # 更新sigma(Σ)
    new_sigma_1 = np.array((2, 2))
    new_sigma_2 = np.array((2, 2))
    for i in range(len(param_dict['Gamma_list'])):
        gamma = param_dict['Gamma_list'][i]
        X = np.array([[Data[i][0]], [Data[i][1]]])
        new_sigma_1 += np.dot((X - new_mu_1), (X - new_mu_1).transpose()) \
            * (1.0 - gamma) / N_1
        new_sigma_2 += np.dot((X - new_mu_2), (X - new_mu_2).transpose()) \
            * gamma / N_2

    # 更新权重
    new_pi = N_2 / len(Data)

    # 将更新参数写回dict
    param_dict['Mu_1'] = new_mu_1
    param_dict['Mu_2'] = new_mu_2
    param_dict['Sigma_1'] = new_sigma_1
    param_dict['Sigma_2'] = new_sigma_2
    param_dict['Pi_weight'] = new_pi


def EM_iterate(iter_time, Data,
               mu_1, sigma_1, mu_2, sigma_2, pi_weight, esp=0.0001):
    '''
    EM算法迭代运行
    @param iter_time: 迭代次数，若为None则迭代值约束esp为止
    @param Data: 数据
    @param esp: 终止约束
    '''
    set_param(mu_1, sigma_1, mu_2, sigma_2, pi_weight)
    if iter_time == None:
        while True:
            old_mu_1 = param_dict['Mu_1'].copy()
            old_mu_2 = param_dict['Mu_2'].copy()
            E_step(Data)
            M_step(Data)
            delta_1 = param_dict['Mu_1'] - old_mu_1
            delta_2 = param_dict['Mu_2'] - old_mu_2
            if math.fabs(delta_1[0]) < esp and math.fabs(delta_1[1]) < esp \
                    and math.fabs(delta_2[0]) < esp and math.fabs(delta_2[1]) < esp:
                break
    else:
        for i in range(iter_time):
            pass


@logger
def EM_iterate_trajectories(iter_time, Data,
                            mu_1, sigma_1, mu_2,
                            sigma_2, pi_weight, esp=0.001):
    '''
    EM算法迭代运行，同时画出两个均值点的的轨迹
    @param iter_time: 迭代次数，若为None则迭代至约束esp为止
    @param Data: 数据
    @param esp: 终止约束条件 
    '''
    mean_trace_1 = [[], []]
    mean_trace_2 = [[], []]

    set_param(mu_1, sigma_1, mu_2, sigma_2, pi_weight)
    count = 0

    print('--Start\n')
    if iter_time == None:
        while True:
            count += 1
            print('\n-- Round %d' % count)
            old_mu_1 = param_dict['Mu_1'].copy()
            old_mu_2 = param_dict['Mu_2'].copy()

            # E-M
            E_step(Data)
            M_step(Data)

            delta_1 = param_dict['Mu_1'] - old_mu_1
            delta_2 = param_dict['Mu_2'] - old_mu_2

            mean_trace_1[0].append(param_dict['Mu_1'][0][0])  # 因为Mu是2行1列的形式
            mean_trace_1[1].append(param_dict['Mu_1'][1][0])
            mean_trace_2[0].append(param_dict['Mu_2'][0][0])
            mean_trace_2[1].append(param_dict['Mu_2'][1][0])
            if math.fabs(delta_1[0]) < esp and math.fabs(delta_1[1]) < esp \
                    and math.fabs(delta_2[0]) < esp and math.fabs(delta_2[1]) < esp:
                break

            # show iteration result
            print('Mu_1:\n', param_dict['Mu_1'])
            print('Mu_2:\n', param_dict['Mu_2'])
            print('Sigma_1:\n', param_dict['Sigma_1'])
            print('Sigma_2:\n', param_dict['Sigma_2'])
            print('Pi_weight:\n',
                  param_dict['Pi_weight'], ',', 1.0 - param_dict['Pi_weight'])
    else:
        for i in range(iter_time):
            count += 1
            print('\n-- Round %d' % count)
            old_mu_1 = param_dict['Mu_1'].copy()
            old_mu_2 = param_dict['Mu_2'].copy()
            E_step(Data)
            M_step(Data)
            delta_1 = param_dict['Mu_1'] - old_mu_1
            delta_2 = param_dict['Mu_2'] - old_mu_2

            mean_trace_1[0].append(param_dict['Mu_1'][0][0])  # 因为Mu是2行1列的形式
            mean_trace_1[1].append(param_dict['Mu_1'][1][0])
            mean_trace_2[0].append(param_dict['Mu_2'][0][0])
            mean_trace_2[1].append(param_dict['Mu_2'][1][0])

            # show iteration result
            print('Mu_1:\n', param_dict['Mu_1'])
            print('Mu_2:\n', param_dict['Mu_2'])
            print('Sigma_1:\n', param_dict['Sigma_1'])
            print('Sigma_2:\n', param_dict['Sigma_2'])
            print('Pi_weight:\n',
                  param_dict['Pi_weight'], ',', 1.0 - param_dict['Pi_weight'])

    # 可视化
    plt.subplot(121)
    plt.xlabel('eruptions')
    plt.ylabel('waiting')
    plt.plot(mean_trace_1[0], mean_trace_1[1], 'c<')

    plt.subplot(122)
    plt.xlabel('eruptions')
    plt.ylabel('waiting')
    plt.plot(mean_trace_2[0], mean_trace_2[1], 'c>')
    plt.tight_layout()
    plt.show()
    return mean_trace_1, mean_trace_2


def EM_iterate_times(Data, mu_1, sigma_1, mu_2, sigma_2, pi_weight, esp=0.0001):
    # 返回迭代次数
    set_param(mu_1, sigma_1, mu_2, sigma_2, pi_weight)
    iter_times = 0
    print('-- Start...')
    while True:
        iter_times += 1
        print('\n-- Round %d' % iter_times)
        old_mu_1 = param_dict['Mu_1'].copy()
        old_mu_2 = param_dict['Mu_2'].copy()
        E_step(Data)
        M_step(Data)
        delta_1 = param_dict['Mu_1'] - old_mu_1
        delta_2 = param_dict['Mu_2'] - old_mu_2
        if math.fabs(delta_1[0]) < esp and math.fabs(delta_1[1]) < esp \
                and math.fabs(delta_2[0]) < esp and math.fabs(delta_2[1]) < esp:
            break

        # show iteration result
        print('Mu_1:\n', param_dict['Mu_1'])
        print('Mu_2:\n', param_dict['Mu_2'])
        print('Sigma_1:\n', param_dict['Sigma_1'])
        print('Sigma_2:\n', param_dict['Sigma_2'])
        print('Pi_weight:\n', param_dict['Pi_weight'])
    return iter_times


@logger
def task_1(iter_num=None):
    # 读取数据,猜初始值,运行算法
    Data_list = []
    with open('f:/cluster.txt', 'r') as fh:
        for line in fh.readlines():
            point = []
            point.append(float(line.strip().split()[0]))
            point.append(float(line.strip().split()[1]))
            Data_list.append(point)
    Data = np.array(Data_list)  # turn list into numpy array
    print('=> Data:\n', Data)

    Mu_1 = np.array([20, 25])
    Sigma_1 = np.array([[5, 0], [0, 5]])
    Mu_2 = np.array([25, 20])
    Sigma_2 = np.array([[5, 0], [0, 5]])
    Pi_weight = 0.39

    trace_1, trace_2 = EM_iterate_trajectories(iter_num, Data, Mu_1, Sigma_1,
                                               Mu_2, Sigma_2, Pi_weight)

    types = []
    with open('f:/class.txt', 'r') as fh:
        for line in fh.readlines():
            types.append(int(line.strip().split()[0]))
    pts = []
    with open('f:/cluster.txt', 'r') as fh:
        for line in fh.readlines():
            x = line.strip().split()[0]
            y = line.strip().split()[1]
            pts.append((x, y))
    data = np.array(pts)
    plt.scatter(data[:, 0], data[:, 1], c=types)  # scatter方法详解？

    # plot trace
    # plt.plot(trace_1[0], trace_1[1], 'r>')
    # plt.plot(trace_2[0], trace_2[1], 'r<')
    plt.show()


def task_2():
    '''
    执行50次,观察迭代次数的分布
    这里，协方差矩阵都取[[10, 0], [0, 10]]
    mean值在一定范围内随机生成50组数
    '''

    # 读取数据，猜初始值，运行算法
    Data_list = []
    with open('f:/cluster.txt', 'r') as fh:
        for line in fh.readlines():
            point = []
            point.append(float(line.strip().split()[0]))
            point.append(float(line.strip().split()[1]))
            Data_list.append(point)
    Data = np.array(Data_list)  # turn list into numpy array
    print('Data:\n', Data)

    try:
        Mu_1 = np.array([25, 25])
        Sigma_1 = np.array([[5, 0], [0, 5]])
        Mu_2 = np.array([[30, 25]])
        Sigma_2 = np.array([[5, 0], [0, 5]])
        Pi_weight = 0.35

        iter_times = EM_iterate_times(
            Data, Mu_1, Sigma_1, Mu_2, Sigma_2, Pi_weight)
        print('iter_times:', iter_times)
    except Exception as e:
        print(e)


task_1()
# task_2()


# ref:
# http://blog.csdn.net/xiaopangxia/article/details/53542666
# <<EM算法推导与GMM训练应用>>
