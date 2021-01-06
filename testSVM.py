#_*_coding:utf-8_*_

'''
1. SVM要点
(1). 支持向量是最靠近分类决策面,最难分类的数据点
(2). SVM是一个优化问题: 最大化支持向量到超平面的距离->最佳超平面
(3). 这个优化问题可以转换成一个线性不等式约束下的二次优化问题
(4). 最优超平面的权重向量是训练样本的线性组合, 且只有支持向量影响最终的划分结果
(5). 如果去掉其他非支持向量的样本重新训练,得到相同的分类超平面
(6). 对于新点 x的预测，只需要计算它与训练数据点的内积即可
(7). 参数w是输入向量的线性组合
(8). SVM原理，为了方便求解，把原始最优化问题转化成了其对偶问题,
因为对偶问题是一个凸二次规划问题,这样的凸二次规划问题具有全局最优解
2. 细节
(1). 坐标上升算法每次更新一个参数, 一次更新所有参数
(2). SMO由于约束条件限制, 每次更新2个参数, 跟坐标上升算法一样, 按顺序不断迭代优化
(3). SMO算法选择同时优化两个参数, 固定其他N-2个参数,假设选择的变量为
,固定其他参数,由于其他参数固定,可以简化目标函数为只关于的二元函数。 
'''

import random
import os
import numpy as np
import matplotlib.pyplot as plt


DIR_PATH = os.path.join(os.getcwd() + os.path.sep
                        + 'tests' + os.path.sep + 'datasets' + os.path.sep)


def load_dataset(f_name):
    '''
    加载数据集
    '''
    data_mat = []
    label_mat = []
    with open(f_name) as fr:
        for line in fr.readlines():
            line_data = line.strip().split('\t')
            data_mat.append([float(line_data[0]), float(line_data[1])])
            label_mat.append(float(line_data[2]))
        return np.mat(data_mat), np.mat(label_mat).T  # 列向量


def load_data(f_path):
    '''
    加载数据
    '''
    dataset, labels = [], []
    with open(f_path, 'r') as fh:
        for line in fh:
            line_data = [float(i) for i in line.strip().split()]
            dataset.append(line_data[:-1])
            labels.append(line_data[-1])
    return dataset, labels


def clip(alpha, L, H):
    '''
    修剪alpha
    '''
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha


def select_j(i, m):
    '''
    在m方位内随机选择除了i之外剩余的数
    '''
    l = list(range(m))
    seq = l[:i] + l[i + 1:]  # 先从列表中排除i
    return random.choice(seq)  # 随机选择一个


# 不懂这是怎么计算的?
def get_w(alphas, dataset, labels):
    '''
    通过已知数据点和朗格朗日乘子获得超平面参数w
    '''
    alphas, dataset, labels = np.array(
        alphas), np.array(dataset), np.array(labels)
    yx = labels.reshape(1, -1).T * np.array([1, 1]) * dataset
    w = np.dot(yx.T, alphas)
    return w.tolist()


def simple_smo(dataset, labels, C, max_iter):
    '''
    简化版本的SMO算法,未使用启发式方法对alpha进行选择
    @param dataset: 输入的数据矩阵(不包括标签)
    @param labels: 数据对应的标签矩阵
    @param C: 软间隔常数, 0 <= alpha_i <= C
    @param max_iter: 外层循环最大迭代次数
    '''
    dataset = np.array(dataset)
    m, n = dataset.shape
    labels = np.array(labels)

    # 初始化参数
    alphas = np.zeros(m)
    b = 0
    it = 0

    def f(x):
        '''
        SVM分类函数 y = w^Tx + b: Kernel function vector(核函数向量)
        '''
        x = np.mat(x).T  # 转换成numpy矩阵
        data = np.mat(dataset)
        ks = data * x

        # predict
        wx = np.mat(alphas * labels) * ks  # alpha与label逐个元素相乘
        fx = wx + b
        return fx[0, 0]  # 0行0列

    all_alphas, all_bs = [], []
    while it < max_iter:
        pair_changed = 0
        for i in range(m):
            a_i, x_i, y_i = alphas[i], dataset[i], labels[i]
            fx_i = f(x_i)
            E_i = fx_i - y_i

            j = select_j(i, m)
            a_j, x_j, y_j = alphas[j], dataset[j], labels[j]
            fx_j = f(x_j)
            E_j = fx_j - y_j

            K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(
                x_j, x_j), np.dot(x_i, x_j)
            eta = K_ii + K_jj - 2 * K_ij
            if eta <= 0:
                print('[warning]: eta <= 0')
                continue

            # 更新alpha对
            a_i_old, a_j_old = a_i, a_j
            a_j_new = a_j_old + y_j * (E_i - E_j) / eta

            # 对alpha进行修剪
            if y_i != y_j:
                L = max(0, a_j_old - a_i_old)
                H = min(C, C + a_j_old - a_i_old)
            else:
                L = max(0, a_i_old + a_j_old - C)
                H = min(C, a_j_old + a_i_old)

            a_j_new = clip(a_j_new, L, H)
            a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)

            if abs(a_j_new - a_j_old) < 0.00001:
                print('[Warning]: alpha_j not moving enough')
                continue

            # 更新alpha
            alphas[i], alphas[j] = a_i_new, a_j_new

            # 更新阈值b
            b_i = -E_i - y_i * K_ii * \
                (a_i_new - a_i_old) - y_j * K_ij * (a_j_new - a_j_old) + b
            b_j = -E_j - y_i * K_ij * \
                (a_i_new - a_i_old) - y_j * K_jj * (a_j_new - a_j_old) + b
            if 0 < a_i_new < C:  # python还可以这样写表达式?
                b = b_i
            elif 0 < a_j_new < C:
                b = b_j
            else:
                b = (b_i + b_j) / 2

            all_alphas.append(alphas)
            all_bs.append(b)

            pair_changed += 1
            print('INFO: iteration: {} i: {} pair_changed: {}'.format(
                it, i, pair_changed))

        if pair_changed == 0:
            it += 1
        else:
            it = 0
        print('iteration number: {}'.format(it))
    return alphas, b


def select_j_rand(i, m):
    '''
    选择一个0~m之间, 跟i不同的整数
    '''
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(alpha, H, L):
    '''
    clamp alpha
    '''
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L
    return alpha


def smo_0(data_mat, label_mat, C, toler, max_iter):
    '''
    smo优化算法的简单实现
    '''
    m, n = data_mat.shape
    b = 0
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alpha_pair_changed = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, label_mat).T *
                        (data_mat * data_mat[i, :].T)) + b
            Ei = fXi - float(label_mat[i])  # 预测值与真实值的误差
            if (label_mat[i] * Ei < -toler and alphas[i] < C) or (label_mat[i] * Ei > toler and alphas[i] > 0):
                j = select_j_rand(i, m)
            fXj = float(np.multiply(alphas, label_mat).T *
                        (data_mat * data_mat[j, :].T)) + b
            Ej = fXj - float(label_mat[j])
            alpha_i_old = alphas[i].copy()
            alpha_j_old = alphas[j].copy()

            # 确定上下界
            if label_mat[i] != label_mat[j]:  # i, j的标签是相同还是相反
                L = max(0, alphas[j] - alphas[i])
                H = min(C, C + alphas[j] - alphas[i])
            else:
                L = max(0, alphas[j] + alphas[i] - C)
                H = min(C, alphas[j] + alphas[i])
            if L == H:
                print('L==H')
                continue
            eta = 2.0 * data_mat[i, :] * data_mat[j, :].T - data_mat[i, :] * \
                data_mat[i, :] * data_mat[i, :].T - \
                data_mat[j, :] * data_mat[j, :]  # eta:松弛变量?
            if eta >= 0.0:
                print('eta>=0')
                continue
            alphas[j] -= label_mat[j] * (Ei - Ej) / eta  # 更新alpha_j
            alphas[j] = clip_alpha(alphas[j], H, L)
            if abs(alphas[j] - alpha_j_old) < 0.00001:
                print('j not moving enough')
                continue
            alphas[i] += label_mat[j] * \
                label_mat[i] * (alpha_j_old - alphas[j])
            b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) \
                * data_mat[i, :] * data_mat[i, :].T - label_mat[j] \
                * (alphas[j] - alpha_j_old) * data_mat[i, :] * data_mat[j, :].T
            b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) \
                * data_mat[i, :] * data_mat[j, :].T - label_mat[j] \
                * (alphas[j] - alpha_j_old) * data_mat[j, :] * data_mat[j, :].T
            if alphas[i] > 0 and alphas[i] < C:
                b = b1
            elif alphas[j] > 0 and alphas[j] < C:
                b = b2
            else:
                b = (b1 + b2) * 0.5
            alpha_pair_changed += 1
            print('iter: %d, i: %d, pair changed %d' %
                  (iter, i, alpha_pair_changed))
        if alpha_pair_changed == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas


# 序列最小化优化的简单实现
def testSMO():
    dataset, labels = load_data(os.path.join(DIR_PATH, 'testSet.txt'))
    print('labels:\n', labels)

    # 训练SVM
    alphas, b = simple_smo(dataset, labels, 0.6, 50)
    print('-- final alphas:\n', alphas)

    # 用训练好的模型分类
    pts_clsed = {'+1': [], '-1': []}  # 用一个dict存放分类结果
    for pt, label in zip(dataset, labels):
        if label == 1.0:
            pts_clsed['+1'].append(pt)
        else:
            pts_clsed['-1'].append(pt)

    # 绘制数据点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label, pts in pts_clsed.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)

    # 绘制分割线
    w = get_w(alphas, dataset, labels)
    x1, _ = max(dataset, key=lambda x: x[0])
    x2, _ = min(dataset, key=lambda x: x[0])
    a1, a2 = w
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    ax.plot([x1, x2], [y1, y2])

    # 绘制支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3:
            x, y = dataset[i]
            ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')
    plt.show()


if __name__ == '__main__':
    testSMO()
    # arr = [1023, 1536, 3165, 9119, 1221]
    # f_out = open('test_save_list.txt', 'w', encoding='utf-8')
    # f_out.write(str(arr))
    # f_out.close()
    print('-- Test done.')


# https://github.com/PytLab/MLBox
# https://wenku.baidu.com/view/d2351a2cb307e87101f6967b.html
# https://www.cnblogs.com/steven-yang/p/5658362.html
# http://pytlab.org/2017/09/01/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E5%AE%9E%E8%B7%B5-SVM%E4%B8%AD%E7%9A%84SMO%E7%AE%97%E6%B3%95/
# http://blog.csdn.net/luoshixian099/article/details/51227754
