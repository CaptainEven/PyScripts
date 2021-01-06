# _*_coding:utf-8_*_
import os
import copy
import uuid
from functools import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import exp

DIR_PATH = os.path.join(os.getcwd() + os.path.sep +
                        os.path.sep + 'datasets' + os.path.sep)
print('DIR: ', DIR_PATH)


def load_data(file_name):
    dataset = []  # 2d array
    with open(file_name, 'r') as f:
        for line in f:
            line_data = [float(data) for data in line.split()]  # 并没有指定分割符
            dataset.append(line_data)
    return dataset


def split_dataset(dataset, feat_id, val):
    '''
    split according to feature index and feature value
    '''
    l_data, r_data = [], []
    for item in dataset:
        if item[feat_id] < val:
            l_data.append(item)
        else:
            r_data.append(item)
    return l_data, r_data


def f_leaf(dataset):
    '''
    计算给定数据的叶节点数值，这里取均值
    '''
    dataset = np.array(dataset)
    return np.mean(dataset[:, -1])  # 最后一列是标签?


def f_err(dataset):
    '''
    计算数据误差
    '''
    dataset = np.array(dataset)
    m, _ = dataset.shape
    return np.var(dataset[:, -1]) * m


def choose_best_feature(dataset, f_leaf, f_err, opt):
    '''
    dataset: 待划分的数据集
    f_leaf: 创建子节点的函数, 怎样计算子节点?
    f_err: 计算数据误差的函数
    opt: 回归树参数: python dict
    err_tolerance: 最小误差下降值
    n_tolerance: 数据切分最小样本数
    '''
    dataset = np.array(dataset)  # 格式化为numpy数组
    m, n = dataset.shape
    err_tolerance, n_tolerance = opt['err_tolerance'], opt['n_tolerance']
    err = f_err(dataset)  # 计算误差
    best_feat_id, best_feat_val, best_err = 0, 0, float('inf')  # 初始化

    # 遍历所有特征
    for feat_id in range(n - 1):
        vals = dataset[:, feat_id]  # 该特征值列表

        # 遍历特征列表中每一项
        for val in vals:
            # 按照当前特征和特征值分割数据
            l_data, r_data = split_dataset(dataset.tolist(), feat_id, val)
            if len(l_data) < n_tolerance or len(r_data) < n_tolerance:
                continue  # 如果切分的样本量小于阈值

            # 计算误差
            new_err = f_err(l_data) + f_err(r_data)  # 左、右子树误差累加
            if new_err < best_err:  # 更新最小误差
                best_feat_id = feat_id
                best_feat_val = val
                best_err = new_err

    # 如果误差(方差)变化不大归为一类(子树)
    if abs(err - best_err) < err_tolerance:
        return None, f_leaf(dataset)

    # 检查分割样本是不是太小: 划分到叶子结点
    l_data, r_data = split_dataset(
        dataset.tolist(), best_feat_id, best_feat_val)
    if len(l_data) < n_tolerance or len(r_data) < n_tolerance:
        return None, f_leaf(dataset)
    return best_feat_id, best_feat_val


# 用一个python字典存放树结构
def create_tree(dataset, f_leaf, f_err, opt=None):
    '''
    递归实现回归决策树
    dataset: 待划分的数据集
    f_leaf： 创建叶子结点的函数
    f_err: 计算数据误差的函数
    opt: 回归树参数
    err_tolerance: 最小误差下降值
    n_tolerance: 数据切分的最小样本数
    '''
    if opt is None:  # 初始化参数
        opt = {'err_tolerance': 1, 'n_tolerance': 4}

    # 选择最优特征和特征值
    feat_id, val = choose_best_feature(dataset, f_leaf, f_err, opt)

    # 递归终止条件
    if feat_id is None:
        return val

    # 没有达到终止条件，创建本节点的回归树，并创建子决策树
    tree = {'feat_id': feat_id, 'feat_val': val}

    l_data, r_data = split_dataset(dataset, feat_id, val)  # 在本节点划分
    l_tree = create_tree(l_data, f_leaf, f_err, opt) # 创建子树
    r_tree = create_tree(r_data, f_leaf, f_err, opt)
    tree['left'] = l_tree
    tree['right'] = r_tree

    return tree


# ------------------postprune:后剪枝
def is_leaf(tree):
    '''
    判断一个节点是否是叶子结点
    不是叶子结点即子树
    '''
    return type(tree) is not dict


def collapse(tree):
    '''
    对树进行塌陷处理, 均值取代子树
    '''
    if is_leaf(tree):
        return tree
    else:
        return (collapse(tree['left']) + collaps(tree['right'])) * 0.5


def post_prune(tree, test_data):
    '''
    根据测试数据集对训练好的树结构进行后剪枝
    '''
    if is_leaf(tree):
        return tree

    # 若没有传入测试数据集, 返回塌陷值
    if not test_data:
        return collapse(tree)

    # 比较叶子结点合并前后的方差
    l_tree, r_tree = tree['left'], tree['right']
    if is_leaf(l_tree) and is_leaf(r_tree):
        collapse_val = (l_tree + r_tree) * 0.5

        # 分割测试数据集,用于计算合并前后方差
        l_data, r_data = split_dataset(
            test_data, tree['feat_id'], tree['feat_val'])

        # 分别计算合并前后的方差
        err_split = np.sum((np.array(l_data) - l_tree) ** 2) \
            + np.sum((np.array(r_data) - r_tree) ** 2)
        err_merge = np.sum((np.array(test_data) - collapse_val) ** 2)

        # 判定是否进行合并剪枝处理
        if err_split > err_merge:
            print('[{}, {}] merged into {}'.format(
                l_tree, r_tree, collapse_val))
            return collapse_val
        else:
            return tree  # 不合并

    # 非叶子结点, 递归, 递归处理节点与子节点的关系: 更新分支
    tree['left'] = post_prune(l_tree, test_data)
    tree['right'] = post_prune(r_tree, test_data)
    return tree


def get_nodes_edges(tree, root_node=None):
    '''
    返回决策树所有节点和边
    '''
    Node = namedtuple('Node', ['id', 'label'])  # 使用名称索引,避免内存越界
    Edge = namedtuple('Edge', ['start', 'end'])

    nodes, edges = [], []

    if type(tree) is not dict:  # 递归终止条件,决策树不可继续划分
        return nodes, edges

    if root_node is None:  # 根节点为空,则从空节点开始
        # 难道Tree的ID即本节点的feature ID?
        label = '{}: {}'.format(tree['feat_id'], tree['feat_val'])
        root_node = Node._make([uuid.uuid4(), label])  # 从 iterable 对象中创建新的实例
        nodes.append(root_node)

    for sub_tree in [tree['left'], tree['right']]:  # 遍历左右子树
        if type(sub_tree) is dict:  # 非叶子结点
            node_label = '{}: {}'.format(
                sub_tree['feat_id'], sub_tree['feat_val'])
        else:
            node_label = '{}'.format(sub_tree)  # 叶子结点, 叶子节点的值
        sub_node = Node._make([uuid.uuid4(), node_label])  # 创建子节点
        nodes.append(sub_node)  # 插入子节点

        edge = Edge._make([root_node, sub_node])  # 创建边
        edges.append(edge)  # 插入边

        # 递归子树, 获取子树的节点和边
        sub_nodes, sub_edges = get_nodes_edges(sub_tree, root_node=sub_node)
        nodes.extend(sub_nodes)
        edges.extend(sub_edges)

    return nodes, edges


# 格式化回归决策树的输出，为可视化做准备
def dotify(tree):
    '''
    获取树Graphviz Dot文件的内容
    '''
    content = 'digraph decision_tree {\n'
    nodes, edges = get_nodes_edges(tree)

    for node in nodes:
        content += '    "{}" [label="{}"];\n'.format(node.id, node.label)
    for edge in edges:
        start, end = edge.start, edge.end
        content += '    "{}" -> "{}";\n'.format(start.id, end.id)
    content += '}'
    return content


# 通过训练好的回归决策树模型进行预测
def tree_predict(data, tree):
    '''
    根据模型预测数值
    '''
    # 递归终止条件
    if type(tree) is not dict:
        return tree  # 叶子结点的预测

    feat_id, feat_val = tree['feat_id'], tree['feat_val']
    if data[feat_id] < feat_val:
        sub_tree = tree['left']
    else:
        sub_tree = tree['right']

    # 递归
    return tree_predict(data, sub_tree)


#---------------------------------Model tree(模型树)
# 回归树用二叉树的原因在于: 根本不知道该分为几类
def linear_regression_ana(dataset):
    '''
    最小二乘法解析解
    '''
    dataset = np.matrix(dataset)

    # 分割数据,添加常数列
    X_ori, y = dataset[:, :-1], dataset[:, -1]  # 将特征数据与标签分割
    X_ori, y = np.matrix(X_ori), np.matrix(y)
    m, n = X_ori.shape
    X = np.matrix(np.ones((m, n + 1)))  # 将矩阵初始化为1
    X[:, 1:] = X_ori  # 用原始数据填充矩阵后面的列

    # 通过最小二乘矩阵运算得到解析解
    w = (X.T * X).I * X.T * y  # 回归系数(权重向量)
    print('w:\n', w)
    return w, X, y


# 如何使用numpy matrix避免python for循环?
def LR_grad_sgd(dataset, alpha=0.05, iter_num=20):
    '''
    随机梯度下降
    '''
    dataset = np.matrix(dataset)

    # 分割数据, 添加常数列
    X_ori, y = dataset[:, :-1], np.array(dataset[:, -1])
    m, n = X_ori.shape
    X = np.ones((m, n + 1))  # X 是m行(n+1)列的
    X[:, 1:] = X_ori  # 用原始数据填充矩阵后面的列

    # 初始化权重向量
    w = np.ones(n + 1)

    # 随机梯度下降
    for k in range(iter_num):
        for i in range(m):
            w -= alpha * (np.dot(X[i], w) - y[i]) * X[i]

    w.shape = (n + 1, 1)
    print('w:\n', w)
    return np.asmatrix(w), X, y  # 将w转换成矩阵


# 让学习率刚开始大,后来越来越小
# 如何用numpy矩阵运算代替python循环
def LR_grad_bgd(dataset, alpha=0.1, max_iter=1300):
    '''
    批量梯度下降
    '''
    dataset = np.matrix(dataset)

    # 分割数据, 添加常数列
    X_ori, y = dataset[:, :-1], np.array(dataset[:, -1])
    m, n = X_ori.shape
    X = np.ones((m, n + 1))  # X 是m行(n+1)列的
    X[:, 1:] = X_ori  # 用原始数据填充矩阵后面的列

    # 初始化权重向量
    w = np.ones(n + 1)

    # 初始化梯度
    grad = 0.0
    for i in range(m):
        grad += (np.dot(X[i], w) - y[i]) * X[i]
    grad *= alpha / m
    w -= grad
    dist = np.linalg.norm(grad)

    # 要么达到最大迭代次数,要么梯度变化已经很小
    k = 1
    while k < max_iter and dist > 1.5e-4:  # 1.0e-4
        # 计算累计梯度
        grad = 0.0
        for i in range(m):
            grad += (np.dot(X[i], w) - y[i]) * X[i]
        grad *= alpha / m
        w -= grad
        # print('w:\n', w)
        dist = np.linalg.norm(grad)
        # print('dist: ', dist, '\n')
        k += 1

    w.shape = (n + 1, 1)
    print('w:\n', w)
    return np.asmatrix(w), X, y  # 将w转换成矩阵


# numpy矩阵运算代替python循环
def LR_bgd(dataset, alpha=0.006, max_iter=185):
    '''
    批量梯度下降
    '''
    dataset = np.matrix(dataset)

    # 分割数据,添加常数列
    X_ori, y = dataset[:, :-1], dataset[:, -1]  # 将特征数据与标签分割
    X_ori, y = np.matrix(X_ori), np.matrix(y)  # .reshape(-1, 1) # y是个列向量(标签)
    # print('y:\n', y)
    m, n = X_ori.shape
    X = np.matrix(np.ones((m, n + 1)))  # 将矩阵初始化为1
    X[:, 1:] = X_ori  # 用原始数据填充矩阵后面的列

    # 初始化权重向量
    w = np.ones((n + 1, 1))

    err = X * w - y
    grad = alpha * X.T * err
    k = 0
    while k < max_iter:
        err = X * w - y
        grad = alpha * X.T * err
        w -= grad
        k += 1
    # print('final grad_norm: ', np.linalg.norm(grad))
    print('\nw:\n', w)
    return w, X, y


def f_leaf_lr_ana(dataset):
    '''
    计算给定数据集的线性回归系数: 解析解
    '''
    w, _, _ = linear_regression_ana(dataset)
    return w


def f_leaf_lr_grad_sgd(dataset):
    '''
    计算给定数据集的线性回归系数: 解析解
    '''
    w, _, _ = LR_grad_sgd(dataset)
    return w


def f_leaf_lr_grad_bgd(dataset):
    '''
    计算给定数据集的线性回归系数
    '''
    w, _, _ = LR_grad_bgd(dataset)
    return w


def f_leaf_lr_bgd(dataset):
    '''
    矩阵运算的批量梯度下降,计算回归权重向量
    '''
    w, _, _ = LR_bgd(dataset)
    return w


def f_err_lr_ana(dataset):
    '''
    计算给定数据集的线性回归误差:解析解
    '''
    w, X, y = linear_regression_ana(dataset)
    y_prime = X * w
    return np.var(y - y_prime)


def f_err_lr_grad_sgd(dataset):
    '''
    计算给定数据集的线性回归误差:解析解
    '''
    w, X, y = LR_grad_sgd(dataset)
    y_prime = X * w
    return np.var(y - y_prime)


def f_err_lr_grad_bgd(dataset):
    '''
    计算给定数据集的线性回归误差:解析解
    '''
    w, X, y = LR_grad_bgd(dataset)
    y_prime = X * w
    return np.var(y - y_prime)


def f_err_lr_bgd(dataset):
    '''
    '''
    w, X, y = LR_bgd(dataset)
    y_prime = X * w
    return np.var(y - y_prime)


def tree_predict_mt(data, tree):
    '''
    通过模型树进行预测
    '''
    if type(tree) is not dict:
        w = tree
        y = np.matrix(data) * np.matrix(w)
        return y[0, 0]

    feat_id, feat_val = tree['feat_id'], tree['feat_val']
    if data[feat_id + 1] < feat_val:
        return tree_predict_mt(data, tree['left'])
    else:
        return tree_predict_mt(data, tree['right'])


def re_draw(tol_s, tol_n):
    pass


def draw_new_tree():
    pass


def testTK():
    '''
    测试TK GUI编程
    '''
    # root = Tk()
    # my_label = Label(root, text='Hello world!')
    # my_label.grid()
    # root.mainloop()

    root = Tk()
    Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)
    Label(root, text='tolN').grid(row=1, column=0)
    tolN_entry = Entry(root)
    tolN_entry.grid(row=1, column=1)
    tolN_entry.insert(0, '10')
    Label(root, text='tolS').grid(row=2, column=0)

    tolS_entry = Entry(root)
    tolS_entry.grid(row=2, column=1)
    tolS_entry.insert(0, '1.0')
    Button(root, text='ReDraw', command=draw_new_tree).grid(
        row=1, column=2, rowspan=3)

    chk_btn_var = IntVar()
    chk_btn = Checkbutton(root, text='Model Tree', variable=chk_btn_var)
    chk_btn.grid(row=3, column=0, columnspan=2)

    re_draw.raw_data = np.mat()


# 什么叫做无偏估计?
def get_coeff(data_1, data_2):
    '''
    计算相关系数矩阵
    '''
    cov = np.mean(data_1 * data_2) - np.mean(data_1) * np.mean(data_2)
    return cov / (np.var(data_1) * np.var(data_2)) ** 0.5

# 局部加权回归就是一种非参数学习算法，
# 非参数学习算法的定义: 一个参数数量会随m(训练集大小)增长的算法。
# 通常定义为参数数量随m线性增长。


def lwlr(the_x, X, Y, k):
    '''
    局部加权线性回归
    θ=(XTWX)−1XTWy
    '''
    m = X.shape[0]  # m个数据

    # 创建针对x的权重矩阵: 只有对角线上的数据?
    the_x = np.array(the_x)
    W = np.mat(np.zeros((m, m)))
    for i in range(m):
        xi = np.array(X[i][0])
        W[i, i] = exp((np.linalg.norm(the_x - xi)) / (-2.0 * k**2))

    # 计算基于此点的回归系数
    xWx = X.T * W * X  # n*n
    if np.linalg.det(xWx) == 0:
        print('xWx is a sigular matrix')
        return
    w = xWx.I * X.T * W * Y
    return w


def testTreeRegress():
    f_path = os.path.join(DIR_PATH, 'ex2.txt')
    # print('file_path: ', f_path)
    dataset = load_data(f_path)
    # print('dataset:\n', dataset)
    tree = create_tree(dataset, f_leaf, f_err, opt={'n_tolerance': 4, 'err_tolerance': 1})

    tree_pre_pruned = create_tree(dataset, f_leaf, f_err, opt={'n_tolerance': 10,
                                                               'err_tolerance': 30})

    # dot_file = '{}.dot'.format(f_path.split('.')[0])
    # with open(dot_file, 'w') as f:
    #     content = dotify(tree)
    #     f.write(content)

    # 加载测试数据集
    f_test_path = os.path.join(DIR_PATH, 'ex2test.txt')
    test_data = load_data(f_test_path)

    # 剪枝处理
    tree_post_pruned = post_prune(tree_pre_pruned, test_data)

    # dot_file_pru = '{}.dot'.format(f_path.split('.')[0] + '_pru')
    # with open(dot_file_pru, 'w') as f:
    #     content = dotify(tree_post_pruned)
    #     f.write(content)

    dataset = np.array(dataset)

    # 绘制散点
    plt.scatter(dataset[:, 0], dataset[:, 1])

    # 绘制回归曲线
    x = np.linspace(0, 1, 50)
    y_1 = [tree_predict([i], tree) for i in x]
    y_2 = [tree_predict([i], tree_post_pruned) for i in x]
    y_diff = [tree_predict([i], tree_post_pruned) -
              tree_predict([i], tree) for i in x]
    plt.plot(x, y_1, c='b')
    plt.plot(x, y_2, c='r')
    plt.plot(x, y_diff, c='g')
    plt.show()

    img_path = os.path.join(DIR_PATH, 'ex_2_tree_pru.png')
    print('img_path: ', img_path)
    img = Image.open(img_path)
    img.show()

    #------------------------------测试模型树
    # mt_f_path = os.path.join(DIR_PATH, 'exp2.txt')
    # dataset = load_data(mt_f_path)
    # # print('dataset:\n', dataset)

    # tree = create_tree(dataset,
    #                    f_leaf_lr_bgd, f_err_lr_bgd,
    #                    opt={'err_tolerance': 0.1, 'n_tolerance': 4})

    # # 生成模型树dot文件
    # dot_f_path = os.path.join(DIR_PATH, 'exp2.dot')
    # with open(dot_f_path, 'w') as f:
    #     f.write(dotify(tree))

    # # img_path = os.path.join(DIR_PATH, 'exp2_dot.png')
    # # print('img_path: ', img_path)
    # # img = Image.open(img_path)
    # # img.show()

    # dataset = np.array(dataset)

    # # 绘制散点图
    # plt.scatter(dataset[:, 0], dataset[:, 1])

    # # 绘制回归曲线
    # x = np.sort(dataset[:, 0])
    # y = [tree_predict_mt([1.0] + [i], tree) for i in x]

    # # 计算相关系数(归一化的协方差)
    # coef = get_coeff(dataset[:, 1],
    #                  [tree_predict_mt([1.0] + [i], tree) for i in dataset[:, 0]])
    # print('--Correlation coefficient: ', coef)

    # plt.plot(x, y, c='r')
    # plt.show()
    # print('--Test done')


def load_dataset(file_path):
    ''' 加载数据
    '''
    X, Y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            splited_line = [float(i) for i in line.strip().split()]
            x, y = splited_line[: -1], splited_line[-1]
            X.append(x)
            Y.append(y)
    X, Y = np.mat(X), np.mat(Y).T
    return X, Y


def testLwlr():
    '''
    测试局部加权线性回归
    '''
    # 加载数据
    f_path = os.path.join(DIR_PATH, 'ex0.txt')
    k = 0.12
    X, Y = load_dataset(f_path)
    X = np.array(X)  # 将x按照从小到大排序
    Y = np.array(Y)

    # 训练每一个数据的权重向量
    y_predict = []
    for the_x in X:
        w = lwlr(the_x, X, Y, k).reshape(1, -1).tolist()[0]
        print('w: ', w)
        y_predict.append(np.dot(the_x, w))

    coef = get_coeff(np.array(Y.reshape(1, -1)), np.array(y_predict))
    print('--Coefficient: {}'.format(coef))

    # 可视化
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = X
    y = Y.reshape(1, -1).tolist()[0]
    ax.scatter(x, y)

    # zip是为了按照元组的第一项排序, 反zip是为了将排序后的元组数组拆开
    # zip与反zip
    x, y = list(zip(*sorted(zip(x, y_predict), key=lambda x: x[0])))
    ax.plot(x, y, c='r')
    plt.show()


def testBubbleSort():
    '''
    测试冒泡排序
    '''
    arr = np.array([1, 9, 6, 7, 5, 3, 8, 4, 2])
    print('--Before sort: ', arr)
    SIZE = len(arr)
    for i in range(SIZE):  # 冒泡轮数
        for j in range(SIZE - i - 1):  # 该轮的交换次数
            if arr[j] > arr[j + 1]:
                temp = arr[j + 1]
                arr[j + 1] = arr[j]
                arr[j] = temp
    print('--After sort: ', arr)


# http://www.pytorchtutorial.com/pytorch-sequence-model-and-lstm-networks/ (词性标注)
# http://blog.csdn.net/qq_25762497/article/details/51052861


if '__main__' == __name__:
    # testTK()
    testTreeRegress()
    # testLwlr()
    # testBubbleSort()

# pre-prune:预剪枝是在生成决策树之前通过改变参数然后在树生成的过程中进行的
# post-prune后剪枝则是通过测试数据来自动进行剪枝,
# 不需要用户干预因此是一种更理想的剪枝技术，但是我们需要写剪枝函数来处理

# 为什么要找分割后方差最小的分割点作为最佳分割点呢？
# (1).
# 一条信息的信息量与其不确定性有着直接的的关系
# 信息量等于不确定性的多少,越不确定,包含的信息量越大
# 方差描述数据的离散程度,方差越大,数据越离散,越不一致,
# 模型越可能过拟合,模型识别出来的模式不具有一致性
# (2).
# 相比于方差，熵更适合描述信息的不确定度,方差在某些前提下是可以描述信息的不确定性的
# 在哪些条件下，方差可以描述信息的不确定性呢
# 熵和ln(σ)有很强的正相关的关系：x线性关系
# 当模型是非凸的，存在很多局部极值的时候，方差对信息不确定度的描述能力降低了
# 信息熵: H(x)=−∑pilog(pi)
# 方差公式: σ2=1n∑i=1n(xi−μ)2
# 方差描述变量的离散程度，信息熵描述变量的不确定程度。两者虽然有一定的联系但并不等价。
# http://python.jobbole.com/88822/
# https://zhuanlan.zhihu.com/p/30169110 (异常值检测)
# https://github.com/PytLab/MLBox
