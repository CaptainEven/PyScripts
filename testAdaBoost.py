#_*_coding:utf-8_*_
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from math import log

DIR_PATH = os.path.join(os.getcwd() + os.path.sep
                        + 'tests' + os.path.sep + 'datasets' + os.path.sep)


def load_data_sim():
    '''
    加载数据集: 简单测试
    '''
    data_mat = np.matrix([[1.0, 2.1],
                          [2.0, 1.1],
                          [1.3, 1.0],
                          [1.0, 1.0],
                          [2.0, 1.0]])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, labels


def load_dataset(f_path):
    '''
    加载数据集
    '''
    feat_num = len(open(f_path).readline().split('\t'))

    data_mat = []
    label_mat = []
    fr = open(f_path)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(feat_num - 1):
            line_arr.append(float(cur_line[i])) # instance features
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1])) # instance label
    fr.close()
    return np.mat(data_mat), np.mat(label_mat)


def stump_classify(data_mat, dim, TH, th_ineq):
    '''
    简单分类器
    '''
    ret_arr = np.ones((data_mat.shape[0], 1))  # 所有的初始化为+1
    if th_ineq == 'lt':  # 不等式选项(lt: less than, gt: greater than)
        ret_arr[data_mat[:, dim] <= TH] = -1.0
    else:
        ret_arr[data_mat[:, dim] > TH] = -1.0
    return ret_arr


# 错误向量跟数据权重向量越<远离>分类效果越好(向量点乘数值越小)：
# 因为AdaBoost要求数据权重向量D满足：错误分类的数据权重大
def build_stump(data_arr, labels, D):
    '''
    base classifier: 有了数据权重如何训练下一级分类器?
    单层决策树: decesion stump, 获取给定数据权重向量D下的最佳分割参数
    '''
    # 数据转换成numpy矩阵形式
    data_mat = np.mat(data_arr)
    label_mat = np.mat(labels).T
    m, n = data_mat.shape
    step_num = 10.0  # 数据每一个特征维度都分为10份
    best_stump = {}
    best_label_est = np.mat(np.zeros((m, 1)))
    min_err = np.inf  # 初始化误差为极大值
    for i in range(n):  # 遍历数据的每一个特征维度
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / step_num
        for j in range(-1, int(step_num) + 1):
            for in_eql in ['lt', 'gt']:
                TH = (range_min + float(j) * step_size)  # 分割阈值
                # if TH == 0.0:
                #     continue
                predic_vals = stump_classify(data_mat, i, TH, in_eql)
                err_arr = np.mat(np.ones((m, 1)))  # 初始化错判向量为1
                err_arr[predic_vals == label_mat] = 0  # 判断正确的赋值为0
                weighted_err = D.T * err_arr  # <Adaboost:向量点乘>
                # print('split: dim %d, TH: %.2f, in_eql: %s, the weighted error is %.3f'
                #       % (i, TH, in_eql, weighted_err))
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_label_est = copy.copy(predic_vals)
                    best_stump['dim'] = i
                    best_stump['TH'] = TH
                    best_stump['in_eql'] = in_eql
    return best_stump, min_err, best_label_est


# 逻辑回归分类如何跟AdaBoost结合起来?
def adaBoostTrain(data_arr, labels, iter_num=40):
    '''
    AdaBoost训练过程
    '''
    weak_cls = []  # 弱分类器数组
    m = data_arr.shape[0]
    D = np.mat(np.ones((m, 1))) / m  # 初始化数据权重向量
    agg_label_est = np.mat(np.zeros((m, 1)))

    # 迭代级联
    for i in range(iter_num):
        print('\n--iter {}\nD:{}'.format(i + 1, D.T))
        best_stump, err, label_est = build_stump(data_arr, labels, D)

        # 计算本分类器权重alpha： 防止除以一个极小的数而导致上溢
        alpha = float(0.5 * log((1.0 - err) / max(err, 1e-16)))
        best_stump['alpha'] = alpha # 存储本分类器的权重向量
        weak_cls.append(best_stump)
        print('label_est:\n', label_est.T)

        # 统计哪些分类正确,哪些分类错误,由分类器错误率更新数据权重: alpha -> D
        expon = np.multiply(-1.0 * alpha * np.mat(labels).T, label_est)
        D = np.multiply(D, np.exp(expon)) # 更新样本权重
        D = D / D.sum() # 样本权重归一化

        # 更新预测结果
        agg_label_est += alpha * label_est
        print('agg_label_est:\n', agg_label_est.T)

        # 重新统计误差
        agg_errs = np.multiply(np.sign(agg_label_est) !=
                               np.mat(labels).T, np.ones((m, 1)))
        err_rate = agg_errs.sum() / float(m)
        print('--training error rate: {}%'.format(err_rate*100.0))
        if err_rate == 0.0: # 误差达到要求跳出迭代
            break
    return weak_cls


def adaClassify(data, weak_cls):
    '''
    利用训练好的AdaBoost模型进行分类
    '''
    data = np.mat(data)
    agg_est = np.mat(np.zeros((data.shape[0], 1)))
    for classifier in weak_cls:
        labels = stump_classify(data, classifier['dim'], classifier['TH'], classifier['in_eql'])
        agg_est += classifier['alpha'] * labels
        # print('agg_labels:\n', agg_est)
    return np.sign(agg_est)


def testHourseData(data_path, test_path):
    '''
    测试马的数据集,并与逻辑回归比较
    '''
    data_arr, label_arr = load_dataset(data_path)

    # 训练模型
    cls_arr = adaBoostTrain(data_arr, label_arr, 150)

    # 测试数据集
    test_arr, test_label_arr = load_dataset(test_path)
    predictions = adaClassify(test_arr, cls_arr)
    err_arr = np.mat(np.ones((67, 1)))
    ret = err_arr[predictions != np.mat(test_label_arr).T].sum()
    print('--test error rate: {}%.'.format(ret / 67.0 * 100.0))


if __name__ == '__main__':
    print('DIR_PATH: ', DIR_PATH)
    # data_mat, labels = load_data_sim()

    # D = np.ones((len(labels), 1)) / 5
    # best_stump, min_err, best_label_est = build_stump(data_mat, labels, D)
    # print('best_stump:\n', best_stump)
    # print('min_err: ', min_err)
    # print('best_label_est:\n', np.array(best_label_est))

    # weak_cls_arr = adaBoostTrain(data_mat, labels, 9)
    # print('\nweak clasifiers:\n', weak_cls_arr)
    # label = adaClassify([3, 3], weak_cls_arr)
    # print('predict {} to be {}.'.format([3, 3], label))

    # x = list(data_mat[:, 0])
    # y = list(data_mat[:, 1])
    # plt.scatter(x, y)
    # plt.show()

    #---------测试马数据集
    data_path = os.path.join(DIR_PATH, 'horseColicTraining2.txt')
    test_path = os.path.join(DIR_PATH, 'horseColicTest2.txt')
    testHourseData(data_path, test_path)

    print('-- Test done.\n')

# matplotlib交互式导航
# http://blog.csdn.net/mytestmy/article/details/18983889
