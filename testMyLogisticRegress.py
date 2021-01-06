# _*_coding: utf-8_*_
import numpy as np

# ------------加载数据
# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')
# X, y = mnist['data'], mnist['target']

# the_5_ids = np.array([i for i in range(50000, 70000) if y[i] == 5][:300]) # 取300个正样本
# the_not_5_ids = np.array([i for i in range(0, 10000) if y[i] != 5][:300]) # 取300个负样本
# X_5 = X[the_5_ids] 
# X_not_5 = X[the_not_5_ids] 
# some_5_digit = X_5[1]
# some_not_5_digit = X_not_5[1]
# some_5_image = some_5_digit.reshape(28, 28)
# some_not_5_image = some_not_5_digit.reshape(28, 28)


# ----------------------------处理数据集合:将正样本和负样本混合
# ids = np.concatenate((the_5_ids, the_not_5_ids), axis=0)
# np.random.shuffle(ids) # 洗牌
# print('len(ids): ', len(ids))
# print(ids[:50])
# dataset = X[ids]

# labels = np.zeros(len(ids))
# for i, id in enumerate(ids):
#     labels[i] = 1 if id in the_5_ids else 0
# print(labels[:50])

# ------------序列化生成的数据
import os
# import pickle
# from sklearn.externals import joblib

DIR_PATH = os.path.join(os.getcwd() + os.path.sep + os.path.sep)
print('Dir path:', DIR_PATH)

data_path = os.path.join(DIR_PATH + 'mnist_5.txt')
# np.savetxt(data_path, dataset, fmt='%d')
# print('-- Dump dataset done.\n')
labels_path = os.path.join(DIR_PATH, 'mnist_5_labels.txt')
# np.savetxt(labels_path, labels, fmt='%d')
# print('-- Dump labels done.\n')

# ------------加载序列化的数据和标签
dataset = np.loadtxt(data_path, dtype=np.uint8)  # 用numpy做数据持久化
labels = np.loadtxt(labels_path, dtype=np.int)
# print(labels[:50])

import matplotlib
import matplotlib.pyplot as plt
# plt.imshow(dataset[0].reshape(28, 28), cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.show()

# ------------先用自己实现的逻辑回归分类器(梯度下降优化)
w_path = os.path.join(DIR_PATH, 'w.txt')

# 数据28×28共784维, 只能使用随机梯度下降
from testLogisticRegression import *
clf = LogistRegessClassifier()
# w, ws = clf.grad_descent(dataset, labels, max_iter=100)
w, ws = clf.grad_descent_sgd(dataset, labels, max_iter=120)
print('\n-- Trained done and the final w:\n', w)

# ------------持久化权重向量
np.savetxt(w_path, w)
print('-- Dump weight vector done.')


# ------------加载训练好的权重向量, 用测试数据集测试准确率
weight = np.loadtxt(w_path, dtype=float)
weight = np.matrix(weight).reshape(-1, 1) # 转换成列向量
print('-- Trained weight loaded.')

# 准备测试数据集：取100个测试样本
# test_5_ids = np.array([i for i in range(50000, 70000) if y[i] == 5][300:400])
# test_not_5_ids = np.array([i for i in range(0, 10000) if y[i] != 5][:100])

# test_ids = np.concatenate((test_5_ids, test_not_5_ids), axis=0)
# np.random.shuffle(test_ids) # 洗牌

# test_set = X[test_ids]
# labels_ref = np.zeros(len(test_ids))
# for i, id in enumerate(test_ids):
#     labels_ref[i] = 1 if id in test_5_ids else 0

# 存储测试数据集
test_path = os.path.join(DIR_PATH + 'mnist_5_tst.txt')
# np.savetxt(test_path, dataset, fmt='%d')
# print('dump test dataset done.\n')
test_labels_path = os.path.join(DIR_PATH, 'mnist_5_labels_tst.txt')
# np.savetxt(test_labels_path, labels, fmt='%d')
# print('dump labels done.\n')

# 加载测试数据集和其对应的标签
test_set = np.loadtxt(test_path, dtype=np.uint8)  # 用numpy做数据持久化
labels_ref = np.loadtxt(test_labels_path, dtype=np.int)
print('-- Test dataset loaded.')

# 利用训练好的模型参数(权重向量)对测试数据集进行测试
labels_predict = np.zeros(len(labels_ref))
for i in range(len(labels_ref)):
    labels_predict[i] = clf.classify(test_set[i], weight)

# 统计正确率
count = 0
for i in range(len(labels_ref)):
    if labels_predict[i] == labels_ref[i]:
        count += 1
print('-- Accuracy: {}%'.format(float(count) / float(len(labels_ref)) * 100.0))
print('-- Test done.')

# ------------通过Adaptive boosting的方法
'''
AdaBoost是否对稀疏特征(特征中有很多0的情况)不适用?
'''
# from testAdaBoost import *
# weak_cls = adaBoostTrain(dataset, labels_ref, iter_num=3)
# print('weak classifiers:', weak_cls)

# https://github.com/PytLab/MLBox
