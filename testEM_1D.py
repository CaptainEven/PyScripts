#_*_coding: utf-8_*_

'''
1. 极大似然估计是一种反推：根据结论推导该结果出现可能性最大的条件。
2. 在数学中, 海森矩阵(Hessian matrix或Hessian)是一个自变量为向量的实值函数的二阶偏导数组成的方块矩阵
3. 假设任务是优化一个目标函数f, 求函数的极大极小问题, 可以转化为求解函数f的导数f′=0的问题, 
这样求可以把优化问题看成方程求解问题(f′=0) 
4. 一个二维随机向量不仅要考虑
'''
# coding:gbk
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

isdebug = True

def logger(func):
    def inner(*args, **kwargs):
        print("-- %s's arguments are %s, %s" % (func.__str__(), args, kwargs))
        return func(*args, **kwargs)
    return inner


# 指定k个高斯分布參数。这里指定k=2。注意2个高斯分布具有同样均方差Sigma，分别为Mu1, Mu2。
@logger
def ini_data(Sigma, Mu1, Mu2, k, N):
    global X
    global Mu
    global Expectations
    X = np.zeros(N)
    Mu = np.random.random(2)
    print('Init Mu: ', Mu)
    Expectations = np.zeros((N, k))
    print('-- initial expectations:', Expectations)
    for i in range(N):
        if np.random.random(1) > 0.5:
            X[i] = np.random.normal() * Sigma + Mu1
        else:
            X[i] = np.random.normal() * Sigma + Mu2
    if isdebug:
        print("***********")
        print(u"初始观測数据X：")
        print(X)


# EM算法：步骤1，计算E[zij]
@logger
def e_step(Sigma, k, N):
    global Expectations
    global Mu
    global X
    for i in range(N):
        Denom = 0
        for j in range(k):
            Denom += math.exp((-1 / (2 * (float(Sigma**2))))
                              * (float(X[i] - Mu[j]))**2)
        for j in range(k):
            Numer = math.exp((-1 / (2 * (float(Sigma**2))))
                             * (float(X[i] - Mu[j]))**2)
            Expectations[i, j] = Numer / Denom
    if isdebug:
        print(u"隐藏变量E（Z）：")
        print(Expectations)


# EM算法：步骤2，求最大化E[zij]的參数Mu
@logger
def m_step(k, N):
    global Expectations
    global X
    for j in range(k):
        Numer = 0
        Denom = 0
        for i in range(N):
            Numer += Expectations[i, j] * X[i]
            Denom += Expectations[i, j]
        Mu[j] = Numer / Denom
    print('Mu: ', Mu)


# 算法迭代iter_num次。或达到精度Epsilon停止迭代
@logger
def run(Sigma, Mu1, Mu2, k, N, iter_num, Epsilon):
    ini_data(Sigma, Mu1, Mu2, k, N)
    print(u"初始<u1,u2>:", Mu)
    print(u"初始<sigma1,igma2>:", Sigma, Sigma)
    for i in range(iter_num):
        print('-- Round %d' % i)
        Old_Mu = copy.deepcopy(Mu)
        e_step(Sigma, k, N)
        m_step(k, N)
        print(i, Mu, '\n')
        if sum(abs(Mu - Old_Mu)) < Epsilon:
            break


if __name__ == '__main__':
    run(5.8, 40, 20, 2, 1000, 1000, 0.0001)
    plt.hist(X, 50)
    plt.show()

# http://jacoxu.com/jacobian%e7%9f%a9%e9%98%b5%e5%92%8chessian%e7%9f%a9%e9%98%b5/ (Hessian矩阵)
