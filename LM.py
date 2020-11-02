# encoding=utf-8

"""
Created on 2020-10-29
@author: Yiwen Liu
"""
import copy
import numpy as np
from numpy import matrix as mat
from matplotlib import pyplot as plt
import random
random.seed(0)

N = 100
a1, b1, c1 = 1, 2, 3      # 这个是需要拟合的函数y(x) 的真实参数
X = np.linspace(0, 1, N).reshape(100, 1)       

# 产生包含噪声的数据
Y = [np.exp(a1*i**2 + b1*i + c1) + random.gauss(0, 8) for i in X]

J = mat(np.zeros((N, 3)))  # 雅克比矩阵

r = mat(np.zeros((N, 1)))  # f(x)  100*1  误差
r_tmp = mat(np.zeros((N, 1)))
params = mat([[3.0], [2.0], [1.0]])  # 初始化待优化参数
print('\Initial parameters:\n', params)

# xk = mat([[12.0],[12.0],[12.0]]) # 参数初始化
last_mse = 0
step = 0
u, v = 1, 2
max_iter = 10000


def Func(abc, iput):   # 需要拟合的函数，abc是包含三个参数的一个矩阵[[a],[b],[c]]
    a = abc[0, 0]
    b = abc[1, 0]
    c = abc[2, 0]
    return np.exp(a*iput**2+b*iput+c)


# 对函数求偏导
def Deriv(abc, Xs, i):
    """
    数值逼近的方式求偏导
    """
    Xs = np.reshape(Xs, (-1, 1))
    
    abc_delta_1 = abc.copy()  # deepcopy in numpy
    abc_delta_2 = abc.copy()

    abc_delta_1[i, 0] -= 0.000001
    abc_delta_2[i, 0] += 0.000001

    y1 = Func(abc_delta_1, Xs)
    y2 = Func(abc_delta_2, Xs)

    d = (y2 - y1) * 1.0 / (0.000002)

    return d


while max_iter:
    mse, mse_tmp = 0.0, 0.0
    step += 1

    r = Y - Func(params, X)  # loss function

    mse += sum(r**2)
    mse /= N  # normalize

    # 构建雅各比矩阵
    for j in range(3):  # 3个变量
        J[:, j] = Deriv(params, X, j)  # 数值求导

    H = J.T*J + u*np.eye(3)   # 3*3
    hlm = H.I * J.T * r

    # update parameters
    params_tmp = params.copy()
    params_tmp += hlm

    # current loss
    r_tmp = Y - Func(params_tmp, X)

    mse_tmp = sum(r_tmp[:, 0]**2)
    mse_tmp /= N

    # adaptive adjustment
    q = float((mse - mse_tmp) /
              ((0.5*-hlm.T*(u*-hlm - J.T*r))[0, 0]))
    if q > 0:
        s = 1.0 / 3.0
        v = 2
        mse = mse_tmp
        params = params_tmp
        temp = 1 - pow(2.0*q-1, 3)

        if s > temp:
            u = u*s
        else:
            u = u*temp
    else:
        u = u*v
        v = 2*v
        params = params_tmp

    print("step = %d,abs(mse-lase_mse) = %.8f" % (step, abs(mse - last_mse)))
    print('parameters:\n', params)

    if abs(mse - last_mse) < 0.000001:
        break

    last_mse = mse  # 记录上一个 mse 的位置
    max_iter -= 1

print('\nFinal optimized parameters:\n', params)

# 用拟合好的参数画图
z = [Func(params, i) for i in X]

plt.figure(0)
plt.scatter(X, Y, s=4)
plt.plot(X, z, 'r')
plt.show()