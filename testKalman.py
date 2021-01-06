# -*- coding=utf-8 -*-
# Kalman filter example demo in Python
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# by Even
# _*_coding:utf-8_*_

import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from statsmodels.robust import stand_mad

'''
Kalman滤波本身是贝叶斯滤波体系的，建立在马尔科夫概率模型基础上
'''

# 原函数
def func(x):
    return 0.9**x


# 拟合函数
def fitFunc(x, a):
    return a**x


# 这里是假设A=1，H=1的情况
# intial parameters
n_iter = 500
sz = (n_iter,)  # size of array
x = 0.0

# observations (normal about x, sigma=0.1)
# noise = np.random.normal(0, 0.05, size=sz)
size_1 = int(round(0.375*n_iter))
noise_1 = np.random.normal(0, 0.04, size=size_1)
noise_2 = np.random.normal(0, 0.05, size=n_iter-size_1)
noise = np.append(noise_1, noise_2) # numpy数组拼接
# print('z:\n', z)


'''
Kalman滤波包括: 初始化、预测、更新三个步骤
过程误差噪声Q和测量误差噪声R不是都应该是高斯白噪声吗？
Q, R应该是系统状态向量噪声变量的协方差矩阵
 Q:过程噪声，Q增大，动态响应变快，收敛稳定性变坏
 R:测量噪声，R增大，动态响应变慢，收敛稳定性变好
'''
Q = 0.0007 # process variance(过程噪声即预测噪声或者叫做系统噪声)

# allocate space for arrays
xhat = np.zeros(sz)       # a posterior estimate of x
P = np.zeros(sz)          # a posterior error estimate: estimate bias
xhatminus = np.zeros(sz)  # a priori estimate of x
Pminus = np.zeros(sz)     # a priori error estimate
Kg = np.zeros(sz)         # gain or blending factor

# calculate measurements
X = np.linspace(1, n_iter, n_iter).tolist()
orig_z = np.array(list(map(func, X)))       # true value 
z = orig_z + noise                          # add noise

# estimate of measurement variance(测量误差噪声)
R = 0.08

# intial guesses
xhat[0] = 0.8     # intial value of predict
P[0] = 1.0        # initial value of predict(estimated) error

# 对于单变量(一个状态)P是估计误差，对于多变量，P是估计误差协方差矩阵
# 多系统状态变量，P是估计值和真实值间误差的协方差矩阵
# 预测误差的协方差矩阵P的更新公式是如何推导出来的？(通过最小二乘法优化残差推导出来的)

# Pminus是预测值(上一次的估计值)与真实值之间误差的协方差矩阵

# predict and pdate: the measurements are already done here(z).
for k in range(1, n_iter):
    # time update
    # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k): let A=1, BU(k)=0
    xhatminus[k] = xhat[k - 1]
    Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k): let A=1

    # compute Kalman's gain
    # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R]: let H=1
    Kg[k] = Pminus[k] / (Pminus[k] + R)

    # measurement update
    # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)]: let H=1
    xhat[k] = xhatminus[k] + Kg[k] * (z[k] - xhatminus[k])  # predict:残差(z[k]-xhatminus[k])
    # print('-- round %d estimate: %.3f' % (k, xhat[k]), flush=True)
    
    P[k] = (1 - Kg[k]) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1  # update


# 从概率论贝叶斯模型的观点来看前面预测的结果就是先验，测量出的结果就是后验。
plt.figure()
plt.plot(z, 'r+', label='noisy measurements')       # 测量值
plt.plot(xhat, 'b-', label='kalman predicts')       # 过滤后的值
plt.plot(orig_z, 'y-', label='truth value')         # 系统真实值
plt.legend()
plt.title('Kalman filter')
plt.xlabel('Iteration')
plt.ylabel('Intensity')

# plot Kg
# plt.figure()
# valid_iter = range(1, n_iter)
# plt.plot(valid_iter, Kg[valid_iter], label='Kg')
# plt.xlabel('Iterator')
# plt.ylabel('Kg')

# plot P
plt.figure()
valid_iter = range(1, n_iter)  # Pminus not valid at step 0
plt.plot(valid_iter, P[valid_iter], label='a priori error estimate')
plt.xlabel('Iteration')
plt.ylabel('$(Intensity)^2$')
plt.setp(plt.gca(), 'ylim', [0, .01])

'''
用小波滤波对比
'''
# wavelet decomposition
noisy_coefs = pywt.wavedec(z, 'db8', level=3, mode='per')

# contruct threshold
sigma = 0.2  # stand_mad(noisy_coefs[-1]) # ?
uthresh = sigma * np.sqrt(2.0 * np.log(len(z)))

# compute denoised coefficients by threshold
denoised_coefs = noisy_coefs[:]
denoised_coefs[1:] = (pywt._thresholding.soft(data, value=uthresh) for data in denoised_coefs[1:])
rec_signal = pywt.waverec(denoised_coefs, 'db8', mode='per')
plt.figure()
plt.plot(z, 'r+', label='noisy measurements')       # 测量值
plt.plot(rec_signal, 'b-', label='wavelet filter')  # 过滤后的值
plt.plot(orig_z, 'y-', label='truth value')         # 系统真实值
plt.legend()
plt.title('Wavelet filter')
plt.xlabel('Iteration')
plt.ylabel('Intensity')


'''
用函数曲线拟合滤波来对比
'''
popt, pcov = curve_fit(fitFunc, X, z)
print('Fitted function" y=%.3f^x:' %popt[0])
y_fit = [fitFunc(x_val, popt[0]) for x_val in X]
plt.figure()
plt.plot(z, 'r+', label='noisy measurements')       # 测量值
plt.plot(y_fit, 'b-', label='curve_fit filter')     # 过滤后的值
plt.plot(orig_z, 'y-', label='truth value')         # 系统真实值
plt.legend()
plt.title('Curve fitting')
plt.xlabel('Iteration')
plt.ylabel('Intensity')

plt.show()

'''
为什么噪声必须服从高斯分布，在进行参数估计的时候，估计的一种标准叫最大似然估计，
它的核心思想就是你手里的这些相互间独立的样本既然出现了，那就说明这些样本概率的
乘积应该最大(概率大才出现嘛)。如果样本服从概率高斯分布，对他们的概率乘积取对
数ln后，你会发现函数形式将会变成(一个常数加上样本最小均方误差)的形式。
'''
# ref: http://blog.csdn.net/lwplwf/article/details/74295801
# http://www.cnblogs.com/xmphoenix/p/3634536.html
