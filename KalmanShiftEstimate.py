# _*_coding: utf-8_*_
# By Even

'''
通过python改写匀加速小车位置估计的Kalman滤波模拟
(1). 系统的状态变量总共2个: 位移S和速度v, 且假设这两个系统状态变量不相关
(2). 系统建模——状态转移方程:
            [St, Vt]' = [[1, Δt],  *  [S(t-1), V(t-1)]' + [1/2*(Δt)^2, Δt]' * a(加速度)
                         [0, 1]]
     系统建模——观测方程:
            [St', Vt'] = [1, 0] * [St, Vt] + [ΔSt, ΔVt](误差或噪声)
            测量方式为直接获得位置坐标而无需公式转换，所以观测矩阵H中位移为1,速度为0
'''

import numpy as np
import matplotlib.pyplot as plt


delta_t = 0.1                               # 采样间隔
t = np.linspace(0, 11, 111, endpoint=True)  # 时间序列
# print('t:', t)
N = len(t)
# print(N)
size = (N, 2)
print(size)
u = 10   # 控制向量                          # 重力加速度, 模拟自由落体运动位置估计
shift_real = 0.5 * u * t * t                # 真实位移值
velosity_real = u * t                       # 真是速率值
# print('x: ', x)
shift_noise = np.random.normal(0, 15.0, size=N)          # 位移高斯白噪声
# size_1 = int(round(0.375*N))
# noise_1 = np.random.normal(0, 9.0, size=size_1)
# noise_2 = np.random.normal(0, 11.0, size=N-size_1)
# noise = np.append(noise_1, noise_2) # numpy数组拼接
shift_measure = shift_real + shift_noise                 # 加入高斯白噪声的位移测量值


# 过程噪声的协方差矩阵:协方差(对角线)为0,位移s的过程噪声方差为0
Q = np.array([[0, 0],
              [0, 1.2]])
# print('Q: ', Q)

# 系统测量噪声(状态转移噪声)为常量
# 测量噪声R增大,动态响应变慢,收敛稳定性变好
R = np.array([[5.0, 0],    # R该如何设置初值?
              [0, 15.0]])  # R太小不会收敛


# x系统建模：系统状态转移矩阵A(2*2)和B(2*1)
A = np.array([[1, delta_t], [0, 1]])
# print('A: ', A)
B = np.array([0.5 * delta_t * delta_t, delta_t])
# print('B:', B)
# print('\n')

# 测量值当然是由系统状态变量映射出来的
# 系统状态变量到测量值的转换矩阵:z = h*x+v(测量噪声)
H = np.array([1, 0])          # 系统状态变量中,只测量位移,不测量速度


# 系统初始化
n = Q.shape
# print('n:', n)
m = R.shape
# print('m:', m)

x_hat = np.zeros(size)           # x的后验估计值
# print('x_hat:', x_hat)
# P = np.zeros(n)              # 后验估计误差协方差矩阵(每次迭代最后需更新)
P = np.array([[2, 0],
               [0, 2]])
# print('P: ', P)
x_hat_minus = np.zeros(size)     # x的先验估计值(上一次估计值, 每次迭代开始需更新)
P_minus = np.zeros(n)          # 先验估计误差协方差矩阵(每次迭代开始需更新)
# print('P_minus init:', P_minus)
K = np.zeros((n[0], m[0]))     # Kalman增益矩阵(2*1)
print('K init:\n', K)
I = np.eye(n[0], n[1])         # 单位矩阵
# print('I:', I)

# Kalman迭代过程
for i in range(1, N):
    # t-1 到 t时刻的状态预测，得到前验概率
    x_hat_minus[i] = A.dot(x_hat[i - 1]) + B * u                                # (1).状态转移方程
    P_minus = A.dot(P).dot(A.T) + Q                                             # (2).误差转移方程

    # 根据观察量对预测状态进行修正，得到后验概率，也就是最优值
    K = P_minus.dot(H.T).dot(np.linalg.inv(H.dot(P_minus).dot(H.T) + R))          # (3).Kalman增益
    print('\n--Round %d K:\n' %i, K)

    x_hat[i] = x_hat_minus[i] + K.dot(shift_measure[i] - H.dot(x_hat_minus[i])) # (4).状态修正方程
    # P = (I - K.dot(H)).dot(P_minus)  
    P = P_minus - K.dot(H)*(P_minus)                                        # (5).误差修正方程

    print('--Round %d P:\n' %i, P)

# 取位移和速度
shift_estimate    = [s for (s, v) in x_hat]
velocity = [v for (s, v) in x_hat]

# 迭代过程显示
# 显示kalman滤波
plt.figure()
plt.plot(shift_measure,          'r+', label='measured shift')    # 测量位移值
plt.plot(shift_estimate,         'b-', label='estimated shift')   # 估计位移值
plt.plot(shift_real,             'y-', label='real shift')        # 系统真实值
# plt.plot(velocity,      'c-', label='estimated velocity')       # 估计速率值
# plt.plot(velosity_real, 'g-', label='real velocity')            # 真实速率值
plt.legend()
plt.title('Kalman filter')
plt.xlabel('Iteration')
plt.ylabel('Shift & Velocity')
plt.tight_layout()
plt.show()

# 扩展卡尔曼滤波就是应用在非线性状态转移的环境中的。
# 它的算法和标准算法一样，只是把上面的A、B和H矩阵
# 线性化，也就是用一个线性方程最大程度的逼近非线性方程

# https://blog.csdn.net/u010720661/article/details/63253509