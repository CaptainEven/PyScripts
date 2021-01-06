# _*_coding: utf-8_*_
# By Even

'''
通过python改写匀加速小车位置估计的Kalman滤波模拟
(1). 系统的状态变量总共2个: 位移S和速度v,且假设这两个系统状态变量不相关
(2). 系统建模——状态转移方程:
            [St, Vt]' = [[1, Δt], *  [S(t-1), V(t-1)]' + [1/2*(Δt)^2, Δt]' * a(加速度)
                         [0, 1]]
     系统建模——观测方程:
            [St', Vt'] = [[1, 0], * [St, Vt]' + [ΔSt, ΔVt](误差或噪声)
                           0, 1] 
            测量方式为直接获得位置坐标而无需公式转换，所以观测矩阵H中位移为1,速度为1
'''

import numpy as np
import matplotlib.pyplot as plt


delta_t = 0.1                               # 采样间隔
t = np.linspace(0, 10, 101, endpoint=True)  # 时间序列
# print('t:', t)
N = len(t)
# print(N)
sz = (N, 2)
print(sz)
g = 10                                      # 重力加速度,模拟自由落体运动位置估计
shift_real = 0.5 * g * t * t                # 真实位移值
velosity_real = g * t                       # 真是速率值
# print('x: ', x)
shift_noise = np.random.normal(0, 13.5, size=N)          # 位移高斯白噪声
velocity_noise = np.random.normal(0, 9.9, size=N)        # 速率高斯白噪声
# size_1 = int(round(0.375*N))
# noise_1 = np.random.normal(0, 9.0, size=size_1)
# noise_2 = np.random.normal(0, 11.0, size=N-size_1)
# noise = np.append(noise_1, noise_2) # numpy数组拼接
S_meas = shift_real + shift_noise              # 加入高斯白噪声的位移测量值
V_meas = velosity_real + velocity_noise        # 加入高斯暴躁生的速率测量值

# 过程噪声的协方差矩阵:协方差(对角线)为0,位移s的过程噪声方差为0
Q = np.array([[0.1, 0],
              [0, 1.2]])
# print('Q: ', Q)

# 系统测量噪声(状态转移噪声)为常量
# 测量噪声R增大,动态响应变慢,收敛稳定性变好
R = np.array([[3.0, 0],    # R该如何设置初值?
              [0, 7.0]])  # R太小不会收敛


# x系统建模：系统状态转移矩阵A(2*2)和B(2*1)
A = np.array([[1, delta_t], [0, 1]])
# print('A: ', A)
B = np.array([0.5 * delta_t * delta_t, delta_t])
# print('B:', B)
# print('\n')

# 测量值当然是由系统状态变量映射出来的
# 系统状态变量到测量值的转换矩阵:z = h*x+v(测量噪声)
H = np.array([[1, 0],
              [0, 1]])          # 系统状态变量中,既测量位移,也测量速度，此时相当于单位矩阵


# 系统初始化
n = Q.shape
# print('n:', n)
m = R.shape
# print('m:', m)

x_hat = np.zeros(sz)           # x的后验估计值
# print('x_hat:', x_hat)
# P = np.zeros(n)              # 后验估计误差协方差矩阵(每次迭代最后需更新)
P = np.array([[2, 0],
              [0, 2]])
# print('P: ', P)
x_hat_minus = np.zeros(sz)     # x的先验估计值(上一次估计值, 每次迭代开始需更新)
P_minus = np.zeros(n)          # 先验估计误差协方差矩阵(每次迭代开始需更新)
# print('P_minus init:', P_minus)
K = np.zeros((n[0], m[0]))     # Kalman增益矩阵(2*2)
# print('K:', K)
I = np.eye(n[0], n[1])         # 单位矩阵
# print('I:', I)

# Kalman迭代过程
for i in range(9, N):
    # t-1 到 t时刻的状态预测，得到前验概率
    x_hat_minus[i] = A.dot(x_hat[i - 1]) + B * g  # (1).状态转移方程
    P_minus = A.dot(P).dot(A.T) + Q               # (2).误差转移方程

    # 根据观察量对预测状态进行修正，得到后验概率，也就是最优值
    K = P_minus.dot(H.T).dot(np.linalg.inv(H.dot(P_minus).dot(H.T) + R))          # (3).Kalman增益
    print('\n--Round %d K:\n' % i, K)
    x_hat[i] = x_hat_minus[i] + \
        K.dot(np.array([S_meas[i], V_meas[i]]) -
              H.dot(x_hat_minus[i]))  # (4).状态修正方程
    # (5).误差修正方程
    # P = (I - K.dot(H)).dot(P_minus)
    print(K.dot(H))
    P = P_minus - K.dot(H).dot(P_minus)
    print('--Round %d P:\n' % i, P)

# 取位移和速度的估计值
S_estimate = [s for (s, v) in x_hat]
V_estimate = [v for (s, v) in x_hat]

# Kalman迭代过程
plt.figure()
plt.plot(S_meas, 'r+', label='measured shift')      # 测量位移值
plt.plot(V_meas, 'm+', label='measured velocity')   # 测量速率值
plt.plot(S_estimate, 'b-', label='estimated shift')    # 估计位移值
plt.plot(V_estimate, 'b-', label='estimated velocity')  # 估计速率值
plt.plot(shift_real, 'y-', label='real shift')         # 真实位移值
plt.plot(velosity_real, 'g-', label='real velocity')   # 真实速率值
plt.legend()
plt.title('Kalman filter')
plt.xlabel('Iteration')
plt.ylabel('Shift & Velocity')
plt.tight_layout()
plt.show()

# 下一步考虑track cross坐标定位的Kalman建模：从单个坐标到81个坐标
# 下一步考虑Intensity建模
# 下一步考虑扩展卡尔曼滤波算法(EKF)对非线性系统建模
