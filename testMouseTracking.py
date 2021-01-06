# encoding=utf-8

import cv2
import numpy as np

# 创建一个空帧，定义(600, 600, 3)画图区域
frame = np.ones((600, 600, 3), np.uint8) * 255

# 初始化测量坐标和鼠标运动预测的数组
last_measurement = current_measurement = np.zeros((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)  # 坐标初始化？


# 定义鼠标回调函数，用来绘制跟踪结果
def mouseMove(event, x, y, s, p):
    """
    """
    global frame, current_measurement, \
        measurements, last_measurement, \
        current_prediction, last_prediction

    last_prediction = current_prediction                        # 把当前预测存储为上一次预测
    last_measurement = current_measurement                      # 把当前测量存储为上一次测量
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])  # 当前测量

    # predict phase
    current_prediction = kalman.predict()  

    # update(correct) phase: 用当前测量来校正卡尔曼滤波器
    kalman.correct(current_measurement)

    # 计算卡尔曼预测值，作为当前预测
    # update
    lmx, lmy = last_measurement[0], last_measurement[1]         # 上一次测量坐标
    cmx, cmy = current_measurement[0], current_measurement[1]   # 当前测量坐标
    lpx, lpy = last_prediction[0], last_prediction[1]           # 上一次预测坐标
    cpx, cpy = current_prediction[0], current_prediction[1]     # 当前预测坐标

    # 绘制从上一次测量到当前测量以及从上一次预测到当前预测的两条线
    cv2.line(frame, (int(lmx[0]), int(lmy[0])), (int(
        cmx[0]), int(cmy[0])), (255, 0, 0), 2)        # 蓝色线为测量值
    cv2.line(frame, (int(lpx[0]), int(lpy[0])), (int(
        cpx[0]), int(cpy[0])), (0, 0, 255), 2)        # 红色线为预测值


# 窗口初始化
WIN_NAME = 'KalmanTracker'
cv2.namedWindow(WIN_NAME)

# opencv采用setMouseCallback函数处理鼠标事件，
# 具体事件必须由回调（事件）函数的第一个参数来处理，该参数确定触发事件的类型（点击、移动等）
cv2.setMouseCallback(WIN_NAME, mouseMove)

'''
Kalman滤波包括: 初始化、预测、更新三个步骤
过程误差噪声Q和测量误差噪声R不是都应该是高斯白噪声吗？
Q, R应该是系统状态向量噪声变量的协方差矩阵
 Q:过程噪声，Q增大，动态响应变快，收敛稳定性变坏
 R:测量噪声，R增大，动态响应变慢，收敛稳定性变好
'''

# 4：状态数，包括(x, y, dx, dy)坐标及速度(每次移动的距离)；2: 观测量，能看到的是坐标值
# 系统初始化，初始化Kalman滤波器
kalman = cv2.KalmanFilter(4, 2)  # 4个系统状态变量(col)，2个观测变量(row)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)  # 测量矩阵(H): H也是状态变量到测量的转换矩阵

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)          # 状态转移矩阵(A)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5              # 过程噪声(协方差矩阵)Q
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.001        # 测量噪声(协方差矩阵)R

# 系统状态变量过程噪声的协方差矩阵的协方差部分(处主对角线外)为0说明：
# 系统的状态变量之间各不相关

# ord()函数与cha()r函数配对，以一个字符（长度为1的字符串）作为参数，返回对应的ASCII数值
while True:
    cv2.imshow(WIN_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()


# 卡尔曼滤波原理： http://blog.csdn.net/u010720661/article/details/63253509
# http://www.cnblogs.com/xmphoenix/p/3634536.html
# http://blog.csdn.net/mangzuo/article/details/71171137
# 这个例子理解了，卡尔曼滤波就差不多理解了
# http://blog.csdn.net/heyijia0327/article/details/17487467

# http://blog.csdn.net/u012700322/article/details/52857162

# https://blog.csdn.net/u010720661/article/details/63253509