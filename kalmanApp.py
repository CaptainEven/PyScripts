# encoding=utf-8

import cv2
import numpy as np


def center(points):
    """计算矩阵的质心"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)


class Pedestrian:
    """行人类
    每个行人都有一个专属于他的卡尔曼滤波器，用于跟踪他的行踪
    """

    def __init__(self, id, frame, track_window):
        # 构建感兴趣区域ROI
        self.id = int(id)
        x, y, w, h = track_window
        self.track_window = track_window

        # 转换成HSV颜色空间，可以更好地专注于颜色
        self.roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)

        # 图像彩色直方图 16列直方图，每列直方图以0为左边界，18为右边界
        roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        # 直方图归一化到0~255范围内
        self.roi_hist = cv2.normalize(
            roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # 构建卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03
        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_EPS |
                          cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.update(frame)

    # 更新 行人行踪 的方法
    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject(
            [hsv], [0], self.roi_hist, [0, 180], 1)

        # 使用CamShift跟踪行人的行踪
        ret, self.track_window = cv2.CamShift(
            back_project, self.track_window, self.term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        self.center = center(pts)
        cv2.polylines(frame, [pts], True, 255, 1)

        # 根据行人的的实际位置 矫正卡尔曼滤波器
        self.kalman.correct(self.center)
        prediction = self.kalman.predict()
        cv2.circle(frame, (int(prediction[0]), int(
            prediction[1])), 4, (255, 0, 0), -1)


# 读取视频
camera = cv2.VideoCapture("768x576.avi")
history = 20

# 用BackgroundSubtractorKNN构建背景模型
bs = cv2.createBackgroundSubtractorKNN()

cv2.namedWindow("surveillance")
pedestrians = {}
firstFrame = True
frames = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    grabbed, frame = camera.read()
    fgmask = bs.apply(frame)

    # 前20帧都没有被处理，只是被传递到BackgroundSubtractorKNN分割器
    if frames < history:
        frames += 1
        continue
    
    # 阈值化
    th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]

    # 通过对前景掩模进行膨胀和腐蚀处理，相当于进行闭运算
    # 开运算：先腐蚀再膨胀
    # 闭运算：先膨胀再腐蚀
    th = cv2.erode(th, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)

    # 轮廓提取
    image, contours, hier = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    counter = 0
    for c in contours:
        # 对每一个轮廓，如果面积大于阈值500
        if cv2.contourArea(c) > 500:
            # 绘制外包矩形
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # 仅仅对第一帧中出现的行人实例化
            if firstFrame is True:
                pedestrians[counter] = Pedestrian(counter, frame, (x, y, w, h))
            counter += 1

    for i, p in pedestrians.items():
        p.update(frame)  # 更新行人行踪

    firstFrame = False
    frames += 1
    cv2.imshow("surveillance", frame)
    out.write(frame)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
    
out.release()
camera.release()
