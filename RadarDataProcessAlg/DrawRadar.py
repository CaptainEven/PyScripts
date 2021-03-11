# encoding=utf-8

import time
import cv2 as cv
import numpy as np


# 生成一个900*900的空灰度图像
canvas_size = 900
canvas = np.zeros((canvas_size, canvas_size, 3), np.uint8)

point_size = 1
white = (255, 255, 255)
red = (0, 0, 255)
blue = (255, 0, 0)
yellow = (0, 255, 255)

# 雷达扫描线长度
scan_line_len = int((canvas_size - 100) * 0.5)

# 绘制雷达显示器界面的同心圆
canvas_center = int(canvas_size * 0.5)
num_circles = scan_line_len // 100
for angle_degrees in range(1, num_circles + 1):
    cv.circle(canvas, (canvas_center, canvas_center), angle_degrees * 100, white, 2)

# 绘制十字线
cv.line(canvas, (50, canvas_center), (canvas_size - 50, canvas_center), white, 2)
cv.line(canvas, (canvas_center, 50), (canvas_center, canvas_size - 50), white, 2)

start_point = (int(canvas_center - scan_line_len * np.sin(0.25 * np.pi)),
               int(canvas_center - scan_line_len * np.sin(0.25 * np.pi)))
end_point = (
int(canvas_center + scan_line_len * np.sin(0.25 * np.pi)), int(canvas_center + scan_line_len * np.sin(0.25 * np.pi)))
cv.line(canvas, start_point, end_point, white, 1)
start_point = (int(canvas_center - scan_line_len * np.sin(0.25 * np.pi)),
               int(canvas_center + scan_line_len * np.sin(0.25 * np.pi)))
end_point = (
int(canvas_center + scan_line_len * np.sin(0.25 * np.pi)), int(canvas_center - scan_line_len * np.sin(0.25 * np.pi)))
cv.line(canvas, start_point, end_point, white, 1)

# 绘制方位(正北指向)和距离刻度文字
padding = 30
txt_shift = 10
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(canvas, "N", (canvas_center - txt_shift, padding), font, 1, (255, 255, 255), 1)
cv.putText(canvas, "E", (canvas_size - padding, canvas_center + txt_shift), font, 1, (255, 255, 255), 1)
cv.putText(canvas, "W", (txt_shift, canvas_center + txt_shift), font, 1, (255, 255, 255), 1)
cv.putText(canvas, "S", (canvas_center - txt_shift, canvas_size - txt_shift), font, 1, (255, 255, 255), 1)


# # 添加参数指示文字
# cv.putText(canvas, "Speed(m/s):", (500, 15), font, 0.5, (255, 255, 255), 1)
# cv.putText(canvas, "Rotation:", (530, 35), font, 0.5, (255, 255, 255), 1)
# cv.putText(canvas, "Coordinate(X):", (488, 55), font, 0.5, (255, 255, 255), 1)
# cv.putText(canvas, "Coordinate(Y):", (488, 75), font, 0.5, (255, 255, 255), 1)

# 定义绘制扫描辉亮函数, ang为扫描线所在角度位置
def draw_scanner(ang_degree):
    """
    opencv基于椭圆绘制实心扇形
    :param ang_degree:
    :return:
    """
    img = np.zeros((canvas_size, canvas_size, 3), np.uint8)

    a = 255.0 / 60.0  # 将颜色值255等分60, 60为辉亮夹角
    for i in range(60):
        # 逐次绘制1度扇形，颜色从255到0
        cv.ellipse(img=img,
                   center = (canvas_center, canvas_center),
                   axes=(scan_line_len, scan_line_len),
                   angle=1,
                   startAngle=ang_degree - i,
                   endAngle=ang_degree - i - 1,
                   color=(255 - i * a, 255 - i * a, 255 - i * a),
                   thickness=-1)
    return img


# 运动目标初始值，beta飞行角度,speed速度
beta = 225 / 180 * np.pi  # 飞行方位角
speed = 300  # 飞行速度
point_x = 500
point_y = 10
pointEndX = 500
pintEndY = 500

angle_degrees = 0
delta_t = 1      # 目标运动的比例值
cycle_time = 10  # 雷达10s钟扫描一个周期

sleep_time = 1.0

total_start_time = time.time()
while (1):
    # angle_degrees += 10

    start_time = time.time()

    # ----------
    point_x += int(speed * delta_t * 0.01 * np.cos(beta))
    point_y += -int(speed * delta_t * 0.01 * np.sin(beta))

    cv.circle(canvas, (point_x, point_y), point_size, yellow, 1)  # 目标运动点迹

    # 复制雷达界面, 将目标运动和参数指示绘制在复制图上
    temp = np.copy(canvas)
    cv.circle(temp, (point_x, point_y), point_size, red, 6)  # 目标点
    cv.putText(temp, str(speed), (canvas_size - 100, padding), font, 0.5, (0, 255, 0), 1)
    cv.putText(temp, str(beta / np.pi * 180), (canvas_size - 100, padding + 20), font, 0.5, (0, 255, 0), 1)
    cv.putText(temp, str(point_x), (canvas_size - 100, padding + 40), font, 0.5, (0, 255, 0), 1)
    cv.putText(temp, str(point_y), (canvas_size - 100, padding + 60), font, 0.5, (0, 255, 0), 1)

    angle_degrees += 36.0

    # 绘制扫描辉亮
    scan_img = draw_scanner(angle_degrees)

    # 将雷达显示与扫描辉亮混合
    blend = cv.addWeighted(temp, 1.0, scan_img, 0.6, 0.0)
    cv.imshow('Radar Scanning', blend)
    # ----------

    end_time = time.time()
    pass_time = end_time - start_time
    total_pass_time = end_time - total_start_time
    if int(angle_degrees) % 360 == 0:
        print(pass_time)

    time.sleep(sleep_time - pass_time)

    # 按下ESC键退出
    if cv.waitKey(100) == 27:
        break
