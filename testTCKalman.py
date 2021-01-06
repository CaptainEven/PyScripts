# _*_coding: utf-8_*_

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread, imsave
from rename import renameFiles


def loadPts(path):
    pts = []
    with open(path, 'r') as fh:
        for line in fh.readlines():
            data = line.strip().split()
            pts.append([float(data[0]), float(data[1])])
    return pts


'''
system transform matrix:
affine matrix = [[Sxcosα, sinα, dx],
                 [-sinα, Sycosα, dy]]
Q: 过程噪声，Q增大，动态响应变快，收敛稳定性变坏
R: 测量噪声，R增大，动态响应变慢，收敛稳定性变好
'''

def kalmanTC(img_path, tc_id):
    # detect tracks
    if not os.path.exists(img_path + './0.tif'):
        renameFiles(img_path)
        print('-- detect track cross...')
        import subprocess
        cmd = './TrackDetector.exe ' + img_path + ' ./settings.config'
        print('CMD: ', cmd)
        sub_p = subprocess.Popen(cmd)  # call only once
        sub_p.wait()  # wait the sub_process run to end

    # check
    first_name = img_path + './0_tc.txt'
    second_name = img_path + './1_tc.txt'
    if not os.path.exists(first_name) \
            or not os.path.exists(second_name):
        print('[Error]: please check image path.')

    # load pts
    pts_0 = np.array(loadPts(first_name), dtype=float)
    pts_1 = np.array(loadPts(second_name), dtype=float)
    f_list = os.listdir(img_path)
    x_measure = []
    for f in f_list:
        f_name = os.path.splitext(f)
        if f_name[1] == '.txt':
            pt = loadPts(img_path + './' + f)[tc_id]
            x_measure.append(pt)

    '''
    system modelling
    '''
    # calculate affine matrix
    affine = cv2.estimateRigidTransform(pts_1, pts_0, False)
    print(affine)

    # system process noise covariance matrix:non-diagonal elements all are 0
    Q = np.array([[9.2, 0],
                  [0, 9.2]])  # 过程噪声(协方差矩阵)

    # system measurement noise covariance matrix:non-diagonal elements all are 0
    # dynamic response decreases as R increases but convergence become easier
    R = np.array([[0.2, 0],
                  [0, 0.2]])  # 测量噪声(协方差矩阵)

    A = np.array([[affine[0][0], affine[0][1]],
                  [affine[1][0], affine[1][1]]])  # 状态转移矩阵(过程转移矩阵)
    print('A:\n', A)
    B = np.array([affine[0][2], affine[1][2]])  # 控制矩阵
    print('B:\n', B)

    # measure matrix:z = h*x+v(measured noise)
    H = np.array([[1, 0],   # 测量矩阵H
                  [0, 1]])  # measur both x and y: eqal to the eye matrix

    '''
    system initilization
    '''
    n = Q.shape
    m = R.shape
    sz = (len(x_measure), 2)
    x_hat = np.zeros(sz)           # x's posteriori estimate
    x_hat[0] = x_measure[0]        # init first value of x_hat
    P = np.zeros(n)                # posteriori estimate error covariance matrix
    x_hat_minus = np.zeros(sz)     # x's priori estimate(last time's estimate)
    P_minus = np.zeros(n)          # priori estimate error covariance matrix
    K = np.zeros((n[0], m[0]))     # Kalman gain(2*2)
    I = np.eye(n[0], n[1])         # eye matrix

    # Kalman's iterative process
    for i in range(1, len(x_measure)):
        # ---------- predict phase: (t-1) -> t
        # (1).state transition equation
        x_hat_minus[i] = A.dot(x_hat[i - 1]) + B  # 状态转移方程

        # (2).error transition equation                            
        P_minus = A.dot(P).dot(A.T) + Q           # 误差转移方程                                   

        # ---------- update phase: data fusion
        # correct and update
        # (3).Kalman's gain                       # 卡尔曼增益
        K = P_minus.dot(H.T).dot(np.linalg.inv(H.dot(P_minus).dot(H.T) + R))     
        print('\n--Round %d K:\n' % i, K)

        # (4).state correction equation           # 状态更新方程
        x_hat[i] = x_hat_minus[i] + K.dot(x_measure[i] - H.dot(x_hat_minus[i]))  

        # (5).error correction equation           # 误差更新方程
        # P = (I - K.dot(H)).dot(P_minus)  
        P = P_minus - K.dot(H).dot(P_minus)                                                
        print('--Round %d P:\n' % i, P)

    # get x and y 's estimate
    # print('-- estimated coordinates:\n', x_hat)
    x_estimate = [x for (x, y) in x_hat]
    y_estimate = [y for (x, y) in x_hat]

    # plot meaurements and estimates
    x_mea = [pt[0] for pt in x_measure]

    plt.figure()

    plt.plot(x_mea, 'k+',
             label='measured X')  # X measurement
    plt.plot([pt[1] for pt in x_measure], 'm+',
             label='measured Y')  # Y measurement
    plt.plot(x_estimate, 'y-', label='estimated X')
    plt.plot(y_estimate, 'g-', label='estimated Y')
    plt.legend()
    plt.title('Kalman filter of track cross %d' % tc_id)
    plt.xlabel('Iteration')
    plt.ylabel('X & Y')
    plt.tight_layout()

    # plot difference of estimate and measurements
    x_diff = [x[0] - x[1] for x in zip(x_estimate, x_mea)]
    y_mea = [pt[1] for pt in x_measure]
    y_diff = [y[0] - y[1] for y in zip(y_estimate, y_mea)]
    plt.figure()
    plt.plot(np.arange(len(x_measure)), x_diff, 'r-', label='x_difference')
    plt.plot(np.arange(len(x_measure)), y_diff, 'b-', label='y_difference')
    plt.show()


if __name__ == '__main__':
    print('-- Test start...')
    kalmanTC('d:/test_TC_kalman', 28)
    print('-- Test done.')
