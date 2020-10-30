# coding: utf-8

import numpy as np
from numpy import matrix as mat

import cv2
import os
import math


def undistort(img,                   # image data
              fx, fy, cx, cy,        # camera intrinsics
              k1, k2,                # radial distortion parameters
              p1=None, p2=None,      # tagential distortion parameters
              radial_ud_only=True):
    """
    undistort image using distort model
    test gray-scale image only
    """
    if img is None:
        print('[Err]: empty image.')
        return

    is_bgr = len(img.shape) == 3
    if is_bgr:
        H, W, C = img.shape
    elif len(img.shape) == 2:
        H, W = img.shape
    else:
        print('[Err]: image format wrong!')
        return

    img_undistort = np.zeros_like(img, dtype=np.uint8)

    # fill in each pixel in un-distorted image
    for v in range(H):
        for u in range(W):  # u,v are pixel coordinates
            # convert to camera coordinates by camera intrinsic parameters
            x1 = (u - cx) / fx
            y1 = (v - cy) / fy

            r_square = (x1 * x1) + (y1 * y1)
            r_quadric = r_square * r_square

            if radial_ud_only:  # do radial undistortion only
                x2 = x1 * (1.0 + k1 * r_square + k2 * r_quadric)
                y2 = y1 * (1.0 + k1 * r_square + k2 * r_quadric)
            else:  # do radial undistortion and tangential undistortion
                x2 = x1 * (1.0 + k1 * r_square + k2 * r_quadric) + \
                    2.0 * p1 * x1 * y1 + p2 * (r_square + 2.0 * x1 * x1)
                y2 = y1 * (1.0 + k1 * r_square + k2 * r_quadric) + \
                    p1 * (r_square + 2.0 * y1 * y1) + 2.0 * p2 * x1 * y1

            # convert back to pixel coordinates
            # using nearest neighbor interpolation
            u_corrected = int(fx * x2 + cx + 0.5)
            v_corrected = int(fy * y2 + cy + 0.5)

            # @Todo: using bilinear interpolation...

            # processing pixel outside the image area
            if u_corrected < 0 or u_corrected >= W \
                    or v_corrected < 0 or v_corrected >= H:
                if is_bgr:
                    img_undistort[v, u, :] = 0
                else:
                    img_undistort[v, u] = 0
            else:
                if is_bgr:
                    img_undistort[v, u, :] = img[v_corrected,
                                                 u_corrected, :]  # y, x
                else:
                    img_undistort[v, u] = img[v_corrected, u_corrected]  # y, x

    return img_undistort.astype('uint8')


def test_undistort_img():
    img_path = './distorted.png'

    fx = 458.654
    fy = 457.296
    cx = 367.215
    cy = 248.375
    camera_intrinsics = [fx, fy, cx, cy]

    k1 = -0.28340811
    k2 = 0.07395907
    p1 = 0.00019359
    p2 = 1.76187114e-05

    # Init parameters to be optimized
    params = np.array([[-0.1],
                       [0.1]])  # k1k2

    # ---------- Run LM optimization
    LM_Optimize(params)
    k1 = params[0][0]
    k2 = params[1][0]
    # ----------

    undistort_img(img_path, camera_intrinsics, k1, k2, p1, p2)


def undistort_img(img_path,
                  camera_intrinsics,
                  k1, k2, p1=None, p2=None,
                  is_color=True):
    """
    undistort of image
    given camera matrix and distortion coefficients
    """
    # LM_Optimize()

    fx = camera_intrinsics[0]
    fy = camera_intrinsics[1]
    cx = camera_intrinsics[2]
    cy = camera_intrinsics[3]

    if not os.path.isfile(img_path):
        print('[Err]: invalid image path.')
        return

    img_orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if is_color:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print('[Err]: empty image.')
        return

    # ---------- Do undistortion
    img_undistort = undistort(img,
                              fx, fy, cx, cy,
                              k1, k2, p1, p2)
    # ----------

    cv2.imshow('origin', img_orig)
    cv2.imshow('undistort', img_undistort)

    cv2.waitKey()


def show_points_of_curve():
    """
    visualize points on the curve
    """
    pts_on_curve = [
        [546, 20], [545, 40], [543, 83],
        [536, 159], [535, 170], [534, 180],
        [531, 200], [530, 211], [529, 218],
        [526, 236], [524, 253], [521, 269],
        [519, 281], [517, 293], [515, 302],
        [514, 310], [512, 320], [510, 329],
        [508, 341], [506, 353], [505, 357]
    ]
    print('Total {:d} points on the curve.'.format(len(pts_on_curve)))

    img_path = './distorted.png'
    if not os.path.isfile(img_path):
        print('[Err]: invalid image path.')
        return

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('[Err]: empty image.')
        return

    # Draw points and centroid
    centroid_x, centroid_y = 0.0, 0.0
    for pt in pts_on_curve:
        centroid_x += pt[0]
        centroid_y += pt[1]

        cv2.circle(img, tuple(pt), 5, (0, 255, 0), -1)

    centroid_x /= float(len(pts_on_curve))
    centroid_y /= float(len(pts_on_curve))
    centroid_x = int(centroid_x + 0.5)
    centroid_y = int(centroid_y + 0.5)

    cv2.circle(img, (centroid_x, centroid_y), 7, (0, 0, 255), -1)

    # Draw line of endpoints
    cv2.line(img, tuple(pts_on_curve[0]), tuple(
        pts_on_curve[-1]), (255, 0, 0), 2)

    cv2.imshow('Curve', img)
    cv2.waitKey()


def line_equation(first_x, first_y, second_x, second_y):
    # Ax+By+C=0
    A = second_y - first_y
    B = first_x - second_x
    C = second_x*first_y - first_x*second_y

    # k = -1.0 * A / B
    # b = -1.0 * C / B

    return A, B, C


def dist_of_pt_to_line(pt, A, B, C):
    """
    2D space point to line distance
    """
    # tmp = abs(A*pt[0] + B*pt[1] + C) / math.sqrt(A*A + B*B)
    tmp = -(A*pt[0] + B*pt[1] + C) / math.sqrt(A*A + B*B)

    return tmp
    # return math.sqrt(tmp * tmp)


def undistort_point(u, v,
                    fx, fy, cx, cy,
                    k1, k2, p1=None, p2=None,
                    radial_ud_only=True):
    """
    """
    # convert to camera coordinates by camera intrinsic parameters
    x1 = (u - cx) / fx
    y1 = (v - cy) / fy

    # compute r^2 and r^4
    r_square = (x1 * x1) + (y1 * y1)
    r_quadric = r_square * r_square

    if radial_ud_only:  # do radial undistortion only
        x2 = x1 * (1.0 + k1 * r_square + k2 * r_quadric)
        y2 = y1 * (1.0 + k1 * r_square + k2 * r_quadric)
    else:  # do radial undistortion and tangential undistortion
        x2 = x1 * (1.0 + k1 * r_square + k2 * r_quadric) + \
            2.0 * p1 * x1 * y1 + p2 * (r_square + 2.0 * x1 * x1)
        y2 = y1 * (1.0 + k1 * r_square + k2 * r_quadric) + \
            p1 * (r_square + 2.0 * y1 * y1) + 2.0 * p2 * x1 * y

    # convert back to pixel coordinates
    # using nearest neighbor interpolation
    u_corrected = fx * x2 + cx
    v_corrected = fy * y2 + cy

    return [u_corrected, v_corrected]

# the function


def test_undistort_pts_on_curve():
    """
    """
    fx = 458.654
    fy = 457.296
    cx = 367.215
    cy = 248.375
    k1 = -0.28340811
    k2 = 0.07395907

    k1k2 = np.array([[k1],
                     [k2]])

    pts_orig = [
        [546, 20],  [545, 40],  [543, 83],
        [536, 159], [535, 170], [534, 180],
        [531, 200], [530, 211], [529, 218],
        [526, 236], [524, 253], [521, 269],
        [519, 281], [517, 293], [515, 302],
        [514, 310], [512, 320], [510, 329],
        [508, 341], [506, 353], [505, 357]
    ]

    pts_corrected = undistort_point(
        pts_orig[:, 0], pts_orig[:, 1],
        fx, fy, cx, cy,
        k1k2[0][0], k1k2[1][0]
    )

    img_path = './distorted.png'
    img_orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


def Func(fx, fy, cx, cy, k1k2, input_list):
    ret = np.zeros(len(input_list))

    for i, input_i in enumerate(input_list):
        # using numpy array for SIMD
        pts_orig = np.array(input_i)  #

        # applying undistortion of points
        pts_corrected = undistort_point(
            pts_orig[:, 0], pts_orig[:, 1],
            fx, fy, cx, cy,
            k1k2[0][0], k1k2[1][0]
        )

        # compute centroid of undistorted points
        centroid = np.sum(pts_corrected, axis=1)  # get sum by column
        centroid /= float(pts_orig.shape[0])

        # build line of undistorted endpoints
        A, B, C = line_equation(pts_corrected[0][0], pts_corrected[0][1],
                                pts_corrected[-1][0], pts_corrected[-1][1])

        # build loss function and return
        dist = dist_of_pt_to_line(centroid, A, B, C)

        ret[i] = dist

    ret = np.array(ret)
    ret = np.reshape(ret, (-1, 1))
    return ret


def Deriv(fx, fy, cx, cy,
          k1k2,
          input_list,
          i):
    """
    """

    k1k2_delta_1 = k1k2.copy()
    k1k2_delta_2 = k1k2.copy()

    k1k2_delta_1[i, 0] -= 0.000001
    k1k2_delta_2[i, 0] += 0.000001

    p1 = Func(fx, fy, cx, cy, k1k2_delta_1, input_list)
    p2 = Func(fx, fy, cx, cy, k1k2_delta_2, input_list)

    d = (p2 - p1) * 1.0 / (0.000002)

    return d


def test_func():
    pts_orig = [
        [546, 20],  [545, 40],  [543, 83],
        [536, 159], [535, 170], [534, 180],
        [531, 200], [530, 211], [529, 218],
        [526, 236], [524, 253], [521, 269],
        [519, 281], [517, 293], [515, 302],
        [514, 310], [512, 320], [510, 329],
        [508, 341], [506, 353], [505, 357]
    ]
    input_list = []
    input_list.append(pts_orig)

    fx = 458.654
    fy = 457.296
    cx = 367.215
    cy = 248.375

    # k1k2 = np.array([[0.1],
    #                  [0.1]])

    k1 = -0.28340811
    k2 = 0.07395907
    k1k2 = np.array([[k1],
                     [k2]])

    dists = Func(fx, fy, cx, cy, k1k2, input_list)  # N×1
    print('Dist: {:.3f}'.format(dists[0][0]))


def LM_Optimize(params, max_iter=100):
    """
    """
    # Known parameters(camera intrinsics)
    fx = 458.654
    fy = 457.296
    cx = 367.215
    cy = 248.375

    # Input
    pts_orig = [
        [546, 20],  [545, 40],  [543, 83],
        [536, 159], [535, 170], [534, 180],
        [531, 200], [530, 211], [529, 218],
        [526, 236], [524, 253], [521, 269],
        [519, 281], [517, 293], [515, 302],
        [514, 310], [512, 320], [510, 329],
        [508, 341], [506, 353], [505, 357]
    ]
    input_list = []
    input_list.append(pts_orig)

    N = len(input_list)  # 数据个数
    print('Total {:d} data.'.format(N))

    u, v = 1, 2
    step = 0
    last_mse = 0.0
    while max_iter:
        step += 1

        mse, mse_tmp = 0.0, 0.0

        # loss
        loss = Func(fx, fy, cx, cy, params, input_list)
        mse += sum(loss**2)
        mse /= N  # normalize

        # build Jacobin matrix
        J = mat(np.zeros((N, 2)))  # 雅克比矩阵
        for i in range(2):
            J[:, i] = Deriv(fx, fy, cx, cy, params, input_list, i)
        print('Jacobin matrix:\n', J)

        H = J.T*J + u*np.eye(2)   # 2×2
        params_delta = -H.I * J.T*fx        #

        # update parameters
        params_tmp = params.copy()
        params_tmp += params_delta

        # current loss
        loss_tmp = Func(fx, fy, cx, cy, params_tmp, input_list)
        mse_tmp = sum(loss_tmp[:, 0]**2)
        mse_tmp /= N

        # adaptive adjustment
        q = float((mse - mse_tmp) /
                  ((0.5*params_delta.T*(u*params_delta - J.T*loss))[0, 0]))
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

        print("step = %d, abs(mse-lase_mse) = %.8f" %
              (step, abs(mse-last_mse)))
        if abs(mse - last_mse) < 0.000001:
            break

        last_mse = mse  # 记录上一个 mse 的位置
        max_iter -= 1

    print('\nFinal optimized parameters:\n', params)


if __name__ == '__main__':
    test_undistort_img()
    # show_points_of_curve()
    # test_func()

    print('=> Test done.')
