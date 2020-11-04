# encoding=utf-8
import os
import cv2
import math
import numpy as np
from numpy import matrix as mat


def undistort_img(img,                   # image data
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
            # pt = [u, v]

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

            # if pt in pts_on_curve:
            #     pts_corrected.append([u_corrected, v_corrected])

    # return img_undistort.astype('uint8'), pts_corrected
    return img_undistort.astype('uint8')


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

    pts_corrected = undistort_points(
        pts_orig[:, 0], pts_orig[:, 1],
        fx, fy, cx, cy,
        k1k2[0][0], k1k2[1][0]
    )

    img_path = './distorted.png'
    img_orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


def undistort_img_and_statistics(img_path,
                                 camera_intrinsics,
                                 pts_list,
                                 k1, k2, p1=None, p2=None,
                                 is_color=True):
    """
    undistort of image
    given camera matrix and distortion coefficients
    """
    print('k1k2: {:.3f}, {:.3f}'.format(k1, k2))

    fx = camera_intrinsics[0]
    fy = camera_intrinsics[1]
    cx = camera_intrinsics[2]
    cy = camera_intrinsics[3]

    if not os.path.isfile(img_path):
        print('[Err]: invalid image path.')
        return

    img_orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if is_color:
        img_orig = img_orig[40:, :, :]
    else:
        img_orig = img_orig[40:, :]

    if is_color:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img[40:, :, :]
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img[40:, :]

    if img is None:
        print('[Err]: empty image.')
        return

    # ----- draw original points
    dist, offset = 0.0, 0.0
    pts_count = 0
    for pts in pts_list:
        pts = np.array(pts)
        pts_count += len(pts)

        # 计算dsit和offset
        A, B, C, k, b = line_equation(pts[0][0], pts[0][1],
                                      pts[-1][0], pts[-1][1])

        for pt in pts:
            x, y = pt
            dist += abs(A*x + B*y + C) / math.sqrt(A*A + B*B)
            y_est = k*x + b
            y_delta = abs(y - y_est)
            # print(y_delta)
            offset += y_delta

        # Draw points and centroid
        for pt in pts:
            cv2.circle(img_orig, tuple(pt), 5, (255, 0, 0), -1)

        # Draw line of endpoints
        cv2.line(img_orig, tuple(pts[0]), tuple(pts[-1]), (0, 0, 255), 2)
        # -----

    dist /= float(pts_count)
    offset /= float(pts_count)
    print('Mean dist: {:.3f}'.format(dist))
    print('Mean offset: {:.3f}'.format(offset))

    # ---------- Do undistortion
    # ----- undistort image
    img_undistort = undistort_img(img,
                                  fx, fy, cx, cy,
                                  k1, k2, p1, p2)

    # ----- undistort points
    k1k2 = np.array([[k1],
                     [k2]])

    dist, offset = 0.0, 0.0
    for pts in pts_list:
        pts = np.array(pts)

        pts_corrected = undistort_points(
            pts[:, 0], pts[:, 1],
            fx, fy, cx, cy,
            k1k2[0][0], k1k2[1][0]
        )
        Xs = pts_corrected[0]
        Ys = pts_corrected[1]

        A, B, C, k, b = line_equation(Xs[0], Ys[0], Xs[-1], Ys[-1])

        for x, y in zip(Xs, Ys):
            cv2.circle(img_undistort, (int(x+0.5), int(y+0.5)), 5, (255, 0, 0), -1)
            dist += abs(A*x + B*y + C) / math.sqrt(A*A + B*B)
            y_est = k*x + b
            y_delta = abs(y - y_est)
            # print(y_delta)
            offset += y_delta

        # Draw line of endpoints
        cv2.line(img_undistort,
                 (int(Xs[0]+0.5), int(Ys[0]+0.5)),
                 (int(Xs[-1]), int(Ys[-1])),
                 (0, 0, 255),
                 2)
    # ----------
    dist /= float(pts_count)
    offset /= float(pts_count)
    print('\nAfter LM optimization:')
    print('Mean dist: {:.3f}'.format(dist))
    print('Mean offset: {:.3f}'.format(offset))

    cv2.imshow('origin', img_orig)
    cv2.imshow('undistort', img_undistort)

    cv2.waitKey()


def TestUndistortOptimize():
    img_path = './DistortedImages/11_3014.jpg'

    fx = 3014.00
    fy = 3014.00
    cx = 1280.00
    cy = 720.000
    camera_intrinsics = [fx, fy, cx, cy]

    k1 = -0.28340811
    k2 = 0.07395907
    p1 = 0.00019359
    p2 = 1.76187114e-05

    # Init parameters to be optimized
    params = np.array([[0.1],
                       [0.1]])  # k1k2

    # Input
    pts_on_curve_1 = [
        [1144, 489], [1156, 495], [1169, 502],
        [1183, 510], [1185, 511], [1201, 519],
        [1206, 522], [1223, 530], [1225, 531],
        [1242, 540], [1255, 547], [1285, 562],
        [1312, 575], [1326, 582], [1334, 586],
        [1377, 607], [1419, 627], [1461, 646],
        [1508, 670], [1530, 681], [1534, 682],
        [1763, 802], [1779, 811], [1861, 855],
        [1873, 862], [1886, 869], [1917, 886],
        [1948, 902], [1977, 918], [2018, 940]
    ]
    pts_on_curve_1 = [[pt[0], pt[1]-40] for pt in pts_on_curve_1]

    # pts_on_curve_2 = [
    #     [653, 269], [663, 267], [672, 264],
    #     [679, 263], [685, 262], [694, 259],
    #     [698, 258], [708, 256], [717, 254],
    #     [727, 253], [730, 251], [742, 248],
    #     [746, 247], [760, 244], [766, 243],
    #     [775, 240], [783, 238], [789, 237]
    # ]
    # pts_on_curve_2 = [[pt[0], pt[1]-40] for pt in pts_on_curve_2]

    pts_list = [pts_on_curve_1]

    # ---------- Run LM optimization
    params = LM(params, pts_list, max_iter=100)
    k1 = params[0][0]
    k2 = params[1][0]
    # ----------

    # ---------- Undistort
    undistort_img_and_statistics(img_path, camera_intrinsics, pts_list, k1, k2)


def Func(params, fx, fy, cx, cy, pts_list):
    """
    """
    for i, pts in enumerate(pts_list):
        pts = np.array(pts)

        pts_undistort = undistort_points(
            pts[:, 0], pts[:, 1],
            fx, fy, cx, cy,
            params[0][0], params[1][0]
        )

        X = pts_undistort[0]  # x coordinates
        Y = pts_undistort[1]  # y coordinates
        X = np.reshape(X, (-1, 1))
        Y = np.reshape(Y, (-1, 1))

        # compute k,b
        A, B, C, k, b = line_equation(X[0], Y[0], X[-1], Y[-1])
        Y_est = k*X + b

        if i == 0:
            Y_est_all = Y_est.copy()
            Y_all = Y.copy()
        else:
            Y_est_all = np.concatenate((Y_est_all, Y_est), axis=0)
            Y_all = np.concatenate((Y_all, Y), axis=0)

    return Y_est_all, Y_all


def Deriv(params, fx, fy, cx, cy, pts_list, i):
    """
    """
    params_delta_1 = params.copy()
    params_delta_2 = params.copy()

    params_delta_1[i, 0] -= 0.000001
    params_delta_2[i, 0] += 0.000001

    # compute y_est_1
    y_est_1, _ = Func(params_delta_1, fx, fy, cx, cy, pts_list)

    # compute y_est_2
    y_est_2, _ = Func(params_delta_2, fx, fy, cx, cy, pts_list)

    deriv = (y_est_2 - y_est_1) * 1.0 / (0.000002)

    return deriv


def undistort_points(u, v,
                     fx, fy, cx, cy,
                     k1, k2, p1=None, p2=None,
                     radial_ud_only=True):
    """
    """
    # convert to camera coordinates
    # by camera intrinsic parameters
    x1 = (u - cx) / fx
    y1 = (v - cy) / fy

    # compute r^2 and r^4
    r_square = (x1 * x1) + (y1 * y1)
    r_quadric = r_square * r_square

    if radial_ud_only:  # do radial undistortion only
        x2 = x1 / (1.0 + k1 * r_square + k2 * r_quadric)
        y2 = y1 / (1.0 + k1 * r_square + k2 * r_quadric)
    else:  # do radial undistortion and tangential undistortion
        x2 = x1 / (1.0 + k1 * r_square + k2 * r_quadric) + \
            2.0 / p1 * x1 * y1 + p2 * (r_square + 2.0 * x1 * x1)
        y2 = y1 / (1.0 + k1 * r_square + k2 * r_quadric) + \
            p1 / (r_square + 2.0 * y1 * y1) + 2.0 * p2 * x1 * y

    # convert back to pixel coordinates
    # using nearest neighbor interpolation
    u_corrected = fx * x2 + cx
    v_corrected = fy * y2 + cy

    return [u_corrected, v_corrected]


def line_equation(x1, y1, x2, y2):
    """
    Ax+By+C=0
    """
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2

    k = -1.0 * A / B
    b = -1.0 * C / B

    return A, B, C, k, b
    # return k, b


def LM(params, pts_list, max_iter=100):
    """
    @params: numpy array
    """
    # Known parameters(camera intrinsics)
    fx = 458.654
    fy = 457.296
    cx = 367.215
    cy = 248.375

    # count points
    pts_num_list = [len(x) for x in pts_list]
    N = sum(pts_num_list)
    print('Total {:d} points on the line.'.format(N))

    u, v = 1, 2
    step = 0
    last_mse = 0.0
    while max_iter:
        mse, mse_tmp = 0.0, 0.0
        step += 1

        Y_est, Y = Func(params, fx, fy, cx, cy, pts_list)
        r = Y - Y_est
        mse += sum(r**2)
        mse /= N  # normalize

        # build Jacobin matrix
        J = mat(np.zeros((N, params.shape[0])))
        for i in range(params.shape[0]):  #
            J[:, i] = Deriv(params, fx, fy, cx, cy, pts_list, i)

        H = J.T*J + u*np.eye(2)  # 2*2
        hlm = H.I * J.T * r

        # update parameters
        params_tmp = params.copy()  # deep copy
        params_tmp += hlm

        y_est, Y = Func(params_tmp, fx, fy, cx, cy, pts_list)
        r_tmp = Y - Y_est
        mse_tmp = sum(r_tmp[:, 0]**2)
        mse_tmp /= N

        # adaptive adjustment
        q = float((mse - mse_tmp) /
                  ((0.5*hlm.T*(u*hlm - J.T*r))[0, 0]))
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

        print("step = %d,abs(mse-lase_mse) = %.8f" %
              (step, abs(mse - last_mse)))
        print('parameters:\n', params)

        if abs(mse - last_mse) < 0.001:  # 0.000001
            break

        # update mse and iter_idx
        last_mse = mse
        max_iter -= 1

    print('\nFinal optimized parameters:\n', params)
    return params


def TestLM():
    """
    """
    # k1 = -0.28340811
    # k2 = 0.07395907

    k1 = 0.1
    k2 = 0.1

    k1k2 = np.array([[k1],
                     [k2]])

    LM(k1k2, max_iter=100)


if __name__ == "__main__":
    TestUndistortOptimize()
    # TestLM()
