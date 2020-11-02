# encoding=utf-8
import os
import cv2
import math
import numpy as np
from numpy import matrix as mat


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

    # pts_on_curve = [
    #     [546, 20], [545, 40], [543, 83],
    #     [536, 159], [535, 170], [534, 180],
    #     [531, 200], [530, 211], [529, 218],
    #     [526, 236], [524, 253], [521, 269],
    #     [519, 281], [517, 293], [515, 302],
    #     [514, 310], [512, 320], [510, 329],
    #     [508, 341], [506, 353], [505, 357]
    # ]
    # print('Total {:d} points on the curve.'.format(len(pts_on_curve)))
    # pts_corrected = []

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


def undistort_img(img_path,
                  camera_intrinsics,
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
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print('[Err]: empty image.')
        return

    # -----
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
    pts_on_curve = np.array(pts_on_curve)

    # 计算dsit和offset
    A, B, C, k, b = line_equation(pts_on_curve[0][0], pts_on_curve[0][1],
                                  pts_on_curve[-1][0], pts_on_curve[-1][1])

    dist, offset = 0.0, 0.0
    for pt in pts_on_curve:
        x, y = pt
        dist += abs(A*x + B*y + C) / math.sqrt(A*A + B*B)
        y_est = k*x + b
        y_delta = abs(y - y_est)
        # print(y_delta)
        offset += y_delta
    dist /= float(len(pts_on_curve))
    offset /= float(len(pts_on_curve))

    print('Mean dist: {:.3f}'.format(dist))
    print('Offset: {:.3f}'.format(offset))

    # Draw points and centroid
    for pt in pts_on_curve:
        cv2.circle(img_orig, tuple(pt), 5, (0, 255, 0), -1)

    # Draw line of endpoints
    cv2.line(img_orig,
             tuple(pts_on_curve[0]),
             tuple(pts_on_curve[-1]),
             (255, 0, 0),
             2)
    # -----

    # ---------- Do undistortion
    # undistort image
    img_undistort = undistort(img,
                              fx, fy, cx, cy,
                              k1, k2, p1, p2)

    # undistort points
    k1k2 = np.array([[k1],
                     [k2]])
    pts_corrected = undistort_points(
        pts_on_curve[:, 0], pts_on_curve[:, 1],
        fx, fy, cx, cy,
        k1k2[0][0], k1k2[1][0]
    )
    Xs = pts_corrected[0]
    Ys = pts_corrected[1]

    A, B, C, k, b = line_equation(Xs[0], Ys[0], Xs[-1], Ys[-1])

    # Xs = [x[0] for x in pts_correctted]
    # Ys = [x[1] for x in pts_correctted]

    dist, offset = 0.0, 0.0
    for x, y in zip(Xs, Ys):
        cv2.circle(img_undistort, (int(x+0.5), int(y+0.5)), 5, (0, 255, 0), -1)
        dist += abs(A*x + B*y + C) / math.sqrt(A*A + B*B)
        y_est = k*x + b
        y_delta = abs(y - y_est)
        # print(y_delta)
        offset += y_delta
    dist /= float(Xs.shape[0])
    offset /= float(Xs.shape[0])

    print('Mean dist after optimization: {:.3f}'.format(dist))
    print('Offset after optimization: {:.3f}'.format(offset))

    # Draw line of endpoints
    cv2.line(img_undistort,
             (int(Xs[0]+0.5), int(Ys[0]+0.5)),
             (int(Xs[-1]), int(Ys[-1])),
             (255, 0, 0),
             2)
    # ----------

    cv2.imshow('origin', img_orig)
    cv2.imshow('undistort', img_undistort)

    cv2.waitKey()


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
    params = LM(params)
    k1 = params[0][0]
    k2 = params[1][0]
    # ----------

    # ---------- Undistort
    undistort_img(img_path, camera_intrinsics, k1, k2, p1, p2)


def Func(params, fx, fy, cx, cy, pts_orig):
    """
    """
    pts_orig = np.array(pts_orig)

    pts_corrected = undistort_points(
        pts_orig[:, 0], pts_orig[:, 1],
        fx, fy, cx, cy,
        params[0][0], params[1][0]
    )

    X = pts_corrected[0]  # x coordinates
    Y = pts_corrected[1]  # y coordinates
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))

    # compute k,b
    A, B, C, k, b = line_equation(X[0], Y[0], X[-1], Y[-1])
    Y_est = k*X + b

    return Y_est, Y


def Deriv(params, fx, fy, cx, cy, pts_orig, i):
    """
    """
    pts_orig = np.array(pts_orig)

    params_delta_1 = params.copy()
    params_delta_2 = params.copy()

    params_delta_1[i, 0] -= 0.000001
    params_delta_2[i, 0] += 0.000001

    # compute Y1 
    y_est_1, _ = Func(params_delta_1, fx, fy, cx, cy, pts_orig)

    # compute y2
    y_est_2, _ = Func(params_delta_2, fx, fy, cx, cy, pts_orig)

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


def LM(params, max_iter=100):
    """
    @params: numpy array
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

    N = len(pts_orig)
    print('Total {:d} points on the line.'.format(N))

    u, v = 1, 2
    step = 0
    last_mse = 0.0
    while max_iter:
        mse, mse_tmp = 0.0, 0.0
        step += 1
        
        Y_est, Y = Func(params, fx, fy, cx, cy, pts_orig)
        r = Y - Y_est
        mse += sum(r**2)
        mse /= N  # normalize

        # build Jacobin matrix
        J = mat(np.zeros((N, params.shape[0])))
        for i in range(params.shape[0]):  # 
            J[:, i] = Deriv(params, fx, fy, cx, cy, pts_orig, i)

        H = J.T*J + u*np.eye(2)   # 2*2
        hlm = H.I * J.T * r

        # update parameters
        params_tmp = params.copy()  # deep copy
        params_tmp += hlm

        y_est, Y = Func(params_tmp, fx, fy, cx, cy, pts_orig)
        r_tmp = Y - Y_est
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

        print("step = %d,abs(mse-lase_mse) = %.8f" %
              (step, abs(mse - last_mse)))
        print('parameters:\n', params)

        if abs(mse - last_mse) < 0.000001:
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

    k1 = -0.1
    k2 = 0.1

    k1k2 = np.array([[k1],
                     [k2]])

    LM(k1k2, max_iter=100)




if __name__ == "__main__":
    test_undistort_img()
    # TestLM()
