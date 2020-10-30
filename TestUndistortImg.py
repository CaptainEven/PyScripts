# coding: utf-8

import numpy as np
import cv2
import os


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
                    img_undistort[v, u, :] = img[v_corrected, u_corrected, :]  # y, x
                else:
                    img_undistort[v, u] = img[v_corrected, u_corrected]  # y, x

    return img_undistort.astype('uint8')


def test_img_undistortion(is_color=True):
    k1 = -0.28340811
    k2 = 0.07395907
    p1 = 0.00019359
    p2 = 1.76187114e-05
    fx = 458.654
    fy = 457.296
    cx = 367.215
    cy = 248.375

    img_path = './distorted.png'
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


if __name__ == '__main__':
    test_img_undistortion(is_color=True)
    print('=> Test done.')


# https://blog.csdn.net/weixin_39752599/article/details/82389555
