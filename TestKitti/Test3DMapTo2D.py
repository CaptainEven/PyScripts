# encoding=utf-8

import os
import cv2
import pcl
import numpy as np
from tqdm import tqdm


def Map3DTo2D(img_f_path, pc_bin_f_path='./000008_pc.bin'):
    """
    """
    # KITTI数据集相机参数
    fx = 721.54
    fy = 721.54
    cx = 609.56
    cy = 172.85
    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1]
    ]
    K = np.array(K, dtype=np.float32)
    print(K)

    # 读取BGR图
    bgr = cv2.imread(img_f_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print('[Err]: read bgr iamge failed.')
        return
    H, W = bgr.shape[:2]

    # 读取3D点云
    point_cloud = np.fromfile(str(pc_bin_f_path),
                              dtype=np.float32, 
                              count=-1).reshape([-1, 4])
    print(point_cloud.shape)

    # ---------- 3D ——> 2D
    xyz = point_cloud[:, :-1] 
    print(xyz.shape)

    for pt3d in tqdm(xyz):
        pt3d = np.reshape(pt3d, (3, 1))
        uv_homo = K.dot(pt3d)
        if uv_homo[2] < 1e-8:
            uv = uv_homo / (uv_homo[2] + 1e-8)
        else:
            uv = uv_homo[:2] / uv_homo[2]

        uv = np.squeeze(uv)
        if uv[0] < W and uv[0] >= 0 and uv[1] < H and uv[1] >= 0:
            print(uv)


if __name__ == '__main__':
    Map3DTo2D(img_f_path='./000008.png')
