# encoding=utf-8

import os

import cv2
import numpy as np
import pcl
from Vis3DPointCloud import view_points_cloud


def points2pcd(points, PCD_FILE_PATH):
    """
    """

    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = points.shape[0]

    # pcd头部(重要)
    handle.write('# .PCD v0.7 - Point Cloud Data file format\n'
                 'VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + \
                 str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)

    handle.close()


def points2ply(points, colors, ply_f_path):
    """
    """
    # 读取三维点坐标和颜色信息
    points = np.hstack([points.reshape(-1, 3), colors.reshape(-1, 3)])
    # 必须先写入, 然后利用write()在头部插入ply header
    np.savetxt(ply_f_path, points, fmt='%f %f %f %d %d %d')

    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    \n
    '''

    with open(ply_f_path, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(points)))
        f.write(old)


def test():
    """
    KITTI视差图——>深度图——>点云
    """

    def disp2depth(b, f, disp):
        """
        """
        disp = disp.astype(np.float32)
        non_zero_inds = np.where(disp)

        depth = np.zeros_like(disp, dtype=np.float32)
        depth[non_zero_inds] = b * f / disp[non_zero_inds]

        return depth

    disp_f_path = './disp.png'  # TestDisparity2DepthAndPC
    img_f_path = './img.png'
    if not (os.path.isfile(disp_f_path) or os.path.isfile(img_f_path)):
        print('[Err]: invalid disparity/image file path.')
        return

    # KITTI数据集参数
    f = 721  # pixel
    b = 0.54  # m

    # 读取视差图
    disp = cv2.imread(disp_f_path, cv2.IMREAD_ANYDEPTH)
    print('Disparity image data type: ', disp.dtype)

    # 读取BGR图
    bgr = cv2.imread(img_f_path, cv2.IMREAD_COLOR)
    print('BGR image data type: ', bgr.dtype)

    assert (bgr.shape[:2] == disp.shape[:2])

    H, W = disp.shape[:2]
    print('H, W: {:d}, {:d}'.format(H, W))
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    # print(c, '\n', r)
    # x, y = np.arange(W), np.arange(H)
    cx, cy = W * 0.5, H * 0.5

    # 视差图(uint16)——>深度图(float32)
    depth = disp2depth(b, f, disp)

    # 深度图——>点云x, y, z
    points = np.zeros((H, W, 3), dtype=np.float32)
    colors = np.zeros((H, W, 3), dtype=np.uint8)
    points[r, c, 0] = (c - cx) * depth / f  # x
    points[r, c, 1] = (r - cy) * depth / f  # y
    points[r, c, 2] = depth  # z

    # bgr ——> rgb
    colors = bgr[:, :, ::-1]

    # ----- 过滤掉x, y, z全为0的点
    inds = np.where((points[:, :, 0] != 0.0) |
                    (points[:, :, 1] != 0.0) |
                    (points[:, :, 2] != 0.0))
    points = points[inds]
    colors = colors[inds]

    # # --- convert
    # points[:, 0] = (points[:, 0] + 0.1) * 300.0
    # points[:, 1] = points[:, 1] * 300.0
    # points[:, 2] = points[:, 2] * 300.0 - 11.5

    # Reshaping points 3D
    # points = np.reshape(points, (-1, 3))

    # 保存pcd点云文件
    points2pcd(points, './pc.pcd')
    print('PCD poind cloud saved.')

    # 保存ply点云文件
    points2ply(points, colors, './ply.ply')
    print('Ply poind cloud saved.')

    # ---------- 保存深度图
    depth *= 1000.0  # m ——> mm
    depth = depth.astype(np.uint16)
    cv2.imwrite('./depth.png', depth)
    print('Depth image written.')


def test_xiaomi():
    """
    KITTI视差图——>深度图——>点云
    """

    def disp2depth(b, f, disp):
        """
        """
        disp = disp.astype(np.float32)
        non_zero_inds = np.where(disp)

        depth = np.zeros_like(disp, dtype=np.float32)
        depth[non_zero_inds] = b * f / disp[non_zero_inds]

        return depth

    disp_f_path = './disp_10.png'  # TestDisparity2DepthAndPC
    img_f_path  = './left_10.png'
    if not (os.path.isfile(disp_f_path) or os.path.isfile(img_f_path)):
        print('[Err]: invalid disparity/image file path.')
        return

    # KITTI数据集参数
    f = 721  # pixel
    b = 0.54  # m

    # # xiaomi参数
    # # fx = 998.72290039062500
    # # fy = 1000.0239868164063
    # f = (998.72290039062500 + 1000.0239868164063) * 0.5  # 1000.0
    # cx = 671.15643310546875
    # cy = 384.32458496093750
    # b = 0.12  # m

    # 读取视差图
    disp = cv2.imread(disp_f_path, cv2.IMREAD_ANYDEPTH)
    print('Disparity image data type: ', disp.dtype)

    # 读取BGR图
    bgr = cv2.imread(img_f_path, cv2.IMREAD_COLOR)
    print('BGR image data type: ', bgr.dtype)

    assert (bgr.shape[:2] == disp.shape[:2])

    H, W = disp.shape[:2]
    print('W×H: {:d}×{:d}'.format(W, H))
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    # print(c, '\n', r)
    # x, y = np.arange(W), np.arange(H)
    cx, cy = W * 0.5, H * 0.5

    # ---------- 视差图(uint16)——>深度图(float32)
    depth = disp2depth(b, f, disp)

    # --------- 深度图——>点云x, y, z
    points = np.zeros((H, W, 3), dtype=np.float32)
    colors = np.zeros((H, W, 3), dtype=np.uint8)
    points[r, c, 0] = (c - cx) * depth / f  # x
    points[r, c, 1] = (r - cy) * depth / f  # y
    points[r, c, 2] = depth  # z

    # bgr ——> rgb
    colors = bgr[:, :, ::-1]

    # ----- 过滤掉x, y, z全为0的点
    inds = np.where((points[:, :, 0] != 0.0) |
                    (points[:, :, 1] != 0.0) |
                    (points[:, :, 2] != 0.0))
    points = points[inds]
    colors = colors[inds]

    # # ----- 滤波
    inds = np.where(
        (points[:, 1] > -1.0)
        & (points[:, 1] < 1.0)
    )
    points = points[inds]
    colors = colors[inds]
    print('{:d} 3D points left.'.format(inds[0].size))

    # # --- apply transformations
    # points[:, 0] += 5.0
    # # points[:, 1] *= -1.0  # Y轴颠倒
    # points[:, 2] -= 8.0
    # points = np.reshape(points, (-1, 3))

    view_points_cloud(points)

    # 保存pcd点云文件
    points2pcd(points, './pc_10.pcd')
    print('PCD poind cloud saved.')

    # 保存ply点云文件
    points2ply(points, colors, './ply_10.ply')
    print('Ply poind cloud saved.')

    # ---------- 保存深度图
    depth *= 1000.0  # m ——> mm
    depth = depth.astype(np.uint16)
    cv2.imwrite('./depth_10.png', depth)
    print('Depth image written.')


if __name__ == '__main__':
    test_xiaomi()
