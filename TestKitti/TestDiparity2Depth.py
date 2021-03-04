# encoding=utf-8

import os
import cv2
import pcl
import numpy as np


def points2pcd(points, PCD_FILE_PATH):
    """
    """

    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = points.shape[0]

    # pcd头部（重要）
    handle.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
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
    np.savetxt(ply_f_path, points, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header

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

    disp_f_path = './disp.png'
    img_f_path = './img.png'
    if not (os.path.isfile(disp_f_path) or os.path.isfile(img_f_path)):
        print('[Err]: invalid disparity/image file path.')
        return

    # KITTI数据集参数
    f = 721   # pixel
    b = 0.54  # m

    # 读取视差图
    disp = cv2.imread(disp_f_path, cv2.IMREAD_ANYDEPTH)
    print('Disparity image data type: ', disp.dtype)

    # 读取BGR图
    bgr = cv2.imread(img_f_path, cv2.IMREAD_COLOR)
    print('BGR image data type: ', bgr.dtype)

    assert (bgr.shape[:2] == disp.shape[:2])

    H, W = disp.shape[:2]
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    # print(c, '\n', r)
    # x, y = np.arange(W), np.arange(H)
    cx, cy = W*0.5, H*0.5

    # 视差图(uint16)——>深度图(float32)
    depth = disp2depth(b, f, disp)

    # 深度图——>点云x, y, z
    points = np.zeros((H, W, 3), dtype=np.float32)
    colors = np.zeros((H, W, 3), dtype=np.uint8)
    points[r, c, 0] = (c - cx) * depth / f  # x
    points[r, c, 1] = (r - cy) * depth / f  # y
    points[r, c, 2] = depth                 # z
    # points = np.reshape(points, (H*W, 3))

    colors = bgr[:, :, ::-1]

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


if __name__ == '__main__':
    test()
