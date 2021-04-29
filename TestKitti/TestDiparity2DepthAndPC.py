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

    # disp_f_path = './disp.png'  # TestDisparity2DepthAndPC
    # img_f_path = './img.png'
    disp_f_path = './0000000000.npy'  # TestDisparity2DepthAndPC
    img_f_path = './0000000000.png'
    if not (os.path.isfile(disp_f_path) or os.path.isfile(img_f_path)):
        print('[Err]: invalid disparity/image file path.')
        return

    # KITTI数据集参数
    f = 721  # pixel
    b = 0.54  # m

    # 读取视差图
    if disp_f_path.endswith('.png'):
        disp = cv2.imread(disp_f_path, cv2.IMREAD_ANYDEPTH)
    elif disp_f_path.endswith('.npy'):
        disp = np.load(disp_f_path)
    print('Disparity image data type: ', disp.dtype)

    # 读取BGR图
    bgr = cv2.imread(img_f_path, cv2.IMREAD_COLOR)
    print('BGR image data type: ', bgr.dtype)
    if bgr.shape[:2] != disp.shape[:2]:
        bgr = cv2.resize(bgr, (disp.shape[1], disp.shape[0]))

    assert (bgr.shape[:2] == disp.shape[:2])

    H, W = disp.shape[:2]
    print('H, W: {:d}, {:d}'.format(H, W))
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    # print(c, '\n', r)
    # x, y = np.arange(W), np.arange(H)
    cx, cy = W * 0.5, H * 0.5

    # 视差图(uint16)——>深度图(float32)
    depth = disp2depth(b, f, disp*256.0)

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


def test_depth_to_pointcloud():
    """
    """
    depth_f_path = './apollo_train_0_depth_metric.npy'
    image_f_path = './apollo_train_0.jpg'

    if not (os.path.isfile(depth_f_path) and os.path.isfile(image_f_path)):
        print('[Err]: invalid file path.')
        return

    bgr = cv2.imread(image_f_path, cv2.IMREAD_COLOR)
    depth = np.load(depth_f_path)
    print('Max depth: {:.3f}m.'.format(np.max(depth)))

    # # ---------- 保存深度图
    # # depth *= 256.0  # magnifying 256 times for better visua
    # depth = depth.astype(np.uint16)
    # depth_f_path = './apollo_train_0_depth_pred.png'
    # cv2.imwrite(depth_f_path, depth)
    # print('Depth image {:s} written.'.format(depth_f_path))

    ## KITTI数据集参数
    # f = 721   # pixel
    # b = 0.54  # m

    # ## xiaomi参数
    # # fx = 998.72290039062500
    # # fy = 1000.0239868164063
    # f = (998.72290039062500 + 1000.0239868164063) * 0.5  # 1000.0
    # cx = 671.15643310546875
    # cy = 384.32458496093750
    # b = 0.12  # m

    ## apollo stereo参数
    f = 2301.3147
    cx = 1489.8536
    cy = 479.1750
    b = 0.36  # m

    H, W = bgr.shape[:2]
    print('W×H: {:d}×{:d}'.format(W, H))
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    # print(c, '\n', r)
    # x, y = np.arange(W), np.arange(H)
    cx, cy = W * 0.5, H * 0.5
    
    ## ----- 过滤掉图像坐标系y轴方向
    Y_START = 250
    r = r[Y_START:]
    c = c[Y_START:]
    depth = depth[Y_START:]

    # --------- 深度图——>点云x, y, z
    points = np.zeros((H, W, 3), dtype=np.float32)
    colors = np.zeros((H, W, 3), dtype=np.uint8)
    points[r, c, 0] = (c - cx) * depth / f  # x
    points[r, c, 1] = (r - cy) * depth / f  # y
    points[r, c, 2] = depth                 # z

    # bgr ——> rgb
    colors = bgr[:, :, ::-1]

    # ----- 过滤掉x, y, z全为0的点
    inds = np.where((points[:, :, 0] != 0.0) |
                    (points[:, :, 1] != 0.0) |
                    (points[:, :, 2] != 0.0))
    points = points[inds]
    colors = colors[inds]

    # 保存ply点云文件
    print('Total {:d} 3d points remained.'.format(points.shape[0]))
    ply_f_path = './ply_apollo_train_0.ply'
    points2ply(points, colors, ply_f_path)
    print('{:s} saved.'.format(ply_f_path))


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

    disp_f_path = './disp_2.png'  # TestDisparity2DepthAndPC
    img_f_path  = './left_2.png'
    if not (os.path.isfile(disp_f_path) or os.path.isfile(img_f_path)):
        print('[Err]: invalid disparity/image file path.')
        return

    # # KITTI数据集参数
    # f = 721  # pixel
    # b = 0.54  # m

    # xiaomi参数
    # fx = 998.72290039062500
    # fy = 1000.0239868164063
    f = (998.72290039062500 + 1000.0239868164063) * 0.5  # 1000.0
    cx = 671.15643310546875
    cy = 384.32458496093750
    b = 0.12  # m

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

    # # # ----- 滤波
    # inds = np.where(
    #     (points[:, 1] > -1.0)
    #     & (points[:, 1] < 1.0)
    # )
    # points = points[inds]
    # colors = colors[inds]
    # print('{:d} 3D points left.'.format(inds[0].size))

    # view_points_cloud(points)

    # 保存pcd点云文件
    points2pcd(points, './pc_2.pcd')
    print('PCD poind cloud saved.')

    # 保存ply点云文件
    points2ply(points, colors, './ply_2.ply')
    print('Ply poind cloud saved.')

    # ---------- 保存深度图
    depth *= 1000.0  # m ——> mm
    depth = depth.astype(np.uint16)
    cv2.imwrite('./depth_2.png', depth)
    print('Depth image written.')


def test_kitti():
    """
    KITTI视差图——>深度图——>点云
    """
    ## 超参数: 用于点云截取
    MAX_DEPTH = 80.0
    MAX_HEIGHT = 2.0

     ## KITTI数据集参数
    b = 0.54  # m
    f = 718.335  # pixel
    cx = 609.5593  # pixel
    cy = 172.8540  # pixel

    def disp2depth(b, f, disp):
        """
        """
        disp = disp.astype(np.float32)
        non_zero_inds = np.where(disp)

        depth = np.zeros_like(disp, dtype=np.float32)
        min_disp = np.min(disp[non_zero_inds])
        max_disp = np.max(disp[non_zero_inds])
        print('Min disp: {:.3f}.'.format(min_disp))
        print('Max disp: {:.3f}.'.format(max_disp))

        depth[non_zero_inds] = b * f / (disp[non_zero_inds] + 1e-5)

        return depth

    disp_f_path = ''  # '0000000007_disp_pp.npy'  
    depth_f_path = './0000000200.npy'
    img_f_path   = './0000000200.jpg'
    if not (os.path.isfile(disp_f_path) or os.path.isfile(img_f_path)):
        print('[Err]: invalid disparity/image file path.')
        return

    if os.path.isfile(disp_f_path):
        is_disp = True
        print('Using disparity file.')
    elif os.path.isfile(depth_f_path):
        is_disp = False
        print('Using depth file.')
    elif os.path.isfile(disp_f_path) and os.path.isfile(depth_f_path):
        print('[Err]: both disparity and depth file exists.')
        return
    else:
        print('[Err]: both the disparity and depth image do not exist.')
        return

    if is_disp: # 读取视差图
        if disp_f_path.endswith('.png'):
            disp = cv2.imread(disp_f_path, cv2.IMREAD_ANYDEPTH)
        elif disp_f_path.endswith('.npy'):
            disp = np.load(disp_f_path)
        print('Disparity image data type: ', disp.dtype)

        # ---------- 视差图(uint16)——>深度图(float32)
        depth = disp2depth(b, f, disp)

    else:  # 读取深度图
        if depth_f_path.endswith('.png'):
            depth = cv2.imread(depth_f_path, cv2.IMREAD_ANYDEPTH)
        elif depth_f_path.endswith('.npy'):
            depth = np.load(depth_f_path)
        print('Depth image data type: ', depth.dtype)

    # 读取BGR图
    bgr = cv2.imread(img_f_path, cv2.IMREAD_COLOR)
    print('BGR image data type: ', bgr.dtype)

    if is_disp:  # 如堕读取的是视差图
        if bgr.shape[:2] != disp.shape[:2]:
            cv2.resize(bgr, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_CUBIC)
            print('BGR image is resized to {:d}×{:d}.'.format(disp.shape[1], disp.shape[0]))
    else:
        if bgr.shape[:2] != depth.shape[:2]:
            cv2.resize(bgr, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
            print('BGR image is resized to {:d}×{:d}.'.format(depth.shape[1], depth.shape[0]))

    if is_disp:
        H, W = disp.shape[:2]
    else:
        H, W = depth.shape[:2]
    print('W×H: {:d}×{:d}'.format(W, H))

    ## Build x, y pixel coordinates
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    if cx == 0.0 or cy == 0.0:
        cx, cy = W * 0.5, H * 0.5

    # ---------- 深度图滤波
    mask = depth > 0.0
    depth = depth * mask
    mask = depth < MAX_DEPTH
    depth = depth * mask
    print('Max depth: {:.3f}m.'.format(np.max(depth)))

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

    # ----- 点云滤波
    inds = np.where(
        (points[:, 1] < MAX_HEIGHT)
        & (points[:, 1] > -MAX_HEIGHT)
    )
    points = points[inds]
    colors = colors[inds]
    print('{:d} 3D points left.'.format(inds[0].size))

    # view_points_cloud(points)
    print('Total {:d} 3D points remained.'.format(points.shape[0]))

    # 保存pcd点云文件
    pcd_f_path = './pc.pcd'
    points2pcd(points, pcd_f_path)
    print('PCD poind cloud {:s} saved.'.format(pcd_f_path))

    # 保存ply点云文件
    ply_f_path = './ply.ply'
    points2ply(points, colors, ply_f_path)
    print('Ply poind cloud {:s} saved.'.format(ply_f_path))

    # ---------- 保存深度图
    # depth *= 256.0  # magnifying 256 times for better visua
    depth = depth.astype(np.uint16)
    depth_f_path = './depth.png'
    cv2.imwrite(depth_f_path, depth)
    print('Depth image {:s} written.'.format(depth_f_path))

    
def test_apollo():
    """
    Apollo视差图/深度图——>点云
    """
    ## 超参数: 用于点云截取
    MAX_DEPTH = 80.0
    MAX_HEIGHT = 2.0
    
    ## Apollo数据集参数
    f = 2301.3147
    cx = 1489.8536
    cy = 479.1750
    b = 0.36  # m

    def disp2depth(b, f, disp):
        """
        """
        disp = disp.astype(np.float32)
        non_zero_inds = np.where(disp)

        depth = np.zeros_like(disp, dtype=np.float32)
        min_disp = np.min(disp[non_zero_inds])
        max_disp = np.max(disp[non_zero_inds])
        print('Min disp: {:.3f}.'.format(min_disp))
        print('Max disp: {:.3f}.'.format(max_disp))

        depth[non_zero_inds] = b * f / (disp[non_zero_inds] + 1e-5)

        return depth

    disp_f_path = ''  # '0000000007_disp_pp.npy'  
    depth_f_path = './apollo_train_1_00000.npy'
    img_f_path   = './apollo_train_1_00000.jpg'
    if not (os.path.isfile(disp_f_path) or os.path.isfile(img_f_path)):
        print('[Err]: invalid disparity/image file path.')
        return

    if os.path.isfile(disp_f_path):
        is_disp = True
        print('Using disparity file.')
    elif os.path.isfile(depth_f_path):
        is_disp = False
        print('Using depth file.')
    elif os.path.isfile(disp_f_path) and os.path.isfile(depth_f_path):
        print('[Err]: both disparity and depth file exists.')
        return
    else:
        print('[Err]: both the disparity and depth image do not exist.')
        return

    if is_disp: # 读取视差图
        if disp_f_path.endswith('.png'):
            disp = cv2.imread(disp_f_path, cv2.IMREAD_ANYDEPTH)
        elif disp_f_path.endswith('.npy'):
            disp = np.load(disp_f_path)
        print('Disparity image data type: ', disp.dtype)

        # ---------- 视差图(uint16)——>深度图(float32)
        depth = disp2depth(b, f, disp)

    else:  # 读取深度图
        if depth_f_path.endswith('.png'):
            depth = cv2.imread(depth_f_path, cv2.IMREAD_ANYDEPTH)
        elif depth_f_path.endswith('.npy'):
            depth = np.load(depth_f_path)
        print('Depth image data type: ', depth.dtype)

    # 读取BGR图
    bgr = cv2.imread(img_f_path, cv2.IMREAD_COLOR)
    print('BGR image data type: ', bgr.dtype)

    if is_disp:  # 如堕读取的是视差图
        if bgr.shape[:2] != disp.shape[:2]:
            cv2.resize(bgr, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_CUBIC)
            print('BGR image is resized to {:d}×{:d}.'.format(disp.shape[1], disp.shape[0]))
    else:
        if bgr.shape[:2] != depth.shape[:2]:
            cv2.resize(bgr, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
            print('BGR image is resized to {:d}×{:d}.'.format(depth.shape[1], depth.shape[0]))

    if is_disp:
        H, W = disp.shape[:2]
    else:
        H, W = depth.shape[:2]
    print('W×H: {:d}×{:d}'.format(W, H))

    ## Build x, y pixel coordinates
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    if cx == 0.0 or cy == 0.0:
        cx, cy = W * 0.5, H * 0.5

    # ---------- 深度图滤波
    mask = depth > 0.0
    depth = depth * mask
    mask = depth < MAX_DEPTH
    depth = depth * mask
    print('Max depth: {:.3f}m.'.format(np.max(depth)))

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

    # ----- 点云滤波
    inds = np.where(
        (points[:, 1] < MAX_HEIGHT)
        & (points[:, 1] > -MAX_HEIGHT)
    )
    points = points[inds]
    colors = colors[inds]
    print('{:d} 3D points left.'.format(inds[0].size))

    # view_points_cloud(points)
    print('Total {:d} 3D points remained.'.format(points.shape[0]))

    # 保存pcd点云文件
    pcd_f_path = './pc.pcd'
    points2pcd(points, pcd_f_path)
    print('PCD poind cloud {:s} saved.'.format(pcd_f_path))

    # 保存ply点云文件
    ply_f_path = './ply.ply'
    points2ply(points, colors, ply_f_path)
    print('Ply poind cloud {:s} saved.'.format(ply_f_path))

    # ---------- 保存深度图
    # depth *= 256.0  # magnifying 256 times for better visua
    depth = depth.astype(np.uint16)
    depth_f_path = './depth.png'
    cv2.imwrite(depth_f_path, depth)
    print('Depth image {:s} written.'.format(depth_f_path))



if __name__ == '__main__':
    # test_xiaomi()
    # test()
    # test_depth_to_pointcloud()
    test_apollo()
    # test_kitti()
