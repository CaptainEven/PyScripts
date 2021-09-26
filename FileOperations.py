# coding:utf-8

import os
import shutil

import cv2
import math
import numpy as np
from tqdm import tqdm


def FindFileWithSuffix(root, suffix, f_list):
    """
    递归的方式查找特定后缀文件
    """
    for f in os.listdir(root):
        f_path = os.path.join(root, f)
        if os.path.isfile(f_path) and f.endswith(suffix):
            f_list.append(f_path)
        elif os.path.isdir(f_path):
            FindFileWithSuffix(f_path, suffix, f_list)


def ChangeSuffix(root, src_suffix, dst_suffix):
    """
    """
    if not os.path.isdir(root):
        print('[Err]: invalid root.')
        return

    f_list = []
    FindFileWithSuffix(root, src_suffix, f_list)

    for f in tqdm(f_list):
        new_f = f.replace(src_suffix, dst_suffix)
        os.rename(f, new_f)


def RMFilesWithSuffix(root, suffix):
    """
        删除指定根目录下指定后缀的所有文件
    """
    if not os.path.isdir(root):
        print('[Err]: invalid root.')
        return

    f_list = []
    FindFileWithSuffix(root, suffix, f_list)

    for f in tqdm(f_list):
        os.remove(f)


def ImageRotate(image, angle, scale):
    """
    :param image:
    :param angle:
    :param scale:
    :return:
    """

    H, W, C = image.shape
    center = (H / 2, H / 2)  # H: rows

    # 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 进行仿射变换，边界填充为255，, borderValue=(255, 255, 255)
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(H, W))

    return image_rotation, M

def opencv_rotate(img, angle):
    """
    :param img: 
    :param angle: 
    :return: 
    """

    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    # 2.1获取M矩阵
    """
    M矩阵
    [
    cosA -sinA (1-cosA)*centerX+sinA*centerY
    sinA  cosA  -sinA*centerX+(1-cosA)*centerY
    ]
    """
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 2.2 新的宽高，radians(angle) 把角度转为弧度 sin(弧度)
    new_H = int(w * abs(math.sin(math.radians(angle))) + h * abs(math.cos(math.radians(angle))))
    new_W = int(h * abs(math.sin(math.radians(angle))) + w * abs(math.cos(math.radians(angle))))

    # 2.3 平移
    M[0, 2] += (new_W - w) / 2
    M[1, 2] += (new_H - h) / 2

    rotate = cv2.warpAffine(img, M, (new_W, new_H), borderValue=(0, 0, 0))
    return rotate, M


def test_img_rotation2():
    """
    :return:
    """
    img_path = "f:/5/1_25.jpg"
    if not os.path.isfile(img_path):
        print("[Err]: invalid file path.")
        return

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    H, W, C = img.shape

    pt = (962, 505)
    print("Pixel before rotate 90 CW: ", pt)

    cv2.circle(img, pt, radius=10, color=[0, 255, 0], thickness=2)
    cv2.imshow("origin", img)
    cv2.waitKey()

    img_rotate, M = opencv_rotate(img, -90)
    print("Affine matrix: \n", M)

    pt_rot = M.dot(np.array([pt[0], pt[1], 1.0]))  # 2×3 dot 3×1 ——> 2×1
    pt_rot = (int(pt_rot[0] + 0.5), int(pt_rot[1] + 0.5))
    print("Pixel after rotate 90 CW:  ", pt_rot)

    cv2.circle(img_rotate, pt_rot, radius=10, color=[0, 255, 0], thickness=2)
    cv2.imshow("rotate", img_rotate)
    cv2.waitKey()

    # cv2.imshow("rotate", img_rotate)
    # cv2.waitKey()

    cv2.imwrite("./test_rotate2.jpg", img_rotate)


def test_img_rotation():
    """
    :return:
    """
    img_path = "f:/5/1_25.jpg"
    if not os.path.isfile(img_path):
        print("[Err]: invalid file path.")
        return

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    H, W, C = img.shape

    pt = (962, 505)
    print("Pixel before rotate 90 CW: ", pt)

    cv2.circle(img, pt, radius=10, color=[0, 255, 0], thickness=2)
    cv2.imshow("origin", img)
    cv2.waitKey()

    img_rotate, M = ImageRotate(img, -90, 1.0)
    print("Affine matrix: \n", M)

    pt_rot = M.dot(np.array([pt[0], pt[1], 1.0]))  # 2×3 dot 3×1 ——> 2×1
    pt_rot = (int(pt_rot[0] + 0.5), int(pt_rot[1] + 0.5))
    print("Pixel after rotate 90 CW:  ", pt_rot)

    cv2.circle(img_rotate, pt_rot, radius=10, color=[0, 255, 0], thickness=2)
    cv2.imshow("rotate", img_rotate)
    cv2.waitKey()

    cv2.imwrite("./test_rotate.jpg", img_rotate)
    cv2.destroyAllWindows()
    print("\n\n\n")


def rotate_img(img_path="", degree=0):
    """
    """
    if not os.path.isfile(img_path):
        print("[Err]: invalid image path.")
        return

    dir_name = os.path.dirname(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w, c = img.shape
    print("W×H: {:d}×{:d}".format(w, h))

    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), degree, 1.0)   # 顺时针90°
    # rotated = cv2.warpAffine(img, M, (w, h))

    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    print(rotated.shape)

    # cv2.imshow("Rotated", rotated)
    # cv2.waitKey()

    img_name = os.path.split(img_path)[-1]
    ext = "." + img_name.split(".")[-1]
    # print(ext)

    img_name = img_name[:-len(ext)]
    save_rotate_path = dir_name + "/" + img_name + "_rotate_cw90" + ext
    # print(save_rotate_path)
    cv2.imwrite(save_rotate_path, rotated)
    print("{:s} saved.".format(save_rotate_path))


def rotate_imgs(img_dir="d:/st", ext=".jpg"):
    """
    """
    if not os.path.isdir(img_dir):
        print("[Err]: invalid img dir.")
        return

    img_paths = [img_dir + "/" + x for x in os.listdir(img_dir)
                 if x.endswith(ext)]
    print("Total {:d} imgs need to be rotated...".format(len(img_paths)))
    for img_path in img_paths:
        if not os.path.isfile(img_path):
            print("[Warning]: invalid img path.")
            continue

        ret = rotate_img(img_path, degree=-90)
        if ret is None:
            print("[Warning]: invalid img {:s}".format(img_path))


def points2pcd(points, PCD_FILE_PATH):
    """
    :param points:
    :param PCD_FILE_PATH:
    :return:
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
    :param points:
    :param colors:
    :param ply_f_path:
    :return:
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


def disp2depth(b, f, disp):
    """
    :param b:
    :param f:
    :param disp:
    :return:
    """
    disp = disp.astype(np.float64)
    non_zero_inds = np.where(disp)

    depth = np.zeros_like(disp, dtype=np.float64)
    depth[non_zero_inds] = b * f / disp[non_zero_inds]

    return depth


def get_img_pair_alignment(left, right, n_lines=16):
    """
    """
    if left is None or right is None:
        print("[Err]: invalid img.")
        return

    if left.shape != right.shape:
        print("[Err]: shape not equal!")
        return

    h, w, c = left.shape
    res = np.zeros((h, w * 2, 3), dtype=np.uint8)
    res[:, :w, :] = left
    res[:, w:w * 2, :] = right

    # -----draw lines
    n_bins = n_lines + 1
    stride = h / float(n_bins)
    for i in range(n_bins):
        y_start = i * stride
        y_end = (i + 1) * stride

        start_pt = (0, int(y_end + 0.5))
        end_pt = (w * 2 - 1, int(y_end + 0.5))
        cv2.line(res, start_pt, end_pt, (0, 255, 255), 1)

    return res


# TODO: LRC algorithm


def test_stereo_align(l_path, r_path):
    """
    :param l_path:
    :param r_path:
    :return:
    """
    if not (os.path.isfile(l_path) and os.path.isfile(r_path)):
        print("[Err]: invalid img-pair path.")
        return

    l_img = cv2.imread(l_path, cv2.IMREAD_COLOR)
    r_img = cv2.imread(r_path, cv2.IMREAD_COLOR)

    if l_img is None or r_img is None:
        print("[Err]: read image failed.")
        return

    assert l_img.shape == r_img.shape

    l_dir, l_img_name = os.path.split(l_path)[0], os.path.split(l_path)[1]
    r_dir, r_img_name = os.path.split(r_path)[0], os.path.split(r_path)[1]

    ext = "." + l_img_name.split(".")[-1]

    h, w, c = l_img.shape

    # ----- alignment check
    res = get_img_pair_alignment(l_img, r_img, n_lines=64)
    if res is None:
        print("[Err]: get alignment image failed.")
        return

    # resize alignment img to show
    if w >= h:
        while w > 1000:
            res = cv2.resize(res, (w // 2, h // 2), cv2.INTER_AREA)
            w = w // 2
            h = h // 2
    else:
        while h > 1000:
            res = cv2.resize(res, (w // 2, h // 2), cv2.INTER_AREA)
            w = w // 2
            h = h // 2

    cv2.imshow("Alignment", res)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    save_path = "{:s}/{:s}-{:s}_alignment.jpg" \
        .format(l_dir, l_img_name[:-len(ext)], r_img_name[:-len(ext)])
    cv2.imwrite(save_path, res)
    print("{:s} saved.".format(os.path.abspath(save_path)))


def test_disp_depth_pointcloud():
    """
    :return:
    """
    left_img_path = "./component_l.jpg"
    right_img_path = "./component_r.jpg"
    left_img_path = os.path.abspath(left_img_path)
    right_img_path = os.path.abspath(right_img_path)
    if not (os.path.isfile(left_img_path) and os.path.isfile(right_img_path)):
        print("[Err]: invalid image path {:s} or {:s}."
              .format(left_img_path, right_img_path))
        return

    l_bgr = cv2.imread(left_img_path, cv2.IMREAD_COLOR)  # "./img/01.bmp"
    r_bgr = cv2.imread(right_img_path, cv2.IMREAD_COLOR)  # "./img/02.bmp"
    if l_bgr is None or r_bgr is None:
        print("[Err]: empth img.")

    h, w, c = l_bgr.shape

    # get file name
    f_name = os.path.split(left_img_path)[-1][:-4]

    # ----- alignment check
    res = get_img_pair_alignment(l_bgr, r_bgr, n_lines=64)
    if res is None:
        print("[Err]: get alignment image failed.")
        return

    # resize aligment img to show
    if w >= h:
        while w > 1000:
            res = cv2.resize(res, (w // 2, h // 2), cv2.INTER_AREA)
            w = w // 2
            h = h // 2
    else:
        while h > 1000:
            res = cv2.resize(res, (w // 2, h // 2), cv2.INTER_AREA)
            w = w // 2
            h = h // 2

    cv2.imshow("Alignment", res)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    save_alignment_path = "./{:s}_align.jpg".format(f_name)
    cv2.imwrite(save_alignment_path, res)
    print("{:s} saved.".format(save_alignment_path))
    # -----

    # 将图片置为灰度图，为StereoBM作准备
    l_gray = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2GRAY)
    img_channels = 1

    # cv2.imshow("gray1", l_gray)
    # cv2.imshow("gray2", r_gray)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()

    # 两个track bar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    num = num if num > 0 else 1
    block_size = cv2.getTrackbarPos("blockSize", "depth")
    if block_size % 2 == 0:
        block_size += 1  # odd block size
    if block_size < 5:
        block_size = 5  # 5
    print('Num: ', num)
    print('Block size: ', block_size)

    # 根据Block Matching方法生成视差图(opencv里也提供了SGBM/Semi-Global Block Matching算法)
    n_disps = 16 * num
    n_disps = ((l_gray.shape[1] // 8) + 15) & -16
    print("Num of disparities: ", n_disps)

    # stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    stereo = cv2.StereoSGBM_create(numDisparities=n_disps,
                                   blockSize=block_size,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   P1=8 * img_channels * block_size * block_size,
                                   P2=32 * img_channels * block_size * block_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)

    # make border to get rid of black edges
    l_gray = cv2.copyMakeBorder(l_gray, 0, 0, n_disps, 0, cv2.BORDER_REPLICATE)
    r_gray = cv2.copyMakeBorder(r_gray, 0, 0, n_disps, 0, cv2.BORDER_REPLICATE)

    # ----- compute disparity map
    disparity = stereo.compute(l_gray, r_gray)
    # -----

    # get rid of black edges
    disparity = disparity[:, n_disps:]  # H×W

    disp_img_uint8 = cv2.normalize(disparity,
                                   disparity,
                                   alpha=0,
                                   beta=255,
                                   norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)
    # cv2.validateDisparity()
    # cv2.waitKey()

    save_disp_path = "{:s}_disp_uint8.png".format(f_name)
    cv2.imwrite(save_disp_path, disp_img_uint8)
    print("{:s} saved.".format(save_disp_path))

    ## 超参数: 用于点云截取
    MAX_DEPTH = 0.06
    MAX_HEIGHT = 5.0

    # # KITTI数据集参数
    # b = 0.54  # m
    # f = 718.335  # pixel
    # cx = 609.5593  # pixel
    # cy = 172.8540  # pixel

    # ## xiaomi参数
    # f = (998.72290039062500 + 1000.0239868164063) * 0.5  # 1000.0
    # cx = 671.15643310546875
    # cy = 384.32458496093750
    # b = 0.12  # m

    # 双目零件工作平台参数
    b = 2.6e-3  # mm -> m
    cx, cy = 0, 0
    pixel_size = 1.4e-6  # μm  -> m
    f_mm = 1.185e-3  # mm -> m
    f_pixel = f_mm / pixel_size
    f = f_pixel
    print("f(pixel): {:.3f}".format(f_pixel))

    H, W = disparity.shape[:2]
    print('W×H: {:d}×{:d}'.format(W, H))
    if cx == 0.0 or cy == 0.0:
        cx, cy = W * 0.5, H * 0.5
    print("cx: {:d}, cy: {:d}".format(int(cx), int(cy)))
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    # print(c, '\n', r)
    # x, y = np.arange(W), np.arange(H)

    # ---------- 视差图(uint16)——>深度图(float64)
    disparity = disparity / 16.0
    print("Min disparity: {:.3f}".format(np.min(disparity)))
    print("Max disparity: {:.3f}".format(np.max(disparity)))

    depth = disp2depth(b, f, disparity)
    max_depth = np.max(depth)

    # ---------- 深度图滤波
    mask = depth > 0.0
    depth = depth * mask
    mask = depth < MAX_DEPTH
    depth = depth * mask
    print('Max depth: {:.5f}mm.'.format(np.max(depth) * 1000.0))

    # --------- 深度图——>点云x, y, z
    points = np.zeros((H, W, 3), dtype=np.float64)
    colors = np.zeros((H, W, 3), dtype=np.uint8)
    points[r, c, 0] = (c - cx) * depth / f  # x
    points[r, c, 1] = (r - cy) * depth / f  # y
    points[r, c, 2] = depth  # z

    # bgr ——> rgb
    colors = l_bgr[:, :, ::-1]

    # ----- 过滤掉深度值<=0的点
    inds = np.where(points[:, :, 2] > 0.0)
    points = points[inds]
    colors = colors[inds]

    # ----- 过滤掉x, y, z全为0的点
    inds = np.where((points[:, 0] != 0.0) &
                    (points[:, 1] != 0.0) &
                    (points[:, 2] != 0.0))
    points = points[inds]
    colors = colors[inds]

    # # ----- 过滤掉高度值
    # inds = np.where(
    #     (points[:, 1] < MAX_HEIGHT)
    #     & (points[:, 1] > -MAX_HEIGHT)
    # )
    # points = points[inds]
    # colors = colors[inds]
    print('{:d} 3D points left.'.format(points.shape[0]))

    # 保存pcd点云文件
    pc_path = './{:s}.pcd'.format(f_name)
    points2pcd(points, pc_path)
    print('PCD poind cloud {:s} saved.'.format(pc_path))

    # 保存ply点云文件
    ply_path = './{:s}.ply'.format(f_name)
    points2ply(points, colors, ply_path)
    print('Ply poind cloud {:s} saved.'.format(ply_path))

    # ---------- 保存深度图
    depth *= 1000.0  # m ——> mm
    if np.max(depth) > 255.0:
        depth = depth.astype(np.uint16)
    else:
        depth = depth.astype(np.uint8)

    depth_save_path = './{:s}_depth.png'.format(f_name)
    cv2.imwrite(depth_save_path, depth)
    print('Depth image {:s} written.'.format(depth_save_path))


def filter_debug_lib(lib_path):
    """
    :param lib_path:
    :return:
    """
    lib_f_name = os.path.split(lib_path)[-1]
    return lib_f_name[:-4][-1] != 'd'


def cpFiles(src_dir, dst_dir, ext, filter_func=None):
    """
    :param src_dir:
    :param dst_dir:
    :param ext:
    :param filter_func:
    :return:
    """
    if not os.path.isdir(src_dir):
        print("[Err]: invalid src dir.")
        return

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        print("{:s} made.".format(dst_dir))

    f_paths = [src_dir + "/" +
               x for x in os.listdir(src_dir) if x.endswith(ext)]
    if filter_func is not None:
        f_paths = list(filter(filter_func, f_paths))

    for f_path in f_paths:
        f_name = os.path.split(f_path)[-1]
        # print(f_name)

        dst_path = dst_dir + "/" + f_name
        if not os.path.isfile(dst_path):
            shutil.copy(f_path, dst_dir)
            print("{:s} cp to {:s}".format(f_path, dst_dir))

    print("Total {:d} files copied.".format(len(f_paths)))


if __name__ == '__main__':
    # ChangeSuffix(root='d:/office/dense/stereo/depth_maps/dslr_images_undistorted',
    #              src_suffix='.geometric.bin',  # bin.jpg
    #              dst_suffix='.geometric_win5.bin')

    # RMFilesWithSuffix(root='d:/workspace/resultPro/depth_maps',
    # 				  suffix='_sigma_alpha_0_2_.jpg')

    # rotate_img(img_path="d:/st/camL_0001_ud.jpg")
    # rotate_imgs(img_dir="C:/TestCppProjs/StereoCalibrate/checkboards/CalibStereoMeasure/right")

    # test_stereo_align(l_path="E:/MyStereo-Refactor/data/StereoPairs/6_l.jpg",
    #                   r_path="E:/MyStereo-Refactor/data/StereoPairs/6_r.jpg")

    # test_disp_depth_pointcloud()

    # ## filter_debug_lib
    # cpFiles(src_dir="C:/Program Files/PCL 1.12.0/3rdParty/VTK/lib",
    #         dst_dir="C:/MyStereo-Refactor/3rdParty/VTK/lib",
    #         ext=".lib",
    #         filter_func=filter_debug_lib)

    test_img_rotation()
    test_img_rotation2()

    print('--Test done.\n')
