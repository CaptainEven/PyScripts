# encoding=utf-8

import os
import cv2
import exifread
from PIL import Image, ExifTags

# file_path = 'f:/tmp/baby.jpg'
# img = Image.open(file_path)
# for k, v in img._getexif().items():
#     print(ExifTags.TAGS[k], ', ', v)

# exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
# f = open(file_path, 'rb')
# tags = exifread.process_file(f)
# print(tags)


# -----------------------
# from exif import EXIF

# file_path = 'f:/tmp/baby.jpg'
# with open(file_path, 'rb') as f_h:
#     exif = EXIF(f_h)

#     focal_35, focal_ratio = exif.extract_focal()
#     print(focal_35)
#     print(focal_ratio)

import PIL.Image as Image
import numpy as np


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()  # 这个运算前后没啥区别
    print("-----------------------------------------------------------")
    print("s[0:][::2]=", s[0:][::2])  # 这个获取的是变化的像素的位置序号的列表
    # ['1', '13']
    # 这个获取的是相同像素的长度列表（分别记录每个变化的像素后面连续的同等像素值的连续长度）
    print("s[1:][::2]=", s[1:][::2])
    #['2', '2']

    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    print("看下最初的starts=", starts)  # 变化的像素的位置序号的列表
    print("lengths=", lengths)
    starts -= 1
    ends = starts + lengths
    print("ends=", ends)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):  # 进行恢复
        img[lo:hi] = 1

    return img.reshape(shape, order='F')



def test_rle_decode():
    rle_str = '1O10000O10000O1O100O100O1O100O1000000000000000O100O102N5K00O1O1N2O110OO2O001O1NTga3'
    shape = (375, 1242)

    img = rle_decode(mask_rle=rle_str, shape=shape)
    print(img)


def test_png_format_mask():
    """
    """
    img = np.array(Image.open("f:/000017.png"))
    obj_ids = np.unique(img)

    # to correctly interpret the id of a single object
    obj_id = obj_ids[0]
    class_id = obj_id // 1000
    obj_instance_id = obj_id % 1000

    print('Obj ID: ', obj_id)
    print('Class ID: ', class_id)
    print('Instance ID: ', obj_instance_id)


def cvt_jpg_to_png(img_dir):
    """
    """
    if not os.path.isdir(img_dir):
        print('[Err]: invalid image directory.')
        return

    jpg_names = [x for x in os.listdir(img_dir) if x.endswith('.jpg')]
    for jpg_name in jpg_names:
        jpg_path = img_dir + '/' + jpg_name
        png_path = img_dir + '/' + jpg_name[:-4] + '.png'

        # read in jpg img
        img = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        cv2.imwrite(png_path, img)
        os.remove(jpg_path)
        print('{:s} converted to {:s}.'.format(jpg_name, png_path))



def test():
    img_path = './humin.jpg'
    if not os.path.isfile(img_path):
        print('[Err]: invalid image path.')
        return

    img = cv2.imread(img_path)  # HWC
    if img is None:
        print('[Warning]: empty image.')
        return

    print(img[418, 0, 0])
    print(img[419, 0, 0])
    print(img[420, 0, 0])

    img_remain = img[420: 420+1080, :, :]
    print(img_remain.shape)

    img_rs = cv2.resize(img_remain, (472, 472), interpolation=cv2.INTER_CUBIC)
    img_rt = img_rs[:, 59: 59+354, :]
    print(img_rt.shape)

    # inds = np.where(img > 1.0)
    # img_remain = img[inds]
    # cv2.imshow('remain', img_rt)
    # cv2.waitKey()

    cv2.imwrite('./humin_.jpg', img_rt)


if __name__ == '__main__':
    # test_png_format_mask()
    
    # test_rle_decode()

    # cvt_jpg_to_png(img_dir='F:/MVE/mve_build/apps/dog')

    test()

    print('Done.')
