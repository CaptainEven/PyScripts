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


if __name__ == '__main__':
    # test_png_format_mask()
    
    test_rle_decode()

    print('Done.')
