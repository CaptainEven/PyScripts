# encoding=utf-8

import os
import shutil
import numpy as np

 # pypiwin32
import win32gui
import win32con 

from PIL import Image


def get_img_format(img_path):
    """
    """
    img_path = os.path.abspath(img_path)
    if os.path.isfile(img_path):
        img = Image.open(img_path)
        return img, img.format
    else:
        print("[Err]: invalid img path: {:s}"
              .format(img_path))
        return None, None


src_dir = "C:/Users/Administrator/AppData/Local/Packages/Microsoft.Windows.ContentDeliveryManager_cw5n1h2txyewy/LocalState/Assets"
src_dir = os.path.abspath(src_dir)
if not os.path.isdir(src_dir):
    print("[Err]: {:s} not exists, exit now!".format(src_dir))

dst_dir = "E:/BKGImgs"
dst_dir = os.path.abspath(dst_dir)
if not os.path.isdir(dst_dir):
    print("[Err]: {:s} not exists, exit now!".format(dst_dir))


## ---------- 拷贝新壁纸到指定目录
new_dst_img_paths = []
for file_name in os.listdir(src_dir):
    src_path = src_dir + "/" + file_name
    img, img_format = get_img_format(src_path)
    img_format = img_format.lower()
    if img.width < img.height:
        continue
    print("[Info]: Img for mat: ", img_format)

    dst_path = dst_dir + "/" + file_name + "." + img_format
    if not os.path.isfile(dst_path):
        shutil.copyfile(src_path, dst_path)
        print("[Info]: {:s} cp to {:s}".format(src_path, dst_dir))
    else:
        print("[Info]: {:s} already exists".format(dst_path))
    if not dst_path in new_dst_img_paths:
        new_dst_img_paths.append(dst_path)


## ---------- 从新壁纸中随机选一张设置为当前壁纸
choosen_img_path = np.random.choice(new_dst_img_paths)
print("[Info]: setting {:s} as current wallpaper".format(choosen_img_path))
win32gui.SystemParametersInfo(win32con.SPI_SETDESKWALLPAPER, choosen_img_path, 1)
print("[Info]: set wallpaper done")