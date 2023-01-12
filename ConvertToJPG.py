# encoding=utf-8

import os
import cv2
import numpy as np


def convert_to_jpg(img_path):
    """
    @param img_path:
    @return:
    """
    if not os.path.isfile(img_path):
        print("[Err]: invalid img path: {:s}"
              .format(img_path))
        exit(-1)

    img = cv2.imread(img_path)
    if img is None:
        print("[Err]: empty img")
        exit(-1)

    img_name = os.path.split(img_path)[-1]
    img_dir_path = os.path.split(img_path)[0]
    pure_name = img_name.split(".")[0]
    save_path = img_dir_path + "/" + pure_name + ".jpg"
    cv2.imwrite(save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print("[Info]: {:s} saved"
          .format(save_path))


if __name__=="__main__":
    convert_to_jpg(img_path="e:/Download/even.png")
