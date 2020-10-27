# encoding=utf-8

import os
import cv2


def resize_img(img_in_path, w, h):
    """
    """
    if not os.path.isfile(img_in_path):
        print('[Err]: invalid image file path.')
        return
    
    img = cv2.imread(img_in_path)
    img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)
    cv2.imwrite(img_in_path, img)


if __name__ == '__main__':
    resize_img(img_in_path='f:/FairMOT/results/frame/00232.jpg', w=640, h=360)
    print('Work done.')