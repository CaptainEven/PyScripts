# encoding=utf-8

import os
import cv2
import pcl
import numpy as np


def Map3DTo2D(txt_f_path):
    """
    """
    # KITTI数据集参数
    f = 721   # pixel
    b = 0.54  # m

    # 读取3D点云
