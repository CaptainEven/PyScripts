# encoding=utf-8

import math
import numpy as np


def test_clip():
    """
    :return:
    """
    arr = np.array([0, 2, 3, -5, 9, 7], dtype=np.int)
    arr = arr.clip(1)
    print(arr)

    prod = arr.prod()
    print(prod)



def box_iou_np(box1, box2):
    """
    向量化IOU计算: 利用numpy/pytorch的广播机制, 使用None扩展维度
    :param box1: (n, 4)
    :param box2: (m, 4)
    :return: (n, m)
    numpy 广播机制 从后向前对齐。 维度为1 的可以重复等价为任意维度
    eg: (4, 3, 2)   (3, 2)     (3, 2) 扩充为(4, 3, 2)
        (4, 1, 2)   (3, 2)  (4, 1, 2) 扩充为(4, 3, 2)  (3, 2)扩充为(4, 3, 2)
    扩充的方法为重复广播会在numpy的函数 如sum, maximun等函数中进行
    pytorch同理。
    扩充维度的方法：
    eg: a  a.shape: (3, 2)  a[:, None, :] a.shape: (3, 1, 2) None 对应的维度相当于newaxis
    """
    # print("box2[:, :2]:\n", box2[:, :2])
    # print("box1[:, None, :2]\n:", box1[:, None, :2])
    # print(box2[:, :2].shape)
    # print(box1[:, None, :2].shape)

    # tmp1 = box1[:, None, :2]
    # tmp2 = box2[:,  :2]
    # tmp = np.maximum(tmp1, tmp2)

    lt = np.maximum(box1[:, None, :2], box2[:,  :2])  # left_top     (x, y)
    rb = np.minimum(box1[:, None, 2:], box2[:,  2:])  # right_bottom (x, y)

    wh = np.maximum(rb - lt + 1, 0)                   # inter_area (w, h)
    inter = wh[:, :, 0] * wh[:, :, 1]                 # shape: (n, m)

    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)
    iou_matrix = inter / (box1_area[:, None] + box2_area - inter + 1e-5)

    return iou_matrix


def BBOX_IOU_NP(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    """
    :param box1: N×4
    :param box2: N×4
    :param x1y1x2y2:
    :param GIoU:
    :param DIoU:
    :param CIoU:
    :return:
    """
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = box1.T
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2

        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    b1_x2 = b1_x2.reshape(-1, 1)
    b2_x2 = b2_x2.reshape(-1, 1)
    b1_x2 = b1_x2[:, None, :]
    b2_x2 = b2_x2[None, :, :]

    b1_x1 = b1_x1.reshape(-1, 1)
    b2_x1 = b2_x1.reshape(-1, 1)
    b1_x1 = b1_x1[:, None, :]
    b2_x1 = b2_x1[None, :, :]

    r = np.minimum(b1_x2, b2_x2)
    l = np.maximum(b1_x1, b2_x1)
    r = r.clip(0)
    l = l.clip(0)
    inter_w = r - l

    b1_y2 = b1_y2.reshape(-1, 1)
    b2_y2 = b2_y2.reshape(-1, 1)
    b1_y2 = b1_y2[:, None, :]
    b2_y2 = b2_y2[None, :, :]

    b1_y1 = b1_y1.reshape(-1, 1)
    b2_y1 = b2_y1.reshape(-1, 1)
    b1_y1 = b1_y1[:, None, :]
    b2_y1 = b2_y1[None, :, :]

    b = np.minimum(b1_y2, b2_y2)
    t = np.maximum(b1_y1, b2_y1)
    b = b.clip(0)
    t = t.clip(0)
    inter_h = b - t

    inter = inter_w * inter_h
    # inter = inter.squeeze()

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    iou = inter / (union + 1e-16)  # iou
    if GIoU or DIoU or CIoU:
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height

        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            iou = iou - (c_area - union) / c_area  # GIoU
            iou = iou[:, :, 0]
            return iou

        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16

            # center-point distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                iou = iou - rho2 / c2  # DIoU
                iou = iou[:, :, 0]
                return iou

            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * np.power(np.arctan(w2 / h2) - np.arctan(w1 / h1), 2)
                alpha = v / (1 - iou + v)
                iou = iou - (rho2 / c2 + v * alpha)  # CIoU
                iou = iou[:, :, 0]
                return iou

    iou = iou[:, :, 0]
    return iou


def BBOX_ALPHA_IOU_NP(box1, box2,
                      x1y1x2y2=True,
                      GIoU=False, DIoU=False, CIoU=False,
                      eps=1e-10, alpha=2):
    """
    :param box1:
    :param box2:
    :param x1y1x2y2:
    :param GIoU:
    :param DIoU:
    :param CIoU:
    :param eps:
    :param alpha:
    :return:
    """
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = box1.T
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2

        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    b1_x2 = b1_x2.reshape(-1, 1)
    b2_x2 = b2_x2.reshape(-1, 1)
    b1_x2 = b1_x2[:, None, :]
    b2_x2 = b2_x2[None, :, :]

    b1_x1 = b1_x1.reshape(-1, 1)
    b2_x1 = b2_x1.reshape(-1, 1)
    b1_x1 = b1_x1[:, None, :]
    b2_x1 = b2_x1[None, :, :]

    r = np.minimum(b1_x2, b2_x2)
    l = np.maximum(b1_x1, b2_x1)
    r = r.clip(0)
    l = l.clip(0)
    inter_w = r - l

    b1_y2 = b1_y2.reshape(-1, 1)
    b2_y2 = b2_y2.reshape(-1, 1)
    b1_y2 = b1_y2[:, None, :]
    b2_y2 = b2_y2[None, :, :]

    b1_y1 = b1_y1.reshape(-1, 1)
    b2_y1 = b2_y1.reshape(-1, 1)
    b1_y1 = b1_y1[:, None, :]
    b2_y1 = b2_y1[None, :, :]

    b = np.minimum(b1_y2, b2_y2)
    t = np.maximum(b1_y1, b2_y1)
    b = b.clip(0)
    t = t.clip(0)
    inter_h = b - t

    inter = inter_w * inter_h
    # inter = inter.squeeze()

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1  + area2 - inter

    # iou = inter / (union + eps)  # iou
    iou = np.power(inter / union + eps, alpha)
    if GIoU or DIoU or CIoU:
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height

        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            iou = iou - (c_area - union) / c_area  # GIoU
            iou = iou[:, :, 0]
            return iou

        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16

            # center-point distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                iou = iou - rho2 / c2  # DIoU
                iou = iou[:, :, 0]
                return iou

            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * np.power(np.arctan(w2 / h2) - np.arctan(w1 / h1), 2)
                alpha = v / (1 - iou + v + eps)
                iou = iou - (rho2 / c2 + v * alpha)  # CIoU
                iou = iou[:, :, 0]
                return iou

    iou = iou[:, :, 0]
    return iou


def test_iou():
    """
    :return:
    """
    box1 = np.array([[5, 10, 15, 20],
                     [10, 15, 30, 25]], dtype=np.float)

    box2 = np.array([[0, 15, 20, 25],
                     [10, 5, 20, 25],
                     [10, 10, 25, 30]],
                    dtype=np.float)

    ious = box_iou_np(box1, box2)
    print("IOUs:\n", ious)

    ## ---------- Advanced IOUs
    print("\n")
    ious = BBOX_IOU_NP(box1, box2, x1y1x2y2=True, GIoU=True)
    print("GIOUS:\n", ious)

    ious = BBOX_IOU_NP(box1, box2, x1y1x2y2=True, DIoU=True)
    print("DIOUS:\n", ious)

    ious = BBOX_IOU_NP(box1, box2, x1y1x2y2=True, CIoU=True)
    print("CIOUS:\n", ious)

    ## ---------- ALPHA IOUs
    print("\n")
    ious = BBOX_ALPHA_IOU_NP(box1, box2, x1y1x2y2=True, GIoU=True, alpha=1.5)
    print("α-GIOUS:\n", ious)

    ious = BBOX_ALPHA_IOU_NP(box1, box2, x1y1x2y2=True, DIoU=True, alpha=1.5)
    print("α-DIOUS:\n", ious)

    ious = BBOX_ALPHA_IOU_NP(box1, box2, x1y1x2y2=True, CIoU=True, alpha=1.5)
    print("α-CIOUS:\n", ious)



if __name__ == "__main__":
    # test_clip()
    test_iou()
