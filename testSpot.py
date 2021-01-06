#_*_coding:utf-8_*_

import copy
import os
import random
from enum import Enum
from math import floor, log, sqrt
from sys import float_info
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pywt

from imgSim import AirySpot
from tifffile import imread, imsave, imshow

from scipy import stats
from statsmodels.robust import stand_mad

# enumeration of locate methods


class LocateMethod(Enum):
    FIT = 0
    CENT_DEFAULT = 1  # the method using in our projects: radius:1
    # all the same with CENT_DEFAULT other than radius:2(using surrounding background)
    CENT_BKG = 2
    # consider the impact of second peek intensity(radius:2(using surrounding background))
    CENT_SPE = 3
    # using row/col's sum(radius:2(using surrounding background))
    GRA_DEFAULT = 4
    # using row/col's sum(radius:2(using surrounding background)) and consider second peek intensity
    GRA_SPE = 5
    INTERP_FIT = 6  # interpolate from 3*3 to 5*5 and do gauss2d fitting


# enumeration of prepare ints methods
class IntsMethod(Enum):
    DEFAULT = 0
    BI_LI_INTERP = 1


def loadPts(path):
    pts = []
    with open(path, 'rU') as fh:
        for line in fh.readlines():
            data = line.strip().split()
            pts.append((float(data[0]), float(data[1])))
    return pts


def simulateSpots(img, space, step, spot_int, airy_radius,
                  in_path=None):
    height, width = img.shape
    # fill with spots
    pts_orig = []
    bias_x = 0.0
    bias_y = 0.0
    if not in_path:
        start_x = int(round(width * 0.1))
        start_y = int(round(height * 0.1))
        end_x = int(round(width * 0.9))
        end_y = int(round(height * 0.9))
        for y in range(start_y, end_y, space):
            for x in range(start_x, end_x, space):
                the_x = x + bias_x
                the_y = y + bias_y
                ap.createSpot(img, the_y, the_x, airy_radius, spot_int)
                pts_orig.append((the_x, the_y))
                bias_x = bias_x + step if bias_x + step <= 1.0 else 0.0
            bias_y = bias_y + step if bias_y + step <= 1.0 else 0.0
        return (img, pts_orig)
    else:
        pts = loadPts(in_path)
        for pt in pts:
            ap.createSpot(img, pt[1], pt[0], 5.0, spot_int)
        return (img, pts)


# gauss2d fit with 5*5, radius = 2
# q.shape == (25, 25)
# r.shape == (5, 5)
def gauss2dFit(pt_round, ints):
    vect_A = np.zeros(25)
    matr_B = np.zeros((25, 5))
    i = 0
    for y in range(5):
        for x in range(5):
            vect_A[i] = ints[y][x] * log(ints[y][x])
            matr_B[i][0] = ints[y][x]
            matr_B[i][1] = ints[y][x] * x
            matr_B[i][2] = ints[y][x] * y
            matr_B[i][3] = ints[y][x] * x * x
            matr_B[i][4] = ints[y][x] * y * y
            i += 1
    # matrix decomposion and block operation
    q, r = np.linalg.qr(matr_B)
    C = np.linalg.inv(r).dot(q.T.dot(vect_A)[:6])
    x = float(pt_round[0]) - 2.0 - 0.5 * C[1] / C[3]
    y = float(pt_round[1]) - 2.0 - 0.5 * C[2] / C[4]
    return (x, y)


# compute spot background
def calcSpotBkg(img, pt, radius):
    x = pt[0]
    y = pt[1]
    length = (radius << 1) + 1
    min_val = float_info.max
    for r in range(length):
        for c in range(length):
            if img[y - radius + r][x - radius + c] < min_val:
                min_val = img[y - radius + r][x - radius + c]
    return min_val


# calculate centroid
def calcSpotCentDefault(img, pt, radius=1):
    x = pt[0]
    y = pt[1]
    bkg = calcSpotBkg(img, pt, radius)
    # print('bkg: %.3f' %bkg)
    the_x = (img[y][x + 1] - img[y][x - 1]) / \
        (img[y][x - 1] + img[y][x] + img[y][x + 1] - 3.0 * bkg)
    the_y = (img[y + 1][x] - img[y - 1][x]) / (img[y - 1]
                                               [x] + img[y][x] + img[y + 1][x] - 3.0 * bkg)
    return (float(x) + the_x, float(y) + the_y)


# calculate centroid using more detailed and specific strategy
def calcSpotCentSpe(img, pt, ratio_th, radius=2):
    x = pt[0]
    y = pt[1]
    sec_peek = max(max(img[y - 1][x], img[y + 1][x]),
                   max(img[y][x - 1], img[y][x + 1]))
    ratio = sec_peek / img[y][x]
    if ratio < ratio_th:
        return calcSpotCentDefault(img, pt, radius)
    else:  # we need to consider second peek
        bkg = calcSpotBkg(img, pt, radius)  # calculate background
        # print('bkg: %.3f' %bkg)
        if sec_peek == img[y][x - 1]:  # left
            the_x = (-2.0 * (img[y][x - 2] - bkg) - (img[y][x - 1] - bkg) + (img[y][x + 1] - bkg)) / (
                img[y][x - 2] + img[y][x - 1] + img[y][x] + img[y][x + 1] - 4.0 * bkg)
            the_y = (img[y + 1][x] - img[y - 1][x]) / (img[y - 1]
                                                       [x] + img[y][x] + img[y + 1][x] - 3.0 * bkg)
        elif sec_peek == img[y][x + 1]:  # right
            the_x = ((img[y][x + 1] - bkg) - (img[y][x - 1] - bkg) + 2.0 * (img[y][x + 2] - bkg)
                     ) / (img[y][x - 1] + img[y][x] + img[y][x + 1] + img[y][x + 2] - 4.0 * bkg)
            the_y = (img[y + 1][x] - img[y - 1][x]) / (img[y - 1]
                                                       [x] + img[y][x] + img[y + 1][x] - 3.0 * bkg)
        elif sec_peek == img[y - 1][x]:  # up
            the_x = (img[y][x + 1] - img[y][x - 1]) / \
                (img[y][x - 1] + img[y][x] + img[y][x + 1] - 3.0 * bkg)
            the_y = (-2.0 * (img[y - 2][x] - bkg) - (img[y - 1][x] - bkg) + (img[y + 1][x] - bkg)) / (
                img[y - 2][x] + img[y - 1][x] + img[y][x] + img[y + 1][x] - 4.0 * bkg)
        else:  # down
            the_x = (img[y][x + 1] - img[y][x - 1]) / \
                (img[y][x - 1] + img[y][x] + img[y][x + 1] - 3.0 * bkg)
            the_y = ((img[y + 1][x] - bkg) - (img[y - 1][x] - bkg) + 2.0 * (img[y + 2][x] - bkg)
                     ) / (img[y - 1][x] + img[y][x] + img[y + 1][x] + img[y + 2][x] - 4.0 * bkg)
        return (float(x) + the_x, float(y) + the_y)


# using row/col's sum as intensity: length*length
def calcSpotGravDefault(img, pt, radius=2):
    # prepare ints
    x = pt[0]
    y = pt[1]
    length = (radius << 1) + 1
    bkg = calcSpotBkg(img, pt, radius)
    ints_r = np.zeros(length)
    ints_c = np.zeros(length)
    for i in range(length):
        int_r = 0.0
        int_c = 0.0
        for j in range(length):
            int_r += img[y - radius + i][x - radius + j] - bkg
            int_c += img[y - radius + j][x - radius + i] - bkg
        ints_r[i] = int_r
        ints_c[i] = int_c

    # calculate gravity
    nume_r = 0.0
    nume_c = 0.0  # 分子
    deno_r = 0.0
    deno_c = 0.0  # 分母
    for i in range(length):
        nume_r += float(i - radius) * (ints_r[i])
        nume_c += float(i - radius) * (ints_c[i])
        deno_r += ints_r[i]
        deno_c += ints_c[i]
    return (float(x) + nume_c / deno_c, float(y) + nume_r / deno_r)


# using row/col's sum as intensity: length*length
# consider second peek
def calcSpotGravSpe(img, pt, ratio_th, radius=2):
    x = pt[0]
    y = pt[1]
    sec_peek = max(max(img[y - 1][x], img[y + 1][x]),
                   max(img[y][x - 1], img[y][x + 1]))
    ratio = sec_peek / img[y][x]
    if ratio < ratio_th:
        return calcSpotGravDefault(img, pt, radius)
    else:  # we need to consider second peek
        bkg = calcSpotBkg(img, pt, radius)  # calculate background
        if sec_peek == img[y][x - 1]:  # left
            ints_r_n1 = img[y - 1][x - 2] + img[y - 1][x - 1] + \
                img[y - 1][x] + img[y - 1][x + 1] - 4.0 * bkg
            ints_r_0 = img[y][x - 2] + img[y][x - 1] + \
                img[y][x] + img[y][x + 1] - 4.0 * bkg
            ints_r_p1 = img[y + 1][x - 2] + img[y + 1][x - 1] + \
                img[y + 1][x] + img[y + 1][x + 1] - 4.0 * bkg
            the_y = (ints_r_p1 - ints_r_n1) / \
                (ints_r_n1 + ints_r_0 + ints_r_p1)
            ints_c_n2 = img[y - 1][x - 2] + \
                img[y][x - 2] + img[y + 1][x - 2] - 3.0 * bkg
            ints_c_n1 = img[y - 1][x - 1] + \
                img[y][x - 1] + img[y + 1][x - 1] - 3.0 * bkg
            ints_c_0 = img[y - 1][x] + img[y][x] + img[y + 1][x] - 3.0 * bkg
            ints_c_p1 = img[y - 1][x + 1] + \
                img[y][x + 1] + img[y + 1][x + 1] - 3.0 * bkg
            the_x = (-2.0 * ints_c_n2 - ints_c_n1 + ints_c_p1) / \
                (ints_c_n2 + ints_c_n1 + ints_c_0 + ints_c_p1)
        elif sec_peek == img[y][x + 1]:  # right
            ints_r_n1 = img[y - 1][x - 1] + img[y - 1][x] + \
                img[y - 1][x + 1] + img[y - 1][x + 2] - 4.0 * bkg
            ints_r_0 = img[y][x - 1] + img[y][x] + \
                img[y][x + 1] + img[y][x + 2] - 4.0 * bkg
            ints_r_p1 = img[y + 1][x - 1] + img[y + 1][x] + \
                img[y + 1][x + 1] + img[y + 1][x + 2] - 4.0 * bkg
            the_y = (ints_r_p1 - ints_r_n1) / \
                (ints_r_n1 + ints_r_0 + ints_r_p1)
            ints_c_n1 = img[y - 1][x - 1] + \
                img[y][x - 1] + img[y + 1][x - 1] - 3.0 * bkg
            ints_c_0 = img[y - 1][x] + img[y][x] + img[y + 1][x] - 3.0 * bkg
            ints_c_p1 = img[y - 1][x + 1] + \
                img[y][x + 1] + img[y + 1][x + 1] - 3.0 * bkg
            ints_c_p2 = img[y - 1][x + 2] + \
                img[y][x + 2] + img[y + 1][x + 2] - 3.0 * bkg
            the_x = (ints_c_p1 - ints_c_n1 + 2.0 * ints_c_p2) / \
                (ints_c_n1 + ints_c_0 + ints_c_p1 + ints_c_p2)
        elif sec_peek == img[y - 1][x]:  # up
            ints_r_n2 = img[y - 2][x - 1] + \
                img[y - 2][x] + img[y - 2][x + 1] - 3.0 * bkg
            ints_r_n1 = img[y - 1][x - 1] + \
                img[y - 1][x] + img[y - 1][x + 1] - 3.0 * bkg
            ints_r_0 = img[y][x - 1] + img[y][x] + img[y][x + 1] - 3.0 * bkg
            ints_r_p1 = img[y + 1][x - 1] + \
                img[y + 1][x] + img[y + 1][x + 1] - 3.0 * bkg
            the_y = (-2.0 * ints_r_n2 - ints_r_n1 + ints_r_p1) / \
                (ints_r_n2 + ints_r_n1 + ints_r_0 + ints_r_p1)
            ints_c_n1 = img[y - 2][x - 1] + img[y - 1][x - 1] + \
                img[y][x - 1] + img[y + 1][x - 1] - 4.0 * bkg
            ints_c_0 = img[y - 2][x] + img[y - 1][x] + \
                img[y][x] + img[y + 1][x] - 4.0 * bkg
            ints_c_p1 = img[y - 2][x + 1] + img[y - 1][x + 1] + \
                img[y][x + 1] + img[y + 1][x + 1] - 4.0 * bkg
            the_x = (ints_c_p1 - ints_c_n1) / \
                (ints_c_n1 + ints_c_0 + ints_c_p1)
        else:  # down
            ints_r_n1 = img[y - 1][x - 1] + \
                img[y - 1][x] + img[y - 1][x + 1] - 3.0 * bkg
            ints_r_0 = img[y][x - 1] + img[y][x] + img[y][x + 1] - 3.0 * bkg
            ints_r_p1 = img[y + 1][x - 1] + \
                img[y + 1][x] + img[y + 1][x + 1] - 3.0 * bkg
            ints_r_p2 = img[y + 2][x - 1] + \
                img[y + 2][x] + img[y + 2][x + 1] - 3.0 * bkg
            the_y = (ints_r_p1 - ints_r_n1 + 2.0 * ints_r_p2) / \
                (ints_r_n1 + ints_r_0 + ints_r_p1 + ints_r_p2)
            ints_c_n1 = img[y - 1][x - 1] + img[y][x - 1] + \
                img[y + 1][x - 1] + img[y + 2][x - 1] - 4.0 * bkg
            ints_c_0 = img[y - 1][x] + img[y][x] + \
                img[y + 1][x] + img[y + 2][x] - 4.0 * bkg
            ints_c_p1 = img[y - 1][x + 1] + img[y][x + 1] + \
                img[y + 1][x + 1] + img[y + 2][x + 1] - 4.0 * bkg
            the_x = (ints_c_p1 - ints_c_n1) / \
                (ints_c_n1 + ints_c_0 + ints_c_p1)
    return (float(x) + the_x, float(y) + the_y)


# bilinear interpolation from 3*3 to len*len
def biliInterpolate(img, pt, len):
    x = pt[0]
    y = pt[1]
    img_inter = np.zeros((len, len))
    for r in range(len):
        for c in range(len):
            the_x = (float(c) + 0.5) * 3.0 / float(len) - 0.5
            the_y = (float(r) + 0.5) * 3.0 / float(len) - 0.5
            X = floor(the_x)
            Y = floor(the_y)
            u = the_x - float(X)
            v = the_y - float(Y)
            img_inter[r][c] = (1.0 - u) * (1.0 - v) * img[y - 1 + Y][x - 1 + X] \
                + (1.0 - u) * v * img[y + Y][x - 1 + X] \
                + u * (1.0 - v) * img[y - 1 + Y][x + X] \
                + u * v * img[y + Y][x + X]
    return img_inter


# compute signal noise ratio of a DNB spot
# center is rounded center
# something's wrong?
def calcSNRMean(img, noise, pts_ori):
    SNRs = np.zeros(len(pts_ori))
    for k, pt in enumerate(pts_ori):
        x = round(pt[0])
        y = round(pt[1])
        total_enenrgy = 0.0
        total_noise = 0.0
        for r in range(5):
            for c in range(5):
                total_enenrgy += img[y - 2 + r][x - 2 + c]
                total_noise += noise[y - 2 + r][x - 2 + c]
        SNRs[k] = (total_enenrgy - total_noise) / total_noise
        # print('Spot %d SNR: %.3f' %(k, SNRs[k]))
    return np.mean(SNRs)


# @ints: intensities of 5*5 points
def calcDeltas(img, pts_orig, method=LocateMethod.FIT):
    pts_out = []
    deltas_x = []
    deltas_y = []
    delta_sum_x = 0.0
    delta_sum_y = 0.0
    for i, pt in enumerate(pts_orig):
        # prepare pt_round and  5*5 ints
        pt_round = (round(pt[0]), round(pt[1]))
        ints = prepInts(img, pt_round, IntsMethod.DEFAULT)
        # do sub_pixel locating
        if method == LocateMethod.FIT:
            the_x, the_y = gauss2dFit(pt_round, ints)
            pts_out.append((the_x, the_y))
        elif method == LocateMethod.CENT_DEFAULT:
            the_x, the_y = calcSpotCentDefault(img, pt_round)
            pts_out.append((the_x, the_y))
        # compute delta
        delta_x = the_x - pts_orig[i][0]
        delta_y = the_y - pts_orig[i][1]
        deltas_x.append(delta_x)
        deltas_y.append(delta_y)
        delta_sum_x += abs(delta_x)  # update delta's sum
        delta_sum_y += abs(delta_y)
        # print('delta: [%.3f, %.3f]' % (delta_x, delta_y))
    # compute delta's mean
    delta_mean_x = delta_sum_x / float(len(deltas_x))
    delta_mean_y = delta_sum_y / float(len(deltas_y))
    print('delta_x mean: %.3fpixel\ndelta_y mean: %.3fpixel' %
          (delta_mean_x, delta_mean_y))
    return pts_out, deltas_x, deltas_y


# calculate x,y delta mean
def calcDeltaMean(img, pts_orig, locate_method=LocateMethod.FIT):
    delta_sum_x = 0.0
    delta_sum_y = 0.0
    for i, pt in enumerate(pts_orig):
        # prepare pt_round and 5*5 ints
        pt_round = (round(pt[0]), round(pt[1]))
        # do sub_pixel locating
        if locate_method == LocateMethod.FIT:
            ints = prepInts(img, pt_round, IntsMethod.DEFAULT)
            the_x, the_y = gauss2dFit(pt_round, ints)  # gauss2d(default)
        elif locate_method == LocateMethod.INTERP_FIT:
            ints = prepInts(img, pt_round, IntsMethod.BI_LI_INTERP)
            # gauss2d(interpolate from 3*3 to 5*5)
            the_x, the_y = gauss2dFit(pt_round, ints)
        elif locate_method == LocateMethod.CENT_DEFAULT:
            ints = prepInts(img, pt_round, IntsMethod.DEFAULT)
            the_x, the_y = calcSpotCentDefault(
                img, pt_round, 1)  # centroid(default)
        elif locate_method == LocateMethod.CENT_BKG:
            ints = prepInts(img, pt_round, IntsMethod.DEFAULT)
            # centroid(using surrounding bkg, radius:2 )
            the_x, the_y = calcSpotCentDefault(img, pt_round, 2)
        elif locate_method == LocateMethod.CENT_SPE:
            ints = prepInts(img, pt_round, IntsMethod.DEFAULT)
            # centroid(consider second peek and radius:2)
            the_x, the_y = calcSpotCentSpe(img, pt_round, 0.8, 2)
        elif locate_method == LocateMethod.GRA_DEFAULT:
            ints = prepInts(img, pt_round, IntsMethod.DEFAULT)
            the_x, the_y = calcSpotGravDefault(
                img, pt_round, 2)  # gravity(default)
        elif locate_method == LocateMethod.GRA_SPE:
            ints = prepInts(img, pt_round, IntsMethod.DEFAULT)
            # gravity(consider second peek)
            the_x, the_y = calcSpotGravSpe(img, pt_round, 0.8, 2)

        # update delta's sum
        delta_sum_x += abs(the_x - pts_orig[i][0])
        delta_sum_y += abs(the_y - pts_orig[i][1])
        # print('delta: [%.3f, %.3f]' % (delta_x, delta_y))
    # compute delta's mean
    delta_mean_x = delta_sum_x / float(len(pts_orig))
    delta_mean_y = delta_sum_y / float(len(pts_orig))
    return (delta_mean_x, delta_mean_y)


# prepare 5*5 ints from 3*3 ints
def prepInts(img, pt, method=IntsMethod.DEFAULT):
    ints = np.zeros((5, 5))
    if method == IntsMethod.DEFAULT:
        for y in range(0, 5):
            for x in range(0, 5):
                ints[y][x] = img[pt[1] - 2 + y][pt[0] - 2 + x]
    elif method == IntsMethod.BI_LI_INTERP:
        ints = biliInterpolate(img, pt, 5)
    return ints


# judge DNB Spot in real image
def judgeDNB(img, pt):
    x = pt[0]
    y = pt[1]
    if (x < 2) or (x >= img.shape[1] - 2) \
            or (y < 2) or (y >= img.shape[0] - 2):
        print("[Error]: pt out of range.")
        return False
    # judge whether 0 > 1
    if ((img[y][x] < img[y - 1][x]) or
        (img[y][x] < img[y + 1][x]) or
        (img[y][x] < img[y][x - 1]) or
            (img[y][x] < img[y][x + 1])):
        return False
    # judge whether 1 > sqrt(2)
    ints_1 = np.zeros(4)
    ints_1[0] = img[y][x - 1]
    ints_1[1] = img[y][x + 1]
    ints_1[2] = img[y - 1][x]
    ints_1[3] = img[y + 1][x]
    ints_sqrt2 = np.zeros(4)
    ints_sqrt2[0] = img[y - 1][x - 1]
    ints_sqrt2[1] = img[y - 1][x + 1]
    ints_sqrt2[2] = img[y + 1][x - 1]
    ints_sqrt2[3] = img[y + 1][x + 1]
    for i in range(4):
        for j in range(4):
            if ints_1[i] < ints_sqrt2[j]:
                return False
    # judge whether sqrt(2) > 2
    ints_2 = np.zeros(4)
    ints_2[0] = img[y - 2][x]
    ints_2[1] = img[y + 2][x]
    ints_2[2] = img[y][x - 2]
    ints_2[3] = img[y][x + 2]
    for i in range(4):
        for j in range(4):
            if ints_sqrt2[i] < ints_2[j]:
                return False
    # judge whether 2 > 2sqrt2
    ints_2sqrt2 = np.zeros(4)
    ints_2sqrt2[0] = img[y - 2][x - 2]
    ints_2sqrt2[1] = img[y - 2][x + 2]
    ints_2sqrt2[2] = img[y + 2][x - 2]
    ints_2sqrt2[3] = img[y + 2][x + 2]
    for i in range(4):
        for j in range(4):
            if ints_2sqrt2[i] > ints_2[j]:
                return False
    return True


# scan the whole real image for DNB Spots
# 5 pixels interval between the DNBs at least
def scanDNBSpots(img):
    DNBs = []
    rows, cols = img.shape
    rows -= 2
    cols -= 2
    y = 2
    while y < rows:
        flag_DNB = False
        x = 2
        while x < cols:
            if judgeDNB(img, (x, y)):
                DNBs.append((x, y))
                x += 5
                flag_DNB = True
            else:
                x += 1
        if flag_DNB:
            y += 2
        else:
            y += 1
    return DNBs


# plot deltas
def plotXYDeltas(deltas_x, deltas_y):
    num = len(deltas_x) if len(deltas_x) == len(deltas_y) else 0
    if num == 0:
        print('[Error]: parameter wrong.')
        return
    x = np.linspace(1, num, num)
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    plt.title('X,Y offsets')
    plt.xlabel('Point ID')
    plt.ylabel('Offset(pixel)')
    line_x = ax.plot(x, deltas_x, 'bo', label='X offset')
    line_y = ax.plot(x, deltas_y, 'ro', label='Y offset')
    legend = ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    legend.get_frame().set_facecolor('#00FFCC')
    plt.show()


# plot accuracy statistics, and only compare 2 lines
def plotAccuracy(SNRs, deltas_0, deltas_1):
    num = len(deltas_0) if len(deltas_0) == len(deltas_1) else 0
    if num == 0:
        print('[Error]: parameter wrong.')
        return
    x = np.linspace(1, num, num)
    fig = plt.figure(figsize=(8, 8))
    ax_0 = plt.subplot(211)
    plt.title('Accuracy comparision')
    plt.ylabel('Offset(pixel)')
    line_0 = ax_0.plot(x, deltas_0, 'ro-', label='Fit')
    line_1 = ax_0.plot(x, deltas_1, 'bo-', label='Cen')
    legend = ax_0.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    legend.get_frame().set_facecolor('#00FFCC')
    ax_1 = plt.subplot(212)
    plt.xlabel('Round ID')
    plt.ylabel('SNR')
    line_2 = ax_1.plot(x, SNRs, 'ko-', label='SNR')
    plt.show()


# presume: all delta has indentical size and compare 3 lines
def plotAccuracies(SNRs, deltas):
    line_num = len(deltas)
    pt_num = len(deltas[0])
    x = np.linspace(1, pt_num, pt_num)
    fig = plt.figure(figsize=(9, 8))
    ax_0 = plt.subplot(211)
    plt.title('Accuracy comparision')
    plt.ylabel('Offset(pixel)')
    color_map = plt.get_cmap("RdYlGn")
    for i in range(line_num):
        if i == 0:
            label = 'fit'
        elif i == 1:
            label = 'cent1'
        elif i == 2:
            label = 'cent2'
        elif i == 3:
            label = 'grav1'
        elif i == 4:
            label = 'grav2'
        line = ax_0.plot(x, deltas[i], 'o-',
                         color=color_map(abs(i / line_num * 0.45)), label=label)
    legend_0 = ax_0.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    legend_0.get_frame().set_facecolor('#00FFCC')
    ax_1 = plt.subplot(212)
    plt.xlabel('Round ID')
    plt.ylabel('SNR')
    line_snr = ax_1.plot(x, SNRs, 'bo-', label='SNR')
    legend_1 = ax_1.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    legend_1.get_frame().set_facecolor('#00FFCC')
    plt.show()


# Output
def storePts(pts, path='./pts.txt'):
    with open(path, 'w') as fh:
        for pt in pts:
            fh.write(str(pt[0]))
            fh.write('\t')
            fh.write(str(pt[1]))
            fh.write('\n')


def output(pts_ori, pts_fit,
           ori_path='./pts_ori.txt', fit_path='./pts_out.txt'):
    fh_ori = open(ori_path, 'w')
    fh_fit = open(fit_path, 'w')
    for pt in zip(pts_ori, pts_fit):
        fh_ori.write(str(pt[0][0]))
        fh_ori.write('\t')
        fh_ori.write(str(pt[0][1]))
        fh_ori.write('\n')
        fh_fit.write(str(pt[1][0]))
        fh_fit.write('\t')
        fh_fit.write(str(pt[1][1]))
        fh_fit.write('\n')
    fh_ori.close()
    fh_fit.close()


# remove files under a directory with specific suffix
def rmFiles(dir='./', suffix='.tif'):
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith(suffix):
                print('[delete file] %s' % name)
                os.remove(os.path.join(root, name))


################################Test functions################################

# different x,y in subpxiel lead to different distribution shape of DNB Spot
ap = AirySpot(3)

# Test real image's DNB Spot location


def testRealSpot(file_path):
    print('-- Test real spot', flush=True)
    src = imread(file_path)
    DNBs = scanDNBSpots(src)
    print('DNB spot number: %d' % len(DNBs), flush=True)
    # storePts(DNBs, './DNBs.txt')
    DNBs_fit = []
    for pt in DNBs:
        ints = np.zeros((5, 5))
        for y in range(5):
            for x in range(5):
                ints[y][x] = src[pt[1] - 2 + y][pt[0] - 2 + x]
        the_x, the_y = gauss2dFit(pt, ints)
        DNBs_fit.append((the_x, the_y))
        print('ori:[%.3f, %.3f],fit:[%.3f, %.3f]' %
              (pt[0], pt[1], the_x, the_y), flush=True)
    # storePts(DNBs_fit, './DNBs_fit.txt')


# Test ideal DNB Spots without random backgrounds
def testIdealSpots():
    print('-- Test ideal simulated spot')
    # init img with non background
    src = np.zeros((400, 400))
    # simulate spots
    img, pts_ori = simulateSpots(src, 15, 0.1, 120000.0, 3.0)
    # do gauss2d fitting and compute accuracy
    pts_fit, deltas_x, deltas_y = calcDeltas(img, pts_ori)
    output(pts_ori, pts_fit)
    plotXYDeltas(deltas_x, deltas_y)


# Test DNB Spots add Gauss distributed background noise
def testNoisedSpots(out_path='./spot.tif'):
    print('\n-- Test noised simulated spot')
    # add normal distributed background noise
    src = np.random.normal(2000, 300, (400, 400))
    noise = copy.copy(src)

    # simulate spots
    img, pts_ori = simulateSpots(src, 15, 0.1, 30000.0, 3.0)
    # imsave(out_path, img.astype(dtype='uint16'))

    # calc SNRs' mean
    SNR = calcSNRMean(img, noise, pts_ori)
    print('SNR mean: %.3f' % SNR)

    # locate and calculate deltas(accuracy)
    pts_fit, deltas_x, deltas_y = calcDeltas(img, pts_ori, LocateMethod.FIT)
    output(pts_ori, pts_fit)
    plotXYDeltas(deltas_x, deltas_y)
    imshow(img, 'Simulated spots')
    plt.show()


# compare different locating methods
def compLocateMethods(energy_init, energy_end, step):
    print('-- Compare guass_2d and centroid')
    if step <= 0.0 or energy_init <= energy_end:
        print('[Error]: parameters wrong.')
        return None
    SNRs = []
    deltas_fit_x = []
    deltas_fit_y = []
    deltas_cen_x_1 = []
    deltas_cen_y_1 = []
    deltas_cen_bkg_x_1 = []
    deltas_cen_bkg_y_1 = []
    deltas_cen_x_2 = []
    deltas_cen_y_2 = []
    deltas_gra_x_1 = []
    deltas_gra_y_1 = []
    deltas_gra_x_2 = []
    deltas_gra_y_2 = []
    accus_x = []
    # accus_y = []

    energy = energy_init
    round_id = 0
    while energy > energy_end:
        print('-- Round: %d' % (round_id + 1))
        # init spotted image
        src = np.random.normal(1600, 300, (400, 400))
        noise = copy.copy(src)
        img, pts_ori = simulateSpots(src, 15, 0.1, energy, 3.0)
        # imsave('./spot.tif', img.astype(dtype='uint16'))

        # compute SNRs
        SNR = calcSNRMean(img, noise, pts_ori)
        SNRs.append(SNR)
        print('SNR mean: %.2f' % SNR)

        # locate by gauss2d fitting and calculate deltas(accuracy)
        delta_x, delta_y = calcDeltaMean(img, pts_ori, LocateMethod.FIT)
        deltas_fit_x.append(delta_x)
        deltas_fit_y.append(delta_y)
        print('fit delta: (%.3f, %.3f)' % (delta_x, delta_y), flush=True)

        # locate by centroid(default) and calculate deltas(accuracy)
        delta_x, delta_y = calcDeltaMean(
            img, pts_ori, LocateMethod.CENT_DEFAULT)
        deltas_cen_x_1.append(delta_x)
        deltas_cen_y_1.append(delta_y)
        print('cen delta_1: (%.3f, %.3f)' % (delta_x, delta_y), flush=True)

        # locate by centroid(bkg) and calculate deltas(accuracy)
        # delta_x, delta_y = calcDeltaMean(img, pts_ori, LocateMethod.CENT_BKG)
        # deltas_cen_bkg_x_1.append(delta_x)
        # deltas_cen_bkg_y_1.append(delta_y)
        # print('cent_bkg delta_1: (%.3f, %.3f)' %(delta_x, delta_y), flush=True)

        # locate by centroid(special) and calculate deltas(accuracy)
        delta_x, delta_y = calcDeltaMean(img, pts_ori, LocateMethod.CENT_SPE)
        deltas_cen_x_2.append(delta_x)
        deltas_cen_y_2.append(delta_y)
        print('cen delta_2: (%.3f, %.3f)' % (delta_x, delta_y), flush=True)

        # locate by gravity(default) and calculate deltas(accuracy)
        delta_x, delta_y = calcDeltaMean(
            img, pts_ori, LocateMethod.GRA_DEFAULT)
        deltas_gra_x_1.append(delta_x)
        deltas_gra_y_1.append(delta_y)
        print('gra delta_1: (%.3f, %.3f)' % (delta_x, delta_y), flush=True)

        # locate by gravity(default) and calculate deltas(accuracy)
        delta_x, delta_y = calcDeltaMean(img, pts_ori, LocateMethod.GRA_SPE)
        deltas_gra_x_2.append(delta_x)
        deltas_gra_y_2.append(delta_y)
        print('gra delta_2: (%.3f, %.3f)' % (delta_x, delta_y), flush=True)

        # update energy and count
        energy -= step
        round_id += 1
    accus_x.append(deltas_fit_x)    # 0
    accus_x.append(deltas_cen_x_1)  # 1
    accus_x.append(deltas_cen_x_2)  # 2
    accus_x.append(deltas_gra_x_1)  # 3
    accus_x.append(deltas_gra_x_2)  # 4
    # accus_x.append(deltas_cen_bkg_x_1)
    plotAccuracies(SNRs, accus_x)
    return (SNRs, accus_x)


# 将SNR与模拟点、真实点以及cycle id对应起来并做统计
# 先为每一轮的结果创建一个子目录，选取同一个ID的Spot，存图
def doStatistics(energy_init, energy_end, step, spot_id):
    if energy_init <= energy_end:
        print('[Error]: parameters wrong')
        return
    print('-- Start statistics...')
    SIZE = int((energy_init - energy_end) / step) + 1
    x = 0
    y = 0
    for i in range(SIZE):
        # make dir
        data_dir = os.getcwd() + '/statistics/round' + str(i)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        rmFiles(data_dir, '.tif')  # clear before save image

        # init spot image
        src = np.random.normal(1500, 300, (400, 400))
        noise = copy.copy(src)
        energy = energy_init - i * step
        img, pts_ori = simulateSpots(src, 15, 0.1, energy, 3.0)

        # calc SNRs' mean
        SNR = calcSNRMean(img, noise, pts_ori)
        print('[Round %d] SNR mean: %.2f' % (i, SNR), flush=True)
        if i == 0:
            x = int(round(pts_ori[spot_id][0]))
            y = int(round(pts_ori[spot_id][1]))

        # save specified spot
        ints = prepInts(img, (x, y), IntsMethod.DEFAULT)
        imsave(data_dir + './snr' + str(round(SNR, 2)) +
               '.tif', ints.astype(dtype='uint16'))


# test bilinear interpolation
def testBilinearInterp(file_path, pt):
    print('-- Test bilinear interpolation')
    src = imread(file_path)
    interp = biliInterpolate(src, pt, 5)
    print('interp:', interp)
    # imshow(interp, './interp.tif')
    # plt.show()
    # imsave('./interp.tif', interp.astype(dtype='uint16'))

# 如何较为准确地估计真实图像中某个Spot的SNR？
# 真实图像根据SNR选取不同的centroid计算方案？


# 读入一张图及其track cross坐标，
# 在此基础上估计每个track cross所在的track区域的背景
# 做曲线(指数)拟合,曲率最大点即background阈值?直接做貌似行不通
def calcTrackBkg(img, id, pt, ratio, R, the_id=-1):
    # prepare track area intensities
    ints, X, Y = getTrackAreaInts(img, pt, R)
    print('Total %d DNB spots.' % len(ints))

    # sort and get background
    ints.sort()
    bkg = ints[int(round(len(ints) * ratio))]
    if id != the_id:
        return (bkg, None, None, None)
    else:
        return (bkg, X, Y, ints)

 # 计算Track区域背景的参数简化版


def computeTrackBkg(img, pt, ratio, R):
    # prepare track area intensities
    ints, X, Y = getTrackAreaInts(img, pt, R)
    print('total %d pixels.' % len(ints))

    # sort and get background
    ints.sort()
    return ints[int(round(len(ints) * ratio))]


# extract track area intensities
def getTrackAreaInts(img, pt, R):
    X = int(round(pt[0]))
    Y = int(round(pt[1]))
    width = img.shape[1]
    height = img.shape[0]
    X = X if X >= 0 else 0
    X = X if X < width else width - 1
    Y = Y if Y >= 0 else 0
    Y = Y if Y < height else height - 1
    print('ref point[%d, %d] ' % (X, Y), end='')
    ints = []

    # horizontal pixels
    start_x = X - R
    end_x = X + R
    start_y = Y - 1
    end_y = Y + 1
    start_x = 0 if start_x <= 0 else start_x
    start_y = 0 if start_y <= 0 else start_y
    end_x = end_x if end_x < width else width - 1
    end_y = end_y if end_y < height else height - 1
    for r in range(start_y, end_y + 1):
        for c in range(start_x, end_x + 1):
            if img[r][c]:
                ints.append(img[r][c])

    # vertical pixels
    start_x = X - 1
    end_x = X + 1
    start_y = Y - R
    end_y = Y + R
    start_x = 0 if start_x <= 0 else start_x
    start_y = 0 if start_y <= 0 else start_y
    end_x = end_x if end_x < width else width - 1
    end_y = end_y if end_y < height else height - 1
    for c in range(start_x, end_x + 1):
        for r in range(start_y, end_y + 1):
            if img[r][c]:
                ints.append(img[r][c])
    return (ints, X, Y)


# 指数函数拟合的目标函数
def func_1(x, a, b, c):
    return a * np.exp(b * x) + c


def func_2(x, a, b):
    return x**a + b


def func_3(x, a, b):
    return a * np.power(x, b)


def func_4(x, a, b):
    return a * np.exp(b / x)


def func_5(x, a, b, c):
    return a**x + b * x + c

# 二次曲线拟合


def func_6(x, a, b, c):
    return a * x * x + b * x + c

# 三次曲线拟合


def func_7(x, a, b, c, d):
    return a * x * x * x + b * x * x + c * x + d

# 高斯曲线拟合


def func_gauss(x, a, b, c, sigma):
    return a * np.exp(-(x - b)**2 / (2 * sigma**2)) + c

# test track area background
# choose 50 percent of sorted intensities as background


def testTrackAreaBkg(img_path, coords_path, ratio, the_id=-1):
    print('-- Test track area background')
    src = imread(img_path)
    pts = loadPts(coords_path)
    X = 0
    Y = 0
    the_x = 0
    the_y = 0
    BKGs = np.zeros(len(pts))
    for i, pt in enumerate(pts):
        bkg, X, Y, y_data = calcTrackBkg(src, i, pt, ratio, 25, the_id)
        BKGs[i] = bkg
        print('%d Bkg: %.2f\n' % (i, bkg), flush=True)
        if y_data != None:
            the_x, the_y = X, Y

        # draw specified track area intensities
        if i == the_id:
            # ints_out = np.array(ints) # tunr python list into numpy array
            # np.savetxt('./ints.txt', ints_out)
            print("Intensity's mean: %.2f" % np.array(y_data).mean())
            x_data = np.linspace(1, len(y_data), len(y_data))
            fig = plt.figure(figsize=(10, 6))
            ax_0 = plt.subplot(211)
            plt.title('Track %d [%d, %d] Sorted ints' % (the_id, the_x, the_y))
            plt.xlabel('Int ID')
            plt.ylabel('Intensity')
            ax_0.plot(x_data, y_data, 'bo-', label='Ints')

            # 指数函数拟合? 是否使用局部加权线性拟合？还是多项式拟合？
            popt, pcov = curve_fit(func_7, x_data, y_data)
            print('popt:', popt)
            y_fit = [func_7(x_val, popt[0], popt[1], popt[2], popt[3])
                     for x_val in x_data]
            ax_0.plot(x_data, y_fit, 'r-', label='fit line')
            legend_0 = ax_0.legend(bbox_to_anchor=(
                1, 1), loc=2, borderaxespad=0.)
            legend_0.get_frame().set_facecolor('#00FFCC')
            # plt.show()
    ax_1 = plt.subplot(212)
    plt.xlabel('Track ID')
    plt.ylabel('Intensity')
    ids = np.linspace(1, len(pts), len(pts))
    ax_1.plot(ids, BKGs, 'ko-', label='BKGs')
    legend_1 = ax_1.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    legend_1.get_frame().set_facecolor('#00FFCC')
    plt.show()


# calculate each DNB spot's background by 4 vertices' background
def calcInsideBlockDNBBkg(img, TCs, X, Y, DNB_NUM_X, DNB_NUM_Y):
    # prepare 4 vertices and width, height  for given block
    vertices_x = np.zeros(4)
    vertices_y = np.zeros(4)
    id_0 = Y * 11 + X
    id_1 = id_0 + 1
    id_2 = id_0 + 12
    id_3 = id_0 + 11
    vertices_x[0] = TCs[id_0][0]
    vertices_x[1] = TCs[id_1][0]
    vertices_x[2] = TCs[id_2][0]
    vertices_x[3] = TCs[id_3][0]

    vertices_y[0] = TCs[id_0][1]
    vertices_y[1] = TCs[id_1][1]
    vertices_y[2] = TCs[id_2][1]
    vertices_y[3] = TCs[id_3][1]
    width = (vertices_x[1] - vertices_x[0] +
             vertices_x[2] - vertices_x[3]) * 0.5  # block width
    height = (vertices_y[3] - vertices_y[0] +
              vertices_y[2] - vertices_y[1]) * 0.5  # block height
    print('width,height[%.2f, %.2f]' % (width, height))

    # prepare 4 vertices's background
    vertex_bkgs = np.zeros(4)
    # parameters are in consistent with the v0.1 doc
    vertex_bkgs[0] = computeTrackBkg(img, TCs[id_0], 0.5, 25)
    vertex_bkgs[1] = computeTrackBkg(img, TCs[id_1], 0.5, 25)
    vertex_bkgs[2] = computeTrackBkg(img, TCs[id_2], 0.5, 25)
    vertex_bkgs[3] = computeTrackBkg(img, TCs[id_3], 0.5, 25)

    # test DNB spots' coordinates and background
    blk_bkgs = np.zeros((DNB_NUM_Y - 3, DNB_NUM_X - 3))  # numpy 2d array
    for y in range(2, DNB_NUM_Y - 1):
        left_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[0] \
            + float(y) / DNB_NUM_Y * vertices_x[3]
        left_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[0] \
            + float(y) / DNB_NUM_Y * vertices_y[3]
        right_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[1] \
            + float(y) / DNB_NUM_Y * vertices_x[2]
        right_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[1] \
            + float(y) / DNB_NUM_Y * vertices_y[2]
        for x in range(2, DNB_NUM_X - 1):
            dnb_x = float(DNB_NUM_X - x) / DNB_NUM_X * \
                left_x + float(x) / DNB_NUM_X * right_x
            dnb_y = float(DNB_NUM_X - x) / DNB_NUM_X * \
                left_y + float(x) / DNB_NUM_X * right_y
            u = (dnb_x - (vertices_x[0] + vertices_x[3]) * 0.5) / width * 0.5
            v = (dnb_y - (vertices_y[0] + vertices_y[1]) * 0.5) / height * 0.5
            bkg = (1 - u) * (1 - v) * vertex_bkgs[0] + (1 - u) * v * vertex_bkgs[3] \
                + u * (1 - v) * vertex_bkgs[1] + u * v * vertex_bkgs[2]
            blk_bkgs[y - 2][x - 2] = bkg
    return blk_bkgs


# 给定所有track cross、block ID、DNB_NUM_X、DNB_NUM_Y
# 返回该block的所有DNB坐标
def calcDNBIntsOfBlock(img, TCs, X, Y, dnb_vect_x, dnb_vect_y, mode):
    if X < 0 or X > 9 or Y < 0 or Y > 9 or len(TCs) != 121:
        print('[Error]: parameters wrong.')
        return None

    DNB_NUM_X = dnb_vect_x[X]
    DNB_NUM_Y = dnb_vect_y[Y]

    # prepare 4 vertices and width, height  for given block
    vertices_x = np.zeros(4)
    vertices_y = np.zeros(4)
    id_0 = Y * 11 + X
    id_1 = id_0 + 1
    id_2 = id_0 + 12
    id_3 = id_0 + 11
    vertices_x[0] = TCs[id_0][0]
    vertices_x[1] = TCs[id_1][0]
    vertices_x[2] = TCs[id_2][0]
    vertices_x[3] = TCs[id_3][0]

    vertices_y[0] = TCs[id_0][1]
    vertices_y[1] = TCs[id_1][1]
    vertices_y[2] = TCs[id_2][1]
    vertices_y[3] = TCs[id_3][1]
    width = (vertices_x[1] - vertices_x[0] +
             vertices_x[2] - vertices_x[3]) * 0.5  # block width
    height = (vertices_y[3] - vertices_y[0] +
              vertices_y[2] - vertices_y[1]) * 0.5  # block height
    print('width,height[%.2f, %.2f]' % (width, height))

    # extract each DNB spot's intensity
    if mode == 0:  # 2d numpy array
        blk_ints_2D = np.zeros(
            (DNB_NUM_Y - 3, DNB_NUM_X - 3))  # numpy 2d array
        for y in range(2, DNB_NUM_Y - 1):
            left_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[0] \
                + float(y) / DNB_NUM_Y * vertices_x[3]
            left_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[0] \
                + float(y) / DNB_NUM_Y * vertices_y[3]
            right_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[1] \
                + float(y) / DNB_NUM_Y * vertices_x[2]
            right_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[1] \
                + float(y) / DNB_NUM_Y * vertices_y[2]
            for x in range(2, DNB_NUM_X - 1):
                dnb_x = float(DNB_NUM_X - x) / DNB_NUM_X * \
                    left_x + float(x) / DNB_NUM_X * right_x
                dnb_y = float(DNB_NUM_X - x) / DNB_NUM_X * \
                    left_y + float(x) / DNB_NUM_X * right_y
                THE_X = floor(dnb_x)
                THE_Y = floor(dnb_y)
                if THE_X >= 0 and THE_X + 1 < img.shape[1] \
                        and THE_Y >= 0 and THE_Y + 1 < img.shape[0]:  # 使用双线性插值
                    u = dnb_x - float(THE_X)
                    v = dnb_y - float(THE_Y)
                    blk_ints_2D[y - 2][x - 2] = (1 - u) * (1 - v) * img[THE_Y][THE_X] \
                        + (1 - u) * v * img[THE_Y + 1][THE_X] \
                        + u * (1 - v) * img[THE_Y][THE_X + 1] \
                        + u * v * img[THE_Y + 1][THE_X + 1]
                else:
                    print('[Warning]: DNB out of boundary.')
                    blk_ints_2D[y - 2][x - 2] = 0  # 用0填充图像外面的部分
        return blk_ints_2D
    elif mode == 1:
        blk_ints_1D = np.zeros((DNB_NUM_Y - 3) * (DNB_NUM_X - 3))
        i = 0
        for y in range(2, DNB_NUM_Y - 1):
            left_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[0] \
                + float(y) / DNB_NUM_Y * vertices_x[3]
            left_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[0] \
                + float(y) / DNB_NUM_Y * vertices_y[3]
            right_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[1] \
                + float(y) / DNB_NUM_Y * vertices_x[2]
            right_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[1] \
                + float(y) / DNB_NUM_Y * vertices_y[2]
            for x in range(2, DNB_NUM_X - 1):
                dnb_x = float(DNB_NUM_X - x) / DNB_NUM_X * \
                    left_x + float(x) / DNB_NUM_X * right_x
                dnb_y = float(DNB_NUM_X - x) / DNB_NUM_X * \
                    left_y + float(x) / DNB_NUM_X * right_y
                THE_X = floor(dnb_x)
                THE_Y = floor(dnb_y)
                if THE_X >= 0 and THE_X + 1 < img.shape[1] \
                        and THE_Y >= 0 and THE_Y + 1 < img.shape[0]:  # 使用双线性插值
                    u = dnb_x - float(THE_X)
                    v = dnb_y - float(THE_Y)
                    blk_ints_1D[i] = (1 - u) * (1 - v) * img[THE_Y][THE_X] \
                        + (1 - u) * v * img[THE_Y + 1][THE_X] \
                        + u * (1 - v) * img[THE_Y][THE_X + 1] \
                        + u * v * img[THE_Y + 1][THE_X + 1]
                else:
                    print('[Warning]: DNB out of boundary.')
                    blk_ints_1D[i] = 0  # 用0填充图像外面的部分
                i += 1
        return blk_ints_1D


# calculate pixel intensity of block
def calcPixelIntsOfBlock(img, TCs, X, Y):
    pass


# calculate block's DNB background
# using different strategies according to block ID
def calcBlockDNBBkgSpe(img, TCs, blk_id, R, DNB_NUM_X, DNB_NUM_Y):
    if blk_id < 0 or blk_id > 99 or len(TCs) != 121:
        print('[Error]: parameters wrong.')
        return None

    X = int(blk_id % 10)
    Y = int(blk_id / 10)

    # prepare 4 vertices and width, height  for given block
    vertices_x = np.zeros(4)
    vertices_y = np.zeros(4)
    id_0 = Y * 11 + X
    id_1 = id_0 + 1
    id_2 = id_0 + 12
    id_3 = id_0 + 11
    vertices_x[0] = TCs[id_0][0]
    vertices_x[1] = TCs[id_1][0]
    vertices_x[2] = TCs[id_2][0]
    vertices_x[3] = TCs[id_3][0]

    vertices_y[0] = TCs[id_0][1]
    vertices_y[1] = TCs[id_1][1]
    vertices_y[2] = TCs[id_2][1]
    vertices_y[3] = TCs[id_3][1]
    width = (vertices_x[1] - vertices_x[0] +
             vertices_x[2] - vertices_x[3]) * 0.5  # block width
    height = (vertices_y[3] - vertices_y[0] +
              vertices_y[2] - vertices_y[1]) * 0.5  # block height
    print('width,height[%.2f, %.2f]' % (width, height))

    # prepare 4 vertices's background
    # parameters are in consistent with the v0.1 doc
    vertex_bkgs = np.zeros(4)
    vertex_bkgs[0] = computeTrackAreaBkg(img, TCs[id_0], 0.5, R)
    vertex_bkgs[1] = computeTrackAreaBkg(img, TCs[id_1], 0.5, R)
    vertex_bkgs[2] = computeTrackAreaBkg(img, TCs[id_2], 0.5, R)
    vertex_bkgs[3] = computeTrackAreaBkg(img, TCs[id_3], 0.5, R)

    # test DNB spots' coordinates and background
    blk_bkgs = np.zeros((DNB_NUM_Y - 3, DNB_NUM_X - 3))  # numpy 2d array
    i = 0
    for y in range(2, DNB_NUM_Y - 1):
        left_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[0] \
            + float(y) / DNB_NUM_Y * vertices_x[3]
        left_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[0] \
            + float(y) / DNB_NUM_Y * vertices_y[3]
        right_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[1] \
            + float(y) / DNB_NUM_Y * vertices_x[2]
        right_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[1] \
            + float(y) / DNB_NUM_Y * vertices_y[2]
        for x in range(2, DNB_NUM_X - 1):
            dnb_x = float(DNB_NUM_X - x) / DNB_NUM_X * \
                left_x + float(x) / DNB_NUM_X * right_x
            dnb_y = float(DNB_NUM_X - x) / DNB_NUM_X * \
                left_y + float(x) / DNB_NUM_X * right_y
            if X > 0 and X < 9 and Y > 0 and Y < 9:  # inside
                u = (dnb_x - (vertices_x[0] +
                              vertices_x[3]) * 0.5) / width * 0.5
                v = (dnb_y - (vertices_y[0] +
                              vertices_y[1]) * 0.5) / height * 0.5
                bkg = (1 - u) * (1 - v) * vertex_bkgs[0] + (1 - u) * v * vertex_bkgs[3] \
                    + u * (1 - v) * vertex_bkgs[1] + u * v * vertex_bkgs[2]
            elif Y == 0 and X > 0 and X < 9:  # up border
                dist_3 = sqrt((dnb_x - vertices_x[3]) * (dnb_x - vertices_x[3])
                              + (dnb_y - vertices_y[3]) * (dnb_y - vertices_y[3]))
                dist_2 = sqrt((dnb_x - vertices_x[2]) * (dnb_x - vertices_x[2])
                              + (dnb_y - vertices_y[2]) * (dnb_y - vertices_y[2]))
                ratio = dist_3 / (dist_3 + dist_2)
                bkg = (1 - ratio) * vertex_bkgs[3] + ratio * vertex_bkgs[2]
            elif Y == 9 and X > 0 and X < 9:  # down border
                dist_0 = sqrt((dnb_x - vertices_x[0]) * (dnb_x - vertices_x[0])
                              + (dnb_y - vertices_y[0]) * (dnb_y - vertices_y[0]))
                dist_1 = sqrt((dnb_x - vertices_x[1]) * (dnb_x - vertices_x[1])
                              + (dnb_y - vertices_y[1]) * (dnb_y - vertices_y[1]))
                ratio = dist_0 / (dist_0 + dist_1)
                bkg = (1 - ratio) * vertex_bkgs[0] + ratio * vertex_bkgs[1]
            elif X == 0 and Y > 0 and Y < 9:  # left border
                dist_1 = sqrt((dnb_x - vertices_x[1]) * (dnb_x - vertices_x[1])
                              + (dnb_y - vertices_y[1]) * (dnb_y - vertices_y[1]))
                dist_2 = sqrt((dnb_x - vertices_x[2]) * (dnb_x - vertices_x[2])
                              + (dnb_y - vertices_y[2]) * (dnb_y - vertices_y[2]))
                ratio = dist_1 / (dist_1 + dist_2)
                bkg = (1 - ratio) * vertex_bkgs[1] + ratio * vertex_bkgs[2]
            elif X == 9 and Y > 0 and Y < 9:  # right border
                dist_0 = sqrt((dnb_x - vertices_x[0]) * (dnb_x - vertices_x[0])
                              + (dnb_y - vertices_y[0]) * (dnb_y - vertices_y[0]))
                dist_3 = sqrt((dnb_x - vertices_x[3]) * (dnb_x - vertices_x[3])
                              + (dnb_y - vertices_y[3]) * (dnb_y - vertices_y[3]))
                ratio = dist_0 / (dist_0 + dist_3)
                bkg = (1 - ratio) * vertex_bkgs[0] + ratio * vertex_bkgs[3]
            elif blk_id == 0:  # corner 0
                bkg = vertex_bkgs[2]
            elif blk_id == 9:  # corner 1
                bkg = vertex_bkgs[3]
            elif blk_id == 99:  # corner 2
                bkg = vertex_bkgs[0]
            elif blk_id == 90:  # corner 3
                bkg = vertex_bkgs[1]
            blk_bkgs[y - 2][x - 2] = bkg
            # print('DNB %d Bkg: %.2f' %(i, bkg))
            i += 1
    # storePts(dnbs, './dnbs.txt')
    return blk_bkgs


# extract block DNB spot's mean intensity
def extractBlkIntsMean(img, TCs, blk_id, DNB_NUM_X, DNB_NUM_Y):
    if blk_id < 0 or blk_id > 99 or len(TCs) != 121:
        print('[Error]: parameters wrong.')
        return None

    X = int(blk_id % 10)
    Y = int(blk_id / 10)

    # prepare 4 vertices and width, height  for given block
    vertices_x = np.zeros(4)
    vertices_y = np.zeros(4)
    id_0 = Y * 11 + X
    id_1 = id_0 + 1
    id_2 = id_0 + 12
    id_3 = id_0 + 11
    vertices_x[0] = TCs[id_0][0]
    vertices_x[1] = TCs[id_1][0]
    vertices_x[2] = TCs[id_2][0]
    vertices_x[3] = TCs[id_3][0]

    vertices_y[0] = TCs[id_0][1]
    vertices_y[1] = TCs[id_1][1]
    vertices_y[2] = TCs[id_2][1]
    vertices_y[3] = TCs[id_3][1]
    width = (vertices_x[1] - vertices_x[0] +
             vertices_x[2] - vertices_x[3]) * 0.5  # block width
    height = (vertices_y[3] - vertices_y[0] +
              vertices_y[2] - vertices_y[1]) * 0.5  # block height
    print('width,height[%.2f, %.2f]' % (width, height))

    # extract each DNB spot's intensity
    blk_ints = np.zeros((DNB_NUM_Y - 3, DNB_NUM_X - 3))  # numpy 2d array
    i = 0
    for y in range(2, DNB_NUM_Y - 1):
        left_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[0] \
            + float(y) / DNB_NUM_Y * vertices_x[3]
        left_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[0] \
            + float(y) / DNB_NUM_Y * vertices_y[3]
        right_x = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_x[1] \
            + float(y) / DNB_NUM_Y * vertices_x[2]
        right_y = float(DNB_NUM_Y - y) / DNB_NUM_Y * vertices_y[1] \
            + float(y) / DNB_NUM_Y * vertices_y[2]
        for x in range(2, DNB_NUM_X - 1):
            dnb_x = float(DNB_NUM_X - x) / DNB_NUM_X * \
                left_x + float(x) / DNB_NUM_X * right_x
            dnb_y = float(DNB_NUM_X - x) / DNB_NUM_X * \
                left_y + float(x) / DNB_NUM_X * right_y

            # if dnb_x >= 0 and dnb_x < img.shape[1] \
            # and dnb_y >= 0 and dnb_y < img.shape[0]: # 使用最近邻法
            # the_x = int(dnb_x)
            # the_y = int(dnb_y)
            # blk_ints[y-2][x-2] = img[the_y][the_x]

            THE_X = floor(dnb_x)
            THE_Y = floor(dnb_y)
            if THE_X >= 0 and THE_X + 1 < img.shape[1] \
                    and THE_Y >= 0 and THE_Y + 1 < img.shape[0]:  # 使用双线性插值
                u = dnb_x - float(THE_X)
                v = dnb_y - float(THE_Y)
                blk_ints[y - 2][x - 2] = (1 - u) * (1 - v) * img[THE_Y][THE_X] \
                    + (1 - u) * v * img[THE_Y + 1][THE_X] \
                    + u * (1 - v) * img[THE_Y][THE_X + 1] \
                    + u * v * img[THE_Y + 1][THE_X + 1]
            else:
                print('[Warning]: DNB out of boundary.')
                blk_ints[y - 2][x - 2] = 0  # 用0填充图像外面的部分
    return np.mean(blk_ints)


# test DNBs in a given block
def testDNBsOfBlock(img_path, pts_path, blk_id, DNB_NUM_X, DNB_NUM_Y):
    print('-- Test DNBs in a given block')

    # load img and track cross
    src = imread(img_path)
    pts = loadPts(pts_path)

    # compute each DNB spot's background
    X = int(blk_id % 10)
    Y = int(blk_id / 10)
    blk_bkgs = calcInsideBlockDNBBkg(src, pts, X, Y, DNB_NUM_X, DNB_NUM_Y)

    # plot heatmap of block background
    plt.imshow(blk_bkgs)  # 绘制热图
    plt.colorbar()  # 绘制色标
    plt.show()


# format DNB vector from input grids
def formatDNBVector(grids_x, grids_y):
    dnb_vect_x = grids_x[:8]
    dnb_vect_y = grids_y[:8]
    dnb_vect_x.insert(0, (grids_x[8] >> 1) + 2)
    dnb_vect_y.insert(0, (grids_y[8] >> 1) + 2)
    dnb_vect_x.append((grids_x[8] >> 1) + 1)
    dnb_vect_y.append((grids_y[8] >> 1) + 1)
    return dnb_vect_x, dnb_vect_y


# test each block's DNB background heatmap
def testImgDNBBkgs(img_path, pts_path, grids_x, grids_y):
    print("-- Test Img block DNBs' background")

    # load img and track cross
    src = imread(img_path)
    pts = loadPts(pts_path)

    # prepare dnb vector of template
    dnb_vect_x, dnb_vect_y = formatDNBVector(grids_x, grids_y)

    # process each block
    for blk_id in range(100):
        print('\n-- processing block %d' % blk_id)
        X = int(blk_id % 10)  # block x
        Y = int(blk_id / 10)  # block y
        blk_bkgs = calcInsideBlockDNBBkg(
            src, pts, X, Y, dnb_vect_x[X], dnb_vect_y[Y])

        # plot heatmap of block background
        plt.title('Block %d background' % blk_id)  # 绘制标题
        plt.imshow(blk_bkgs, cmap=None)  # 绘制热图
        plt.colorbar()  # 绘制色标
        plt.show()


# calculate the whole DNB number according to input template
def calcTotalDNBNum(grids_x, grids_y):
    total = 0
    total_y = 0
    for y in grids_y:
        total_y += y - 3
        if grids_y.index(y) == 0:
            total_x = 0
        for x in grids_x:
            if grids_y.index(y) == 0:
                total_x += x - 3
            total += (y - 3) * (x - 3)
    print('Total DNB num X: %d\nTotal DNB num Y: %d\nTotal DNB num: %d' %
          (total_x, total_y, total))
    return (total_y, total_x, total)


# test all block's DNB background heat map
# R取得越大，相当于低通滤波作用最强, R取值变小则相当于保留更多高频细节
def testAllBlockDNBBkgs(img_path, pts_path, R, grids_x, grids_y):
    print('-- Test the whole image DNB background')

    # load img and track cross
    src = imread(img_path)
    pts = loadPts(pts_path)

    # preprocess image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # eroded = cv2.erode(src, kernel)
    # dst = src - eroded

    # prepare dnb vector of template
    dnb_vect_x, dnb_vect_y = formatDNBVector(grids_x, grids_y)

    # preallocate memory for whole image DNB background
    total_y, total_x, total = calcTotalDNBNum(grids_x, grids_y)
    dnb_bkgs = np.zeros((total_y, total_x))

    # process each block
    for blk_id in range(100):
        print('\n-- processing block %d' % blk_id)
        X = int(blk_id % 10)  # block x
        Y = int(blk_id / 10)  # block y
        # blk_bkgs = calcInsideBlockDNBBkg(src, pts, X, Y, dnb_vect_x[X], dnb_vect_y[Y])
        # using different strategy for different block type
        blk_bkgs = calcBlockDNBBkgSpe(
            src, pts, blk_id, R, dnb_vect_x[X], dnb_vect_y[Y])

        row = 0
        col = 0
        for r in range(Y):
            row += dnb_vect_y[r] - 3
        for c in range(X):
            col += dnb_vect_x[c] - 3
        blk_height = blk_bkgs.shape[0]
        blk_width = blk_bkgs.shape[1]
        for y in range(blk_height):
            for x in range(blk_width):
                index_y = row + y
                index_x = col + x
                dnb_bkgs[index_y][index_x] = blk_bkgs[y][x]

    # plot whole image DNB background heatmap
    plt.title('Background heatmap')  # 绘制标题
    plt.imshow(dnb_bkgs, cmap=None)  # 绘制热图: plt.cm.gray...
    plt.colorbar()  # 绘制色标
    plt.show()
    return dnb_bkgs


# draw single background heatmap
def drawBKGHeatMap(f_path):
    '''
    加载已有数据绘制热图
    '''
    bkg = np.loadtxt(f_path)
    m, n = bkg.shape
    print('row_num: {}, col_num: {}\n'.format(m, n))

    plt.title('Background heatmap')  # 绘制标题
    plt.imshow(bkg, cmap=None)  # 绘制热图: plt.cm.gray...
    plt.colorbar()  # 绘制色标
    plt.show()


import re
def drawBkgHeatmaps(dir_path):
    '''
    draw all bkg heatmap of given dir
    '''
    if not os.path.exists(dir_path):
        print('[Error]: dir not exists.')
        return

    bkgs = []
    f_list = os.listdir(dir_path)
    for f in f_list:
        f_name = os.path.splitext(f)  # 切分名称和后缀
        if f_name[1] == '.txt' and '_bkg' in f_name[0]:  # 加载背景文件
            bkg = np.loadtxt(os.path.join(dir_path, f))
            id = int(re.findall(r'\d+', f_name[0])[0])
            bkgs.append((id, bkg))
            print("--{} loaded.".format(f))

    # 绘制heatmap 2*2
    print('--all bkg file loaded, start drawing...')

    for i, _ in enumerate(bkgs):
        if i != 0 and i % 4 == 0:
            fig = plt.figure(figsize=(10, 8))
            ax_0 = plt.subplot(221)
            plt.title('{} background'.format(bkgs[i - 4][0]))
            plt.imshow(bkgs[i - 4][1])
            plt.colorbar()

            ax_1 = plt.subplot(222)
            plt.title('{} background'.format(bkgs[i - 3][0]))
            plt.imshow(bkgs[i - 3][1])
            plt.colorbar()

            ax_2 = plt.subplot(223)
            plt.title('{} backgound'.format(bkgs[i - 2][0]))
            plt.imshow(bkgs[i - 2][1])
            plt.colorbar()

            ax_3 = plt.subplot(224)
            plt.title('{} backgound'.format(bkgs[i - 1][0]))
            plt.imshow(bkgs[i - 1][1])
            plt.colorbar()

            plt.tight_layout()
            plt.show()
    
    remain_bkgs = bkgs[-int(len(bkgs) % 4):]
    fig = plt.figure(figsize=(10, 8))
    for i, _ in enumerate(remain_bkgs):
        if i < 1:
            ax_0 =  plt.subplot(221)
            plt.title('{} background'.format(remain_bkgs[i][0]))
            plt.imshow(remain_bkgs[i][1])
            plt.colorbar()
        elif i < 2 and i >= 1:
            ax_1 = plt.subplot(222)
            plt.title('{} background'.format(remain_bkgs[i][0]))
            plt.imshow(remain_bkgs[i][1])
            plt.colorbar()
        elif i < 3 and i >= 2:
            ax_2 = plt.subplot(223)
            plt.title('{} background'.format(remain_bkgs[i][0]))
            plt.imshow(remain_bkgs[i][1])
            plt.colorbar()
    plt.tight_layout()
    plt.show()



# plot wavelets
def plotWavelets(data, w, title, mode):
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(3):  # 5 层小波分解
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
    plt.figure(figsize=(12, 9.5))
    ax_main = plt.subplot(len(rec_a) + 1, 1, 1)
    plt.title(title)
    ax_main.plot(data)

    plt.xlim(0, len(data) - 1)
    for i, y in enumerate(rec_a):
        ax = plt.subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        plt.xlim(0, len(y) - 1)
        plt.ylabel('A %d' % (i + 1))
    for i, y in enumerate(rec_d):
        ax = plt.subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        plt.xlim(0, len(y) - 1)
        plt.ylabel('D %d' % (i + 1))
    plt.show()


# test wavelets...
# http://forum.vibunion.com/forum.php?mod=viewthread&tid=124065&extra=page%3D1&page=1
# http://blog.csdn.net/ebowtang/article/details/40481539
def testWavelets_1D(img_path, pts_path, grids_x, grids_y, blk_id):
    print('-- Test wavelets 1D')

    # load img and track cross
    src = imread(img_path)
    pts = loadPts(pts_path)

    # prepare dnb vector of template
    dnb_vect_x, dnb_vect_y = formatDNBVector(grids_x, grids_y)

    # prepare DNBs of given block
    X = int(blk_id % 10)
    Y = int(blk_id / 10)
    blk_ints = calcDNBIntsOfBlock(src, pts, X, Y, dnb_vect_x, dnb_vect_y, 1)
    # plotWavelets(blk_ints, 'db3', 'Src data', 'sym')

    # wavelet decomposition
    noisy_coefs = pywt.wavedec(blk_ints, 'db8', level=9, mode='per')

    # contruct threshold
    sigma = stand_mad(noisy_coefs[-1])  # ?
    uthresh = sigma * np.sqrt(2.0 * np.log(len(blk_ints)))

    # compute denoised coefficients by threshold
    denoised_coefs = noisy_coefs[:]
    denoised_coefs[1:] = (pywt._thresholding.soft(
        data, value=uthresh) for data in denoised_coefs[1:])
    rec_signal = pywt.waverec(denoised_coefs, 'db8', mode='per')

    # plot result
    fig = plt.figure(figsize=(8, 8))
    ax_0 = plt.subplot(211)
    ax_0.plot(blk_ints, 'b-')
    plt.title('Src signal')
    ax_1 = plt.subplot(212)
    ax_1.plot(rec_signal, 'r-')
    plt.title('Denoised signal')
    plt.show()


# test denosing by wavelet thresholding reconstruction
def testWavelets_2D(img_path):
    print('-- Test wavelets 2D')

    # load image original signal
    src = imread(img_path)

    # waveles analysis plot
    # coeffs2 = pywt.dwt2(src, 'bior1.3')
    # LL, (LH, HL, HH) = coeffs2
    # fig = plt.figure(figsize=(8,8))
    # titles = ['Approximation', ' Horizontal detail',
    #       'Vertical detail', 'Diagonal detail']
    # for i, a in enumerate([LL, LH, HL, HH]):
    #     ax = fig.add_subplot(2, 2, i + 1)
    #     ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    #     ax.set_title(titles[i], fontsize=12)
    # plt.show()

    # 将2D信号处理成1D信号
    signal_1D = np.zeros(src.shape[0] * src.shape[1])
    i = 0
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            signal_1D[i] = src[y][x]
            i += 1
    print('Total %d points' % i)

    # wavelet decomposition
    noisy_coefs = pywt.wavedec(signal_1D, 'db8', level=9, mode='per')

    # contruct threshold
    sigma = stand_mad(noisy_coefs[-1])  # ?
    uthresh = sigma * np.sqrt(2.0 * np.log(len(signal_1D)))

    # compute denoised coefficients by threshold
    denoised_coefs = noisy_coefs[:]
    denoised_coefs[1:] = (pywt._thresholding.soft(
        data, value=uthresh) for data in denoised_coefs[1:])
    rec_signal = pywt.waverec(denoised_coefs, 'db8', mode='per')

    # 将1D信号转化为2D信号并保存
    dst = np.zeros(src.shape)
    for i, intensity in enumerate(signal_1D):
        x = int(i % src.shape[1])
        y = int(i / src.shape[1])
        dst[y][x] = rec_signal[i]
    imsave('./denoised_soft.tif', dst.astype('uint16'))

    # 求noise图
    noise_bkg = np.zeros(src.shape)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            bkg = src[y][x] - dst[y][x]
            noise_bkg[y][x] = bkg if bkg >= 0 else 0
            print('noise bkg = %d' % noise_bkg[y][x])
    imsave('./noise_soft.tif', noise_bkg.astype('uint16'))

    # plot result
    fig = plt.figure(figsize=(8, 8))
    ax_0 = plt.subplot(211)
    ax_0.plot(signal_1D, 'b-')
    plt.title('Src signal')
    ax_1 = plt.subplot(212)
    ax_1.plot(rec_signal, 'r-')
    plt.title('Denoised signal')
    plt.show()


# 调用TrackDetector.exe处理制定目录下的tif文件
def detectTCAndCropImg(img_path, R=25):
    LEN = (R << 1) + 1  # sample length

    # determine whether to call TrackDetector.exe
    f_list = os.listdir(img_path)
    tif_count = 0
    txt_count = 0
    for f in f_list:
        f_name = os.path.splitext(f)
        if f_name[1] == '.tif':
            tif_count += 1
        elif f_name[1] == '.txt':
            txt_count += 1
    if tif_count != txt_count:  # if not equal, call detector first
        print('-- detect track cross...')
        import subprocess
        cmd = './TrackDetector.exe ' + img_path + ' ./settings.config'
        print('CMD: ', cmd)
        sub_p = subprocess.Popen(cmd)  # call only once
        sub_p.wait()  # wait the sub_process run to end

    # traverse each tif and its corresponding txt file
    for i, f in enumerate(f_list):
        f_name = os.path.splitext(f)
        if f_name[1] == '.tif':
            print('\n-- processing file %s' % f[0])
            src = imread(img_path + f_name[0] + f_name[1])

            # create dir of each image
            if not os.path.exists(img_path + f_name[0]):
                os.makedirs(img_path + f_name[0])

            # load points
            pts = loadPts(img_path + f_name[0] + '_tc.txt')

            # create each track cross samle file
            for id, tc in enumerate(pts):
                id_x = int(id % 11)
                id_y = int(id / 11)
                if id_x > 0 and id_x < 10 and id_y > 0 and id_y < 10:
                    inner_id = (id_y - 1) * 9 + (id_x - 1)

                    # fill the sample
                    sample = np.zeros((LEN, LEN))
                    X = int(round(tc[0]))
                    Y = int(round(tc[1]))
                    for y in range(LEN):
                        for x in range(LEN):
                            sample[y][x] = src[Y - R + y][X - R + x]

                    # output the sample
                    sample_name = img_path + f_name[0] + '/' + str(inner_id) \
                        + str((round(tc[0] - float(X - R), 5), round(tc[1] - float(Y - R), 5))) \
                        + '.tif'
                    if not os.path.exists(sample_name):
                        imsave(sample_name, sample.astype(dtype='uint16'))
                        print('-- %s sample processed.' % sample_name)
    print('-- Sampling done.')


# 绘制原图的热图
def drawImgHeatMap(img_path):
    print('-- Draw src image heatmap')

    # load img
    src = imread(img_path)

    # preprocess src
    int_mean = np.mean(src)
    print('Mean intensity: %.2f' % int_mean)
    dst = np.where(src < 6.0 * int_mean, src, int_mean)  # 条件过滤

    # draw heatmap
    plt.title('Dst heatmap')
    plt.imshow(dst, cmap=None)  # 绘制热图: plt.cm.gray...
    plt.colorbar()  # 绘制色标
    plt.show()


# 绘制原图block均值的热图: 此热图可以反映原图的intensity大致分布规律
def drawBlockMeanHeatmap(img_path, pts_path, grids_x, grids_y):
    print('-- Draw block mean intensity heatmap')

    # load img
    src = imread(img_path)
    pts = loadPts(pts_path)

    # prepare dnb vector of template
    dnb_vect_x, dnb_vect_y = formatDNBVector(grids_x, grids_y)

    # process each block
    blk_int_means = np.zeros((10, 10))
    for blk_id in range(100):
        print('\n-- processing block %d' % blk_id)
        X = int(blk_id % 10)  # block x
        Y = int(blk_id / 10)  # block y

        val = np.mean(calcDNBIntsOfBlock(
            src, pts, X, Y, dnb_vect_x, dnb_vect_y, 0))
        # val = extractBlkIntsMean(src, pts, blk_id, dnb_vect_x[X], dnb_vect_y[Y])
        print("block %d's DNB intensity mean: %.2f" % (blk_id, val))
        blk_int_means[Y][X] = val

    # draw block intensity mean's heatmap
    plt.title("Block DNB intensity's mean")
    plt.imshow(blk_int_means, cmap=None)  # 绘制热图: plt.cm.gray...
    plt.colorbar()  # 绘制色标
    plt.show()


# 判断某个Track区域的点是否是DNB spot point
def judgeDNBPoint(img, pt, TH=9):
    '''
    presume: pt and its surrounding pts(whithin radius 1)
    are within range of img
    '''
    x = pt[0]
    y = pt[1]
    if x - 1 < 0 or x + 1 >= img.shape[1] \
            or y - 1 < 0 or y + 1 >= img.shape[0]:
        return False

    ints_1 = np.zeros(4)
    ints_1[0] = img[y - 1][x]
    ints_1[1] = img[y + 1][x]
    ints_1[2] = img[y][x - 1]
    ints_1[3] = img[y][x + 1]
    if img[y][x] < ints_1[0] or img[y][x] < ints_1[1] \
            or img[y][x] < ints_1[2] or img[y][x] < ints_1[3]:
        return False

    count = 0
    ints_sqrt2 = np.zeros(4)
    ints_sqrt2[0] = img[y - 1][x - 1]
    ints_sqrt2[1] = img[y - 1][x + 1]
    ints_sqrt2[2] = img[y + 1][x - 1]
    ints_sqrt2[3] = img[y + 1][x + 1]
    for int_1 in ints_1:
        for int_sqrt2 in ints_sqrt2:
            if int_1 < int_sqrt2:
                count += 1
    if count > TH:
        return False
    return True


# 研究更准确的DNB背景提取算法：将Track区域分类
# 只从Track区域背景区域提取背景
def computeTrackAreaBkg(img, pt, ratio, R):
    X = int(round(pt[0]))
    Y = int(round(pt[1]))
    width = img.shape[1]
    height = img.shape[0]
    X = X if X >= 0 else 0
    X = X if X < width else width - 1
    Y = Y if Y >= 0 else 0
    Y = Y if Y < height else height - 1
    print('ref point[%d, %d] ' % (X, Y))

    # prepare all track area points
    pts = []
    start_x = X - R
    end_x = X + R
    start_y = Y - 1
    end_y = Y + 1
    start_x = 0 if start_x <= 0 else start_x
    start_y = 0 if start_y <= 0 else start_y
    end_x = end_x if end_x < width else width - 1
    end_y = end_y if end_y < height else height - 1
    for r in range(start_y, end_y + 1):
        for c in range(start_x, end_x + 1):
            pts.append((c, r))
    start_x = X - 1
    end_x = X + 1
    start_y = Y - R
    end_y = Y + R
    start_x = 0 if start_x <= 0 else start_x
    start_y = 0 if start_y <= 0 else start_y
    end_x = end_x if end_x < width else width - 1
    end_y = end_y if end_y < height else height - 1
    for c in range(start_x, end_x + 1):
        for r in range(start_y, end_y + 1):
            pts.append((c, r))

    # filter track area points for normal points
    dnbs = []
    for pt in pts:
        if judgeDNBPoint(img, pt):
            x = pt[0]
            y = pt[1]
            for i in range(3):
                for j in range(3):
                    dnb_pt = (x - 1 + j, y - 1 + i)
                    dnbs.append(dnb_pt)
    normal_pts = set(pts) - set(dnbs)  # 求差集->normal points

    # sort and draw normal ints' distribution
    normal_ints = [img[y][x] for (x, y) in normal_pts]

    # fig = plt.figure(figsize=(9.5, 7.8))
    # ax_1 = plt.subplot(212)
    # plt.xlabel('Intensity')
    # plt.ylabel('Num')
    # plt.hist(normal_ints, 15, histtype='bar',facecolor='yellowgreen',alpha=0.75)

    normal_ints.sort()

    # x_data = np.linspace(1, len(normal_ints), len(normal_ints))
    # ax_0 = plt.subplot(211)
    # plt.title('Pt(%d, %d)sorted intensity and hist' %(X, Y))
    # plt.xlabel('Point ID')
    # plt.ylabel('Intensity')
    # plt.plot(x_data, normal_ints, 'go-', label='Ints')
    # legend_0 = ax_0.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    # legend_0.get_frame().set_facecolor('#00FFCC')
    # plt.show()

    # storePts(normal_pts, './normal.txt')
    # storePts(dnbs, './dnbs.txt')

    return normal_ints[int(round(ratio * len(normal_ints)))]


# 作为对比，从全部的Track区域提取背景
def calcTrackAreaBkg(img, pt, ratio, R):
    X = int(round(pt[0]))
    Y = int(round(pt[1]))
    width = img.shape[1]
    height = img.shape[0]
    X = X if X >= 0 else 0
    X = X if X < width else width - 1
    Y = Y if Y >= 0 else 0
    Y = Y if Y < height else height - 1
    print('ref point[%d, %d] ' % (X, Y))

    # prepare all track area points
    pts = []
    start_x = X - R
    end_x = X + R
    start_y = Y - 1
    end_y = Y + 1
    start_x = 0 if start_x <= 0 else start_x
    start_y = 0 if start_y <= 0 else start_y
    end_x = end_x if end_x < width else width - 1
    end_y = end_y if end_y < height else height - 1
    for r in range(start_y, end_y + 1):
        for c in range(start_x, end_x + 1):
            pts.append((c, r))
    start_x = X - 1
    end_x = X + 1
    start_y = Y - R
    end_y = Y + R
    start_x = 0 if start_x <= 0 else start_x
    start_y = 0 if start_y <= 0 else start_y
    end_x = end_x if end_x < width else width - 1
    end_y = end_y if end_y < height else height - 1
    for c in range(start_x, end_x + 1):
        for r in range(start_y, end_y + 1):
            pts.append((c, r))

    # sort and draw normal ints' distribution
    ints = [img[y][x] for (x, y) in pts]

    fig = plt.figure(figsize=(9.5, 7.8))
    ax_1 = plt.subplot(212)
    plt.xlabel('Intensity')
    plt.ylabel('Num')
    plt.hist(ints, 15, histtype='bar', facecolor='yellowgreen', alpha=0.75)

    ints.sort()

    x_data = np.linspace(1, len(ints), len(ints))
    ax_0 = plt.subplot(211)
    plt.title('Pt(%d, %d)sorted intensity and hist' % (X, Y))
    plt.xlabel('Point ID')
    plt.ylabel('Intensity')
    plt.plot(x_data, ints, 'go-', label='Ints')
    legend_0 = ax_0.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    legend_0.get_frame().set_facecolor('#00FFCC')
    plt.show()

    # storePts(dnbs, './dnbs.txt')

    return ints[int(round(ratio * len(ints)))]


def testTrackDNBs(img_path, pts_path, pt_id, R, is_clasify):
    print('-- Test track area DNB points')

    # load img and track cross
    src = imread(img_path)
    pts = loadPts(pts_path)

    # test
    bkg = 0
    if is_clasify:
        bkg = computeTrackAreaBkg(src, pts[pt_id], 0.5, R)
    else:
        bkg = calcTrackAreaBkg(src, pts[pt_id], 0.5, R)
    print('Bkg: %.2f' % bkg)


def testPlotInts(path_1, path_2, name_1, name_2):
    ints_1 = np.loadtxt(path_1)
    ints_2 = np.loadtxt(path_2)
    if len(ints_1) != len(ints_2):
        print('[Error]: input parameters are wrong.\n')
        return
    LEN = len(ints_1)
    x = np.linspace(1, LEN, LEN)
    plt.subplot(121)
    plt.plot(x, ints_1, 'b-', label='ints_1', alpha=0.3)
    plt.title(name_1)
    plt.subplot(122)
    plt.plot(x, ints_2, 'r-', label='ints_2', alpha=0.3)
    plt.title(name_2)
    plt.tight_layout()
    plt.show()

# 接下来测试图像处理结合DNB背景提取


# template parameters
# V0.1
# grids_x = [76, 96, 110, 120, 120, 110, 96, 76, 176]
# grids_y = [76, 96, 110, 120, 120, 110, 96, 76, 176]

# V2 standard pitch
grids_x = [70, 112, 168, 196, 196, 168, 112, 70, 84]
grids_y = [48, 64, 128, 176, 176, 128, 64, 48, 160]

# v2 pitch 800
# grids_x = [68, 102, 119, 136, 136, 119, 102, 68, 306]
# grids_y = [68, 102, 119, 136, 136, 119, 102, 68, 306]


if __name__ == '__main__':
    # testIdealSpots()
    # testNoisedSpots()
    # testRealSpot('c:/test_2_2/0.tif')
    # testBilinearInterp('c:/test_2_2/0.tif', (1379, 966))
    # compLocateMethods(80000.0, 11000.0, 3500.0)
    # doStatistics(80000.0, 11000.0, 3500.0, 63)
    # testTrackAreaBkg('e:/test_2_2/4.tif', 'e:/test_2_2/4_tc.txt', 0.5, 26)
    # testDNBsOfBlock('e:/test_2_2/4.tif', 'e:/test_2_2/4_tc.txt', 11, 76, 76)
    # testImgDNBBkgs('e:/test_2_2/4.tif', 'e:/test_2_2/4_tc.txt', grids_x, grids_y)
    # testAllBlockDNBBkgs('e:/test_v2/1.tif',
    #                     'e:/test_v2/1_tc.txt', 120, grids_x, grids_y)
    # drawBlockMeanHeatmap('e:/test0/1.tif', 'e:/test0/1_tc.txt', grids_x, grids_y)
    # testTrackDNBs('e:/vincent_test/l.tif', 'e:/vincent_test/l_tc.txt', 51, 90, False)
    # testWavelets_1D('e:/test_2_2/8.tif', 'e:/test_2_2/8_tc.txt', grids_x, grids_y, 25)
    # testWavelets_2D('e:/test_2_2/0.tif')
    # detectTCAndCropImg('e:/test_2_2/', 25)
    # testPlotInts('f:/ints_x.txt', 'f:/ints_y.txt', 'X direction', 'Y direction')
    # drawBKGHeatMap('e:/test_v2/11_bkg.txt')
    drawBkgHeatmaps('e:/test_v2/')
    print('-- Test done.\n')


# http://blog.csdn.net/bluebelfast/article/details/17999783
# python后端
# http://blog.csdn.net/ayocross/article/details/56509840?utm_source=itdadao&utm_medium=referraladao&utm_medium=referral
# http://imagej.net/Jython_Scripting
