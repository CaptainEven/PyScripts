# coding:utf-8
import numpy as np
import scipy.linalg as linA  # 为了激活线性代数库mkl
from PIL import Image
import os
import glob


def sim_distance(train, test):
    '''
    计算欧氏距离相似度
    :param train: 二维训练集
    :param test: 一维测试集
    :return: 该测试集到每一个训练集的欧氏距离
    '''
    return [np.linalg.norm(i - test) for i in train]


# picture_path = os.getcwd() + '\\pictures\\'
picture_path = os.path.split(os.path.realpath(__file__))[0] + os.path.sep
array_list = []
img_names = glob.glob(picture_path + '*.jpg')
for name in img_names:
    # 读取每张图片并生成灰度（0-255）的一维序列 1*160000
    img = Image.open(name)

    # img_binary = img.convert('1') 二值化
    img_grey = img.convert('L')  # 灰度化
    # img_grey.show()
    dim = img_grey.size[0] * img_grey.size[1]
    array_list.append(np.array(img_grey).reshape((1, dim)))

# 高维数据下的特征值分解
mat = np.vstack((array_list))  # 将上述多个一维序列合并成矩阵 m*160000
P = np.dot(mat, mat.transpose())  # 计算P
print('P:\n', P)

v, d = np.linalg.eig(P)  # 计算P的特征值和特征向量
print('eig vals: ', v)
print('eig vectors:\n', d)

d = np.dot(mat.transpose(), d)  # 计算Sigma的特征向量 160000 * 8
print('eigenv_vector.shape:\n', d.shape)
d = d[:, :3]  # 取3维主成分

train = np.dot(mat, d)  # 计算训练集的主成分值 8*3
print('train:\n', train)  # 8个数据, 3维特征

# 开始测试
test_pic = np.array(Image.open('dog_4.jpg').convert('L')).reshape((1, 160000))
pic_compr = np.dot(test_pic, d)
pic = pic_compr.transpose()
print('pic_comp:\n', pic_compr)

result = sim_distance(train, pic_compr)
print('result: ', result)
print('match result: %d' % result.index(min(result)))

test_pic = np.array(Image.open('dog_5.jpg').convert('L')).reshape((1, 160000))
pic_compr = np.dot(test_pic, d)
pic_compr.reshape(-1, 1)  # 列向量
print('pic_comp:\n', pic_compr)

result = sim_distance(train, np.dot(test_pic, d))
print('result: ', result)
print('match result: %d' % result.index(min(result)))


# https://www.cnblogs.com/yangruiGB2312/p/5914684.html
