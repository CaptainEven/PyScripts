

from collections import defaultdict
import numpy as np
from math import log

'''
1. 朴素贝叶斯的两个假设前提:
(1). 各个特征之间相互独立
(2). 各个特征的重要性相同
2. token包括单词和标点、字符串、数字组：token是一个语义单元？
因为标点符号包含的语义信息较少,故应该去除标点符号先
'''


def loadDataSet():
    posting_list = [['my', 'dog', 'has', 'flea',
                     'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him',
                     'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute',
                     'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                     'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0, 1, 0, 1, 0, 1]
    return posting_list, labels


def createVocabList(dataset):
    vocab_list = set([])
    for doc in dataset:
        vocab_list = vocab_list | set(doc)  # 求两个集合的并集
    return list(vocab_list)


# 词集模型
def words2Vect(vocab, input):
    ret_vect = [0] * len(vocab)
    for word in input:
        if word in vocab:
            ret_vect[vocab.index(word)] = 1
        else:
            print('the word {} is not in the vocabulary.\n'.format(word))
    return np.array(ret_vect)


# 词袋模型
def bagOfWords2Vect(vocab, input):
    ret_vect = [0] * len(vocab)
    for word in input:
        if word in vocab:
            ret_vect[vocab.index(word)] += 1
    return np.array(ret_vect)


# 需要格外关注的是数据处理技巧
def trainNB0(dataset, labels):
    doc_num = len(dataset)
    word_num = len(dataset[0])
    prob_abu = float(sum(labels)) / float(doc_num)

    # 初始化计数
    num_0 = np.ones(word_num)  # 为防止,p(wi|Cj)相称的过程中一个0将整个概率变为0, 该token至少出现一次
    num_1 = np.ones(word_num)

    # 初始化分母
    denom_0 = 1.0  # 分母即该类所有token出现的总次数
    denom_1 = 1.0

    # 遍历统计数据集
    for i in range(doc_num):
        # 2分类问题
        if labels[i] == 1:
            num_1 += dataset[i]
            denom_1 += sum(dataset[i])
        else:
            num_0 += dataset[i]
            denom_0 += sum(dataset[i])
    # 许多小数相乘,为避免下溢,可取log
    return np.log(num_0 / denom_0), np.log(num_1 / denom_1), prob_abu


# 通过向量点乘判断目标与哪个类别更接近: 两个向量越“相似”，它们的点乘越大
def classifyNB(word_vect, p_v_0, p_v_1, p_abu):
    p_1 = sum(word_vect * p_v_1) + log(p_abu)
    p_0 = sum(word_vect * p_v_0) + log(1.0 - p_abu)
    if p_1 > p_0:
        return 1
    else:
        return 0


def testMyBayes():
    # 加载数据和标签
    dataset, labels = loadDataSet()

    # 根据数据创建词汇表
    vocab = createVocabList(dataset)
    print('vocabulary(len:{}):\n'.format(len(vocab)), vocab)

    # vect_0 = words2Vect(vocab, dataset[0])
    # vect_1 = words2Vect(vocab, dataset[1])
    # print('vect_0:\n', vect_0)
    # print('vect_1:\n', vect_1)

    # 创建训练数据集并格式化成0, 1向量
    train_set = []
    for doc in dataset:
        train_set.append(words2Vect(vocab, doc))
    # print(train_set)

    # 训练先验概率和条件概率
    p_v_0, p_v_1, p_abu = trainNB0(np.array(train_set), np.array(labels))
    print('p_v_0:\n', p_v_0, '\np_v_1:\n', p_v_1, '\np_abu:\n', p_abu)

    # 测试分类
    test_0 = ['love', 'my', 'dalmation']  # dalmation: 斑点狗
    test_1 = ['stupid', 'garbage']
    vect_0 = words2Vect(vocab, test_0)
    vect_1 = words2Vect(vocab, test_1)
    label_0 = classifyNB(vect_0, p_v_0, p_v_1, p_abu)
    label_1 = classifyNB(vect_1, p_v_0, p_v_1, p_abu)
    print('{} classified as: {}.'.format(test_0, label_0))
    print('{} classified as: {}.'.format(test_1, label_1))


#--------------------词袋模型
import re
import os
import random

def parseText(text):
    '''
    分割输入文本,返回token数组
    '''
    # 用非单词和数字的字符作为分隔符
    toks = re.split(r'\W*', text) 

    # 去除空字符串和字符太少的字符串
    return [tok.lower() for tok in toks if len(tok) > 2]


def testBOW():
    '''
    测试词袋模型
    '''
    # 初始化目录路径
    DIR_PATH = os.path.join(os.getcwd() + os.path.sep +
                            'tests' + os.path.sep + 'datasets' + os.path.sep)
    # print('DIR_PATH: ', DIR_PATH)
    email_path = os.path.join(DIR_PATH + 'email' + os.path.sep)
    # print('email path: ', email_path)
    ham_path = os.path.join(email_path + 'ham' + os.path.sep)
    print('ham path: ', ham_path)
    spam_path = os.path.join(email_path + 'spam' + os.path.sep)
    # print('spam path: ', spam_path)

    # 测试文本解析
    # test_f_path = os.path.join(ham_path, '6.txt')
    # with open(test_f_path, 'r') as f:
    #     text = f.read()
    #     # print('text:\n', text)
    #     toks = parseText(text)
    # print(toks)

    # 加载数据, 测试spams, hams
    doc_list = []
    label_list = []
    full_text = [] # 这个full_text有什么用?
    for i in range(1, 26):
        # 读取spam目录数据
        word_list = parseText(open(os.path.join(spam_path, '{}.txt'.format(i))).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        label_list.append(1)

        # 读取ham目录数据
        word_list = parseText(open(os.path.join(ham_path, '{}.txt'.format(i))).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        label_list.append(0)
    # print('doc_list:\n', doc_list)
    print('label_list:\n', label_list)
    vocab = createVocabList(doc_list)
    # print('vocabulary:\n', vocab)

    # 随机抽样10个数据作为测试数据集
    train_idx = list(range(50)) # 总共50个数据: 40个训练, 10个测试
    test_set = []
    for i in range(10):
        rand_idx = int(random.uniform(0, len(train_idx)))
        test_set.append(train_idx[rand_idx])
        del(train_idx[rand_idx]) # 不放回随机抽样
    # print(len(train_idx)) # 剩下40个

    # 准备训练数据集
    train_set = []
    train_labels = []
    for doc_id in train_idx:
        train_set.append(words2Vect(vocab, doc_list[doc_id]))
        train_labels.append(label_list[doc_id])
    
    # 训练贝叶斯模型: 计算信阿焰概率和各特征的条件概率
    p_v_0, p_v_1, p_abu = trainNB0(np.array(train_set), np.array(train_labels))
    # print('p_v_0:\n', p_v_0, '\np_v_1:\n', p_v_1, '\np_abu:\n', p_abu)

    # 预测测试数据集并统计错误率
    err_count = 0
    for id in test_set:
        word_vect = words2Vect(vocab, doc_list[id])
        if classifyNB(word_vect, p_v_0, p_v_1, p_abu) != label_list[id]:
            err_count += 1
    print('-- The error rate: {}%'.format(100.0 * float(err_count) / float(len(label_list))))


if __name__ == '__main__':
    # testMyBayes()
    testBOW()
    print('-- Test done.')
