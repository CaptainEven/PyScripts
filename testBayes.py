#_*_coding: utf-8_*_

from collections import defaultdict
import numpy as np

'''
1. 朴素贝叶斯的两个假设前提:
(1). 各个特征之间相互独立
(2). 各个特征的重要性相同
2. token包括单词和标点、字符串、数字组：token是一个语义单元？
'''

class NativeBayesClassifier(object):
    '''
    朴素贝叶斯分类器
    '''

    def train(self, dataset, labels):
        '''
        训练朴素贝叶斯分类器
        @param dataset: 输入数据集:所有训练文档的数据向量: M×N matrix
        @param labels: 输入数据集所有文档类别: 1×N list
        @return cond_probs: 训练的道德条件概率矩阵: M×K matrix
        @return: label_probs: 各种类型的概率: 1×K list
        '''
        # 按照不同类别标记分类
        sub_datasets = defaultdict(lambda: [])  # 传入一个空list作为键对应的值
        label_cnt = defaultdict(lambda: 0)  # 键对应的值初始化为0

        for doc_vect, label in zip(dataset, labels):
            sub_datasets[label].append(doc_vect)
            label_cnt[label] += 1

        # 计算个类别概率
        label_probs = {k: float(v) / float(len(labels))
                       for k, v in label_cnt.items()}

        # 计算不同类型的条件概率
        cond_probs = {}
        dataset = np.array(dataset)
        for label, sub_dataset in sub_datasets.items():
            sub_dataset = np.array(sub_dataset) # 转换成numpy array
            cond_prob_vect = np.log((np.sum(sub_dataset, axis=0) + 1) / (np.sum(dataset) + 2))


#--------------我自己实现的朴素贝叶斯分类器



if __name__ == '__main__':
    print('-- Test done.')

# https://github.com/PytLab/MLBox
