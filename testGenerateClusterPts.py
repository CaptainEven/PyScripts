
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt


# data, target=make_blobs(n_samples=1200, n_features=2, centers=2, cluster_std=[2.35, 2.45])
# print('data:\n', data)
# print('target:\n', target)
# for item in data:
# 	item[0] += 50.0
# 	item[1] += 50.0
# np.savetxt('f:/cluster.txt', data, fmt='%.3f')
# np.savetxt('f:/class.txt', target, fmt='%d')
# plt.scatter(data[:,0], data[:,1], c=target);
# plt.show()

# *******************************************************
types = []
with open('f:/class.txt', 'r') as fh:
	for line in fh.readlines():
		types.append(int(line.strip().split()[0]))
pts = []
with open('f:/cluster.txt', 'r') as fh:
	for line in fh.readlines():
		x = line.strip().split()[0]
		y = line.strip().split()[1]
		pts.append((x, y))
data = np.array(pts)
plt.scatter(data[:,0], data[:,1], c=types) # scatter方法详解？
plt.show()

# *******************************************************
# import pandas as pd
# df = pd.read_csv('e:/simulator/tests/iris/iris.data', names=[
#     'sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
# df.drop(['class'], axis=1, inplace=True) # DataFrame原位丢弃最后一列
# print(df)
# data = np.array(df)
# print(data)
# np.savetxt('f:/iris.txt', data, fmt='%.2f')

print('--Done.')

