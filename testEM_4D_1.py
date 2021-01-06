#_*_coding: utf-8_*_
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# 读取数据
iris = datasets.load_iris()
print(type(iris))
x = iris.data[:, :2]
y = iris.target
mu = np.array([np.mean(x[y == i], axis=0) for i in range(3)])
print('实际均值 = \n', mu)

# K-Means  
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
y_hat_1 = kmeans.fit_predict(x)
y_hat_1[y_hat_1 == 0] = 3
y_hat_1[y_hat_1 == 1] = 0
y_hat_1[y_hat_1 == 3] = 1

mu_1 = np.array([np.mean(x[y_hat_1 == i], axis=0) for i in range(3)])
print('K-Means均值 = \n', mu_1)
print('分类正确率为', np.mean(y_hat_1 == y))

# GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(x)

print('GMM均值 = \n', gmm.means_)

y_hat_2 = gmm.predict(x)
y_hat_2[y_hat_2 == 1] = 3
y_hat_2[y_hat_2 == 2] = 1
y_hat_2[y_hat_2 == 3] = 2
print('分类正确率为', np.mean(y_hat_2 == y))

