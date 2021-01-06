# _*_coding: utf-8_*_
import hashlib
import os
import tarfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from six.moves import urllib

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
# print('currect working path:', os.getcwd())
LOCAL_PATH = os.path.join(os.getcwd(), '/simulator/tests/datasets/')
# print('LOCAL_PATH: ', LOCAL_PATH)
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL,
                       housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    # tgz_path = LOCAL_PATH
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # print('housing_url: ', housing_url)
    urllib.request.urlretrieve(housing_url, tgz_path)  # tgz_path:本地保存路径
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    # print('csv_path: ', csv_path)
    return pd.read_csv(csv_path)


# fetch_housing_data() # fetch data from https
housing = load_housing_data(LOCAL_PATH)
# print(housing.head())
# print(housing.info())
# 这个名叫households的feature是什么意思

# test feature 'ocean_proximity'
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())
# print('--Test done.\n')

# visualize different features
# housing.hist(bins=50, figsize=(12.5, 9)) # DataFrame的每一列(属性)都绘制直方图
# plt.tight_layout()
# plt.show()


def split_train_test(data, test_ratio):
    np.random.seed(42)  # 固定随机数种子
    shuffled_indices = np.random.permutation(len(data))  # 排列组合,打乱,洗牌
    # print('shuffled indices:\n', shuffled_indices)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]  # 前一部分是测试数据集
    train_indices = shuffled_indices[test_set_size:]  # 后面的是训练数据集
    return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), 'train +', len(test_set), 'test')


# 调用hash算法，根据阈值判定bool
def test_set_check(ID, test_ratio, hash):
    return hash(np.int64(ID)).digest()[-1] < 256 * test_ratio


# 通过MD5的哈希算法确定唯一的ID
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]  # 这个id_column是什么东东, 列的名称'index'
    # print('ids:\n', ids)
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    # print('in_test_set:\n', in_test_set)
    return data.loc[~in_test_set], data.loc[in_test_set]  # DataFrame索引方法


# 预处理得到稳定不变的测试数据集
# reset_index是DataFrame自带的API: 0, 1, 2, 3...
housing_with_id = housing.reset_index()  # add an 'index' column
# print(housing_with_id.head())
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
# print(len(train_set), 'train +', len(test_set), 'test')


# 人为构造独一无二的ID特征
# housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
# print(housing_with_id.head())
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')
# print(len(train_set), 'train +', len(test_set), 'test')

# 通过sklearn设置训练集和测试集
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(len(train_set), 'train +', len(test_set), 'test')

# 将中位数收入转化为类别型
# print('original median_income:\n', housing['median_income'])
# fig = plt.figure(figsize=(10, 8))
# fig.add_subplot(221)  # 向图层添加子图
# housing['median_income'].hist(bins=50, figsize=(6, 4))
# plt.title('original median_income hist')
# plt.show()

# 通过取整(向上取整)将连续型特征转化为离散型特征
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
# fig.add_subplot(222)
# housing['income_cat'].hist(bins=50, figsize=(6, 4))
# plt.title('ceiled median_income')

# 这个where什么意思? 条件表达式，< 5的不变， > 5的取5
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
# print('whole set ratio:\n',
#       housing['income_cat'].value_counts() / len(housing))
# fig.add_subplot(223)
# housing['income_cat'].hist(bins=50, figsize=(6, 4))  # 通过直方图显示各个category的比率
# plt.title('income cat hist')
# plt.tight_layout()
# plt.show()
# print(housing['income_cat'])

# 分层取样 stratified samplling: don't understand here
from sklearn.model_selection import StratifiedShuffleSplit
# n_split=1什么意思
split = StratifiedShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42)  # 固定随机种子
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# print('test set ratio:\n',
#       strat_test_set['income_cat'].value_counts() / len(strat_test_set))

# 丢弃前面的分析: 丢弃新加入的feature列
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


'''
exploring the data to gain insights
做机器学习算法的过程中往往伴随着数据分析与挖掘
'''
housing_expl = strat_train_set.copy()

# set alpha to 0.1 to highlight high_density areas
# housing_expl.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# plt.show()

# 通过DataFrame画散点图
# 圆圈半径大小代表人口规模，颜色代表房价中位数
# housing_expl.plot(kind='scatter', x='longitude', y='latitude',
#                   marker='o', alpha=0.5,
#                   s=housing_expl['population'] * 0.01, label='population', c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()
# plt.show()

# looking for correlations
coor_mat = housing_expl.corr()
# print('correlation matrix:\n', coor_mat)

# using pandas's scatter_matrix to explore correlations
from pandas.plotting import scatter_matrix
features = ['median_house_value', 'median_income',
            'total_rooms', 'housing_median_age']
# scatter_matrix(housing_expl[features], figsize=(10, 8))
# plt.show()

# to explore correlation between median_income and median_house_value
# housing_expl.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.2)
# plt.show()

'''
prepare data for learnning algorithm
'''
# 将标签与特征分离
# print(housing_expl.info()) # 查看含有缺失值的特征
housing = strat_train_set.drop('median_house_value', axis=1)  # 从原数据集中剔除label
housing_labels = strat_train_set['median_house_value'].copy()
# print('labels:\n', housing_labels)

# 数据清洗:主要针对消失的特征
# 1. 让中位数替换缺失的值
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")  # strategy是超参数

housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_nume.median().values)
X = imputer.transform(housing_num)  # X是numpy数组:用median补全total_bedrooms的缺失值得到的

# 将numpy数组转化为DataFrame:
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# print(housing_tr.info())

# 文本特征的处理:将文本型特征转化为数值型特征
from sklearn.preprocessing import LabelEncoder  # 标签编码:将文本型特征转化为数值型编码
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
# print('before encoding: ', housing_cat[:10])
housing_cat_encoded = encoder.fit_transform(housing_cat)
# print('after encoding: ', housing_cat_encoded[:10])
# print('mapping: ', encoder.classes_)

# 将<文本特征>转换为<数值特征>出现的编码问题：
# 独热编码（One-hot encoding）：状态编 码中对<每个状态>使用独立的一位表示。
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print('one hot:\n', housing_cat_1hot)
# print(housing_cat_1hot.toarray()) # 展示one-hot编码结果

# 一次性：文本特征转->整数特征->one-hot编码向量
from sklearn.preprocessing import LabelBinarizer  # 标签二值化
encoder = LabelBinarizer()  # sparse_output=True: return sparse matrix
housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)

# 自定义转换器
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6  # 属性索引


class CombinedFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bed_rooms_per_room=True):  # no *args or **kargs
        self.add_bed_rooms_per_room = add_bed_rooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bed_rooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]  # 卧室占比
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            # concatenate
            return np.c_[X, rooms_per_household, population_per_household]


feat_adder = CombinedFeatureAdder(add_bed_rooms_per_room=False)
housing_extra_feat = feat_adder.transform(housing.values)

# print(housing_extra_feat)


'''
归一化(normalization)对外值非常敏感
'''
# 特征缩放是什么意思？
# 使用管道，对数据进行预处理
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # 将数值特征标准化
# from sklearn.preprocessing import DataFrame

# 管道里面的estimator除了最后一个，其余的都必须是transformer:必须实现了fit_transform方法
num_pipeline = Pipeline([('imputer', Imputer(strategy="median")),    # 处理缺失特征值
                         ('feature_adder', CombinedFeatureAdder()),  # 添加特征
                         ('std_scaler', StandardScaler())])          # 将特征标准化
# housing_num_tr = num_pipeline.fit_transform(housing_num)  # 往管道中传入数据
# print('processed feature:\n', housing_num_tr)

# 为数值型和文本型特征分别建立一个管道
from sklearn.pipeline import FeatureUnion

num_features = list(housing_num)
# print('num_features: ', num_features)
cat_features = ['ocean_proximity']

from sklearn.base import BaseEstimator, TransformerMixin

# 定义一个自定义特征的数据帧


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self  # nothing changed

    def transform(self, X, y=None):
        return X[self.feature_names].values


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# 数值特征预处理pipeline
num_pipeline = Pipeline([('selector', DataFrameSelector(num_features)), # 选择特征
                         ('imputer', Imputer(strategy="median")),       # 处理缺失的数值特征数据
                         ('feature_adder', CombinedFeatureAdder()),     # 添加额外特征
                         ('std_scaler', StandardScaler()), ])           # 标准化特征数据

# 文本特征预处理pipeline
# 好好研究一下这个CategoricalEncoder...
cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_features)),
                         ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")), ])

# 将文本型管道和数值型管道结合起来
full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline),
                                               ('cat_pipeline', cat_pipeline)])

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared.shape)


'''
训练与评估
'''
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib


# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
# joblib.dump(lin_reg, './lin_model.pkl')  # save model
# some_data = housing.iloc[:5]
# # print('some_data:\n', some_data)
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print('Predictions:\t', lin_reg.predict(some_data_prepared))
# print('some labels:\n', some_labels)

from sklearn.metrics import mean_squared_error
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print('linear regreesiong rmse: %.3f$' % lin_rmse)


# 选用一个更复杂的模型来解决模型欠拟合问题: 回归树模型，如何实现自己动手实现一个回归树模型？
# from sklearn.tree import DecisionTreeRegressor  # 使用回归树算法
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# joblib.dump(tree_reg, './tree_model.pkl')  # save model
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print('tree regresiong rmse: %5.3f$' % tree_rmse)

# 使用交叉验证进行评估
from sklearn.model_selection import cross_val_score
# scores = cross_val_score(tree_reg, housing_prepared,
#                          housing_labels, scoring='neg_mean_squared_error', cv=10)
# tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std())
# display_scores(tree_rmse_scores)


# 尝试随机森林回归
from sklearn.ensemble import RandomForestRegressor
# forest_reg = RandomForestRegressor() # 工厂模式
# forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# # forest_mse = mean_squared_error(housing_labels, housing_predictions)
# # forest_rmse = np.sqrt(forest_mse)
# # print('forest regresiong rmse: %5.3f$' % forest_rmse)
# scores = cross_val_score(forest_reg, housing_prepared,
#                          housing_labels, scoring='neg_mean_squared_error', cv=10)
# forest_rmse_scores = np.sqrt(-scores)
# display_scores(forest_rmse_scores)

# 使用python得pikle模块来持久化模型及其参数
# joblib.dump(forest_reg, './forest_model.pkl') # save model
# my_forest = joblib.load('./forest_model.pkl')
# housing_predictions = my_forest.predict(housing_prepared)
# scores = cross_val_score(my_forest, housing_prepared,
#                          housing_labels, scoring='neg_mean_squared_error', cv=10)
# forest_rmse_scores = np.sqrt(-scores)
# display_scores(forest_rmse_scores)

# 对模型进行微调
# (1). 网格搜索Grid search: 暴力搜索
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 20, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 5, 7, 10]},
] # 随机森林回归得参数字典

forest_reg = RandomForestRegressor()  # 通过 随机数森林 进行回归
grid_search = GridSearchCV(forest_reg,
                           param_grid,
                           cv=5, # cv是cross_validation
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
print('gird_searched best params: ', grid_search.best_params_)
cures = grid_search.cv_results_
for mean_score, params in zip(cures['mean_test_score'], cures['params']):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
# print('feature_importances:\n', feature_importances)

extra_features = ['rooms_per_household', 'pop_per_household', 'bedrooms_per_room']
cat_one_hot_features = list(encoder.classes_)
print('cat_one_hot_features:\n', cat_one_hot_features)
features = num_features + extra_features + cat_one_hot_features
print('all features:\n', features)

sorted_feature_importance = sorted(zip(feature_importances, features), reverse=True) # 从大到小
print('sorted_feature_importance:\n', sorted_feature_importance)

# 在测试数据集上评价模型
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

# 调用流水线
X_test_prepared = full_pipeline.transform(X_test)

# 预测结果
final_predictions = final_model.predict(X_test_prepared)

# 呈现结果
for x, y in zip(X_test_prepared, final_predictions):
    print('X: ', x, '\n-> ', y, '$')
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('final_rmse: ', final_rmse)


'''
1. 最小二乘估计来源于高斯分布的极大似然估计量
难怪随机变量高斯分布并且相互独立是概率论很多理论的基础,
通过最小二乘解权重向量W
(1). 可以通过最小二乘求解权重向量的矩阵运算式
(2). 如何通过梯度下降求解权重向量w:
通过目标函数(代价函数)求偏导找到权重梯度,
<1>.有了初始权重向量，就可以求出此时的权重梯度向量，
<2>.有了权重梯度向量和给定的学习率就可以更新权重向量，
重复<1>, <2>两步，逐步迭代优化
最小二乘与梯度下降之间有何联系、区别？各有何优缺点？
# https://www.zhihu.com/question/20822481 (最小二乘法与梯度下降法有哪些区别联系?)
# http://blog.csdn.net/july_sun/article/details/53223962
# http://irwenqiang.iteye.com/blog/1552680 (高斯分布极大似然估计量->最小二乘法)

2. 各种距离：欧几里得距离，曼哈顿距离，汉明距离(范数指数不同对大数值敏感性不同，
范数指数越高对大树值越敏感？),
 切比雪夫距离(绝对值最大值)
夹角余弦也是一种距离么？
http://blog.csdn.net/tianlan_sharon/article/details/50904641

3. 分层抽样法也叫类型抽样法。它是从一个可以分成不同子总体（或称为层）的总体中，
按规定的比例从不同层中随机抽取样品（个体）的方法。定量调查中的分层抽样是一种卓越的概率抽样方式。

4. https://www.zhihu.com/question/35508851 (知乎关于机器学习算法的细节)

5. 这周末自己动手实现一个回归树算法 http://python.jobbole.com/88822/
6. 在在弄懂回归树的基础上研究随机森林与其他的决策树 http://database.51cto.com/art/201407/444788.htm
随机森林的优点：
(1). 在数据集上表现良好
(2). 在当前的很多数据集上，相对其他算法有着很大的优势
(3). 它能够处理很高维度(feature很多)的数据，并且不用做特征选择
(4). 在训练完后，它能够给出哪些feature比较重要(这是什么原理?)
(5). 在创建随机森林的时候，对generlization error使用的是无偏估计
(6). 训练速度快
(7). 在训练过程中，能够检测到feature间的互相影响
(8). 容易做成并行化方法
(9). 实现比较简单
'''

# ref:
# https://www.cnblogs.com/Determined22/p/6362951.html
# http://blog.csdn.net/uncle_gy/article/details/78786735
# https://github.com/PytLab/MLBox
# 广义线性回归的最小二乘解的矩阵运算形式推导
