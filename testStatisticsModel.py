#_*_coding:utf-8_*_

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.optimize as opt
import scipy.stats as st
# import seaborn.apionly as sns
import statsmodels.api as sm
import scipy.special as ss

plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00',
          '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']


#****************泊松分布描述的是小概率事件*****************
'''
1. 相对而言是小概率事件
2. 事件之间相互独立
3. 事件发生的概率是相对稳定的
4. pdf(概率分布函数), cdf(累积分布函数), pmf(概率质量函数)
5. 是离散随机变量在各特定取值上的概率。
概率质量函数和概率密度函数不同之处在于
：概率密度函数是对连续随机变量定义的，本身不是概率
，只有对连续随机变量的取值进行积分后才是概率。
'''
# fig = plt.figure(figsize=(11, 3))
# ax = fig.add_subplot(111)
# x_lim = 60
# mu = [5, 20, 40]
# for i in np.arange(x_lim):
#     plt.bar(i, st.poisson.pmf(mu[0], i), color=colors[3])
#     plt.bar(i, st.poisson.pmf(mu[1], i), color=colors[4])
#     plt.bar(i, st.poisson.pmf(mu[2], i), color=colors[5])

#     ax.set_xlim(0, x_lim)
#     ax.set_ylim(0, 0.2)
#     ax.set_ylabel('Probability mass')
#     ax.set_title('Poisson distribution')
#     plt.legend(['$mu$=%s' % mu[0], '$mu$=%s' % mu[1], '$mu$=%s' % mu[2]])

# plt.show()

#***********************************通过泊松分布建模
# 读取数据
path = os.getcwd() + '/hangout_chat_data.csv'
msg = pd.read_csv(path)
# print(msg)
y_obs = msg['time_delay_seconds'].values
# print(len(y_obs))

# fig = plt.figure(figsize=(11, 3))
# plt.title('Frequency of messages by response time')
# plt.xlabel('Response time(seconds')
# plt.ylabel('Number of messages')
# plt.hist(y_obs, range=[0, 60], bins=60, histtype='stepfilled')
# plt.show()

# 这里用的是最大似然估计, 为什么是-1？
# -1是因为调用的是最小化函数minimize_scalar，默认是有化成最小值，故相反


def poissonLogProb(mu, sign=-1):
    return np.sum(sign * st.poisson.logpmf(y_obs, mu=mu))


# 标量函数怎么理解？
freq_results = opt.minimize_scalar(poissonLogProb)
# print(freq_results)
print('The estimated value of mu is: %s' % freq_results['x'])

# 可视化优化过程: 这里又为什么是+1?


def visualOptimize():
    x = np.linspace(1, 60)
    log_vals = [poissonLogProb(i, sign=1) for i in x]
    y_min = np.min(log_vals)
    y_max = np.max(log_vals)
    fig = plt.figure(figsize=(7.5, 4))
    plt.plot(x, log_vals)

    # 填充制定面积
    plt.fill_between(x, log_vals, y_min, color=colors[0], alpha=0.25)

    # 要插入的公式部分由一对美元符号 $ 来进行标识，而具体的排版命令与 TeX 一样
    plt.title('Optimization of $mu$')
    plt.xlabel('$mu$')
    plt.ylabel('Log probability of $mu$ given data')
    plt.vlines(freq_results['x'], y_max, y_min,
               colors='red', linestyles='dashed')
    plt.ylim(ymin=y_min, ymax=0)
    plt.xlim(xmin=1, xmax=60)
    plt.show()


# 对任意一个泊松分布，参数 μ 代表的是均值和方差。
def responseTimeDistribution():
    fig = plt.figure(figsize=(11, 3))
    ax = fig.add_subplot(111)
    x_lim = 60
    mu = np.int(freq_results['x'])
    for i in np.arange(x_lim):
        plt.bar(i, st.poisson.pmf(mu, i), color=colors[3])

    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, 0.1)
    ax.set_xlabel('Response time in seconds')
    ax.set_ylabel('Probability mass')  # 概率质量值: 特定区间的概率值积分
    ax.set_title(
        'Estimated Poisson distribution for Hangout chat response time')
    plt.legend(['$lambda$ = %s' % mu])
    plt.show()


'''
马尔科夫链蒙特卡罗方法（MCMC）:MCMC 可解决高维空间里的积分和优化问题?
(1). 逆变换采样原理: 如果我们有个PDF为P(R)的分布，
那么对齐CDF的反函数进行均匀采样所得的样本分布将符合P(R)的分布。
(2). 有些复杂的分布函数或者高维分布的函数，其CDF无法求出，这样样本生成就十分困难,
此时，需要一些更加复杂的随机模拟方法生成样本, MCMC和Gibbs Sampling就是其中的两种,
这两种方法在现代贝叶斯分析中被广泛使用。
(3). MC(马尔科夫链)及其平稳分布: 马尔科夫链的收敛行为跟初始概率分布n_0无关,
MC的收敛行为由概率转移矩阵P决定的, 最后得到平稳分布的n_ith样本
(4). 玻尔兹曼分布？玻尔兹曼分布的采样问题？
(5). 基于MC采样的关键问题,是如何构造概率转移矩阵q, 使得平稳分布恰好是我们要的分布p(x)?
(6). MCMC的绝妙之处在于, 通过稳态的 Markov Chain 进行转移计算, 等效于从 P(x) 分布采样。
(7). 细致平稳条件(detailed balance condition):
假设我们有一个转移矩阵为q的MC,通常情况下：p(i)q(i->j) ≠ p(j)q(j->i), 也就是细致平稳条件不成立;
proposal distribution q(x) 也是需要小心确定的
我们引入一个 α(i->j), 我们希望
p(i)q(i->j)α(i->j)=p(j)q(j->i)α(j->i) (细致平稳公式是充分不必要条件)

在改造 q 的过程中引入的 α(i->j)称为接受率(acceptance probability),
物理意义可以理解为在原来的马氏链上,
从状态 i 以q(i->j) 的概率转跳转到状态j 的时候, 我们以α(i->j)的概率接受这个转移
可是这个α该怎么求解呢?
取什么样的 α(i->j) 以上等式能成立呢? 最简单的,按照对称性,我们可以取
α(i->j)=p(j)q(j->i), α(j->i)=p(i)q(i->j)即接受率的求解公式
(8). 为什么MCMC可以用于模型的参数估计呢？原理是怎样？
<<Baysian Data Analysis>>
'''

# 测试马尔科夫链的平稳分布(收敛)
# 一个向量不断地乘以一个矩阵更新这个向量，最终必将得到一个收敛的向量，
# 且这个向量只与矩阵相关,而与初始值(向量)无关得到平稳分布的n_ith样本


def MCConvergence(P, n_0, N):
    '''
    @param P: input n*n square matrix
    @param n_0: input vector of n elements
    @param N: iter times
    '''
    n_last = n_0
    print('0 generation:', n_0)
    for i in range(N):
        n_last.shape = (1, n_0.size)
        n_ith = np.dot(n_last, P)
        print('generation %d:' % (i + 1), n_ith)
        n_last = n_ith


def testMC():
    Q = np.zeros((3, 3))  # 概率转移矩阵
    Q[0][0] = 0.65
    Q[0][1] = 0.28
    Q[0][2] = 0.07
    Q[1][0] = 0.15
    Q[1][1] = 0.67
    Q[1][2] = 0.18
    Q[2][0] = 0.12
    Q[2][1] = 0.36
    Q[2][2] = 0.52
    print('transfer matrix:\n', Q)
    
    # n_0 = np.array([0.21, 0.68, 0.11])
    n_0 = np.array([0.75, 0.15, 0.1])
    MCConvergence(Q, n_0, 20)


def pdf(x):
    return 0.5 * x * x * exp(-x)

# beta概率分布函数


def betaS(x, a, b):
    return x**(a - 1) * (1 - x)**(b - 1)


# beta函数
def beta(x, a, b):
    return betaS(x, a, b) / ss.beta(a, b)


def testMCMC(a, b, iter_num=1000):
    cur = np.random.rand()
    states = [cur]
    for i in range(iter_num):
        next, u = np.random.rand(2, 1)
        if u < np.min((betaS(next, a, b) / betaS(cur, a, b), 1)):
            states.append(next)
            cur = next

    x = np.arange(0, 1, .01)
    plt.figure(figsize=(10, 5))
    plt.plot(x, beta(x, a, b), lw=2, label='real dist: a={}, b={}'.format(a, b))
    plt.hist(states[-100:], 25, normed=True, histtype='stepfilled', alpha=0.3,
             label='simu mcmc: a={}, b={}'.format(a, b))
    plt.show()


def testMCMCTrace():
    with pm.Model() as model:
        mu = pm.Uniform('mu', lower=0, upper=60)
        likelihood = pm.Poisson('likelihood', mu=mu,
                                observed=msg['time_delay_seconds'].values)
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(20000, step, start=start, progressbar=True)
        pm.traceplot(trace, varnames=['mu'], lines={'mu': freq_results['x']})
        # fig = plt.figure(figsize=(11, 3))
        # plt.subplot(121)
        # plt.title('Burnin trace')
        # plt.ylim(ymin=16.5, ymax=19.5)
        # plt.plot(trace.get_values('mu')[:1000])
        # fig = plt.subplot(122)
        # plt.title('Full trace')
        # plt.ylim(ymin=16.5, ymax=19.5)
        # plt.plot(trace.get_values('mu'))
        # pm.autocorrplot(trace[:2000], varnames=['mu'])
        # plt.show()
        return trace


'''
贝叶斯模型的一个核心优势就是简单灵活，可以实现一个分层模型。
分层模型有什么应用？
'''
import itertools
import scipy
from IPython.display import Image
from sklearn import preprocessing


def testMsgNum():
    msg_num_by_sender = msg.groupby('prev_sender')['conversation_id'].size()
    ax = msg_num_by_sender.plot(kind='bar', figsize=(12, 3),
                                title='Number of messages are sent per recipient',
                                color=colors[0])
    ax.set_xlabel('Previous Sender')
    ax.set_ylabel('Number of messages')
    plt.xticks(rotation=45)
    plt.show()


# 需要了解pm的API
'''
直观上，如果我们正确估计了模型参数，
那么我们应该可以从模型中采样得到类似的数据。结果很明显不是这样。
可能泊松分布不适合拟合这些数据。一种可选模型是负二项分布，特点
和泊松分布很像，只是有两个参数（μ 和 α），使得方差和均值独立。
回顾泊松分布只有一个参数 μ，既是均值，又是方差。
'''


def getN(mu, alpha):
    return 1.0 / alpha * mu


def getP(mu, alpha):
    return getN(mu, alpha) / (getN(mu, alpha) + mu)


# 测试负泊松分布与负二项分布模型对比
def testModelsComparision():
    fig = plt.figure(figsize=(10, 5))  # 初始化一个figure
    fig.add_subplot(211)  # 添加子图
    x_lim = 70
    mu = [15, 40]  # 测试连个mu的模型对比
    for i in np.arange(x_lim):
        plt.bar(i, st.poisson.pmf(mu[0], i), color=colors[3])
        plt.bar(i, st.poisson.pmf(mu[1], i), color=colors[4])
        plt.xlim(1, x_lim)
        plt.xlabel('Response time in seconds')
        plt.ylabel('Probability mass')
        plt.legend(['$lambda$ = %s' % mu[0], '$lambda$ = %s' % mu[1]])

    fig.add_subplot(212)
    a = [2, 4]  # 测试两个alpha
    for i in np.arange(x_lim):
        plt.bar(i, st.nbinom.pmf(
            i, n=getN(mu[0], a[0]), p=getP(mu[0], a[0])), color=colors[3])
        plt.bar(i, st.nbinom.pmf(
            i, n=getN(mu[1], a[1]), p=getP(mu[1], a[1])), color=colors[4])
        plt.xlabel('Response time in seconds')
        plt.ylabel('Probility mass')
        plt.title('Negative Binominal distribution')
        plt.legend(['$mu = %s, / beta = %s$' % (mu[0], a[0]),
                    '$mu = %s, / beta = %s$' % (mu[1], a[1])])
    plt.tight_layout()
    plt.show()


# 使用之前相同的数据集，继续对负二项分布的参数进行估计。
# 同样地，使用均匀分布来估计 μ 和 α。
# 用什么手段，参数进行模型检验？检验选用的模型是否合适？
def testNegativeBinominal():
    x_lim = 60
    burnin = 50000
    with pm.Model() as model:
        alpha = pm.Exponential('alpha', lam=0.2)
        mu = pm.Uniform('mu', lower=0, upper=100)
        y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)
        # 这个y_esti是什么，有什么用？
        y_esti = pm.NegativeBinomial(
            'y_esti', mu=mu, alpha=alpha, observed=msg['time_delay_seconds'].values)
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(200000, step, start=start, progressbar=True)
        pm.traceplot(trace[burnin:], varnames=['alpha', 'mu'])

    # fig = plt.figure(figsize=(10, 6))
    # fig.add_subplot(211)
    # y_pred = trace[burnin:].get_values('y_pred')
    # plt.hist(y_pred, range=[0, x_lim],
    #          bins=x_lim, histtype='stepfilled', color=colors[1])
    # plt.xlim(1, x_lim)
    # plt.ylabel('Frequency')
    # plt.title('Posterior predictive distribution')

    # fig.add_subplot(212)
    # plt.hist(msg['time_delay_seconds'].values,
    #          range=[0, x_lim], bins=x_lim, histtype='stepfilled')
    # plt.xlabel('Response time in seconds')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of observed data')
    # plt.tight_layout()
    # plt.show()
    return trace


'''
贝叶斯因子与分层模型
Bayes factor = p(x|H1) / p(x|H2)
'''
def testSepModels():
    indiv_traces = {}

    # convert categorical variables to integer
    le = preprocessing.LabelEncoder()
    participants_idx = le.fit_transform(msg['prev_sender'])
    # print('participants_idx:\n', participants_idx)
    participants = le.classes_
    print('participants:\n', participants)
    participants_num = len(participants)

    for p in participants:
        with pm.Model() as model:
            alpha = pm.Uniform('alpha', lower=0, upper=100)
            mu = pm.Uniform('mu', lower=0, upper=100)
            data = msg[msg['prev_sender'] == p]['time_delay_seconds'].values
            y_esti = pm.NegativeBinomial(
                'y_esti', mu=mu, alpha=alpha, observed=data)
            y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)
            start = pm.find_MAP()
            step = pm.Metropolis()
            trace = pm.sample(20000, step, start=start,
                              progressbar=True)  # sampling
            indiv_traces[p] = trace

    # visualize results
    # fig, axs = plt.subplots(3, 2, figsize=(12, 6))
    # axs = axs.ravel()  # obtain subplots
    # y_left_max = 2
    # y_right_max = 2000
    # x_lim = 60
    # ix = [3, 4, 6]  # selected samples

    # for i, j, p in zip([0, 1, 2], [0, 2, 4], participants[ix]):
    #     axs[j].set_title('Observed: %s' % p)
    #     axs[j].hist(msg[msg['prev_sender'] == p]['time_delay_seconds'].values,
    #                 range=[0, x_lim], bins=x_lim, histtype='stepfilled')
    #     axs[j].set_ylim([0, y_left_max])
    # for i, j, p in zip([0, 1, 2], [1, 3, 5], participants[ix]):
    #     axs[j].set_title('Posterior predictive distribution: %s' % p)
    #     axs[j].hist(indiv_traces[p].get_values('y_pred'),
    #                 range=[0, x_lim], bins=x_lim,
    #                 histtype='stepfilled', color=colors[1])
    #     axs[j].set_ylim([0, y_right_max])
    # axs[4].set_xlabel('Response time (seconds)')
    # axs[5].set_xlabel('Response time (seconds)')
    # plt.tight_layout()
    # plt.show()
    return indiv_traces


# 模型融合, 为什么进行模型融合？ 某些样本容量太小导致的误差
# 整体融合
def testModelFusion(trace, indiv_traces):
    global msg
    burnin = 50000
    comb_y_pred = np.concatenate([v.get_values('y_pred')
                                  for k, v in indiv_traces.items()])
    x_lim = 60
    y_pred = trace.get_values('y_pred')

    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(211)
    plt.hist(comb_y_pred, range=[0, x_lim],
             bins=x_lim, histtype='stepfilled', color=colors[1])
    plt.xlim(1, x_lim)
    plt.ylim(0, 20000)
    plt.ylabel('Frequency')
    plt.title('Posterior predictive distribution')

    fig.add_subplot(212)
    plt.hist(msg['time_delay_seconds'].values,
             range=[0, x_lim], bins=x_lim, histtype='stepfilled')
    plt.xlim(0, x_lim)  # 到底从0开始，还是从1开始?
    plt.xlabel('Response time in seconds')
    plt.ylim(0, 20)
    plt.ylabel('Frequency')
    plt.title('Distribution of observed data')
    plt.tight_layout()
    plt.show()


# 测试Gamma分布， 这个分布有什么特点?
'''
Γ函数是严格的凹函数
the gamma distribution is frequently used to model waiting times
'''
def testGammaDistribution():
    mu = [5, 25, 50]
    sd = [3, 7, 2]
    plt.figure(figsize=(11, 3))
    plt.title('Gamma distribution')
    with pm.Model() as model:
        for i, (j, k) in enumerate(zip(mu, sd)):
            samples = pm.Gamma('gamma_%s' % i, mu=j, sd=k).random(size=10**6)
            plt.hist(samples, bins=100, range=(0, 60),
                     color=colors[i], histtype='bar', alpha=1)
        plt.legend(['$mu$ = %s, $sigma$ = %s' % (mu[a], sd[a])
                    for a in [0, 1, 2]])
        plt.show()


'''
超参数是在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据。通常情况下，
需要对超参数进行优化，给学习机选择一组最优超参数，以提高学习的性能和效果。 
关键还在于理解Gamma分布的意义
'''
def testPartialFusionModel():
    global msg
    with pm.Model() as model:
        # hyper paramers
        hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=60)
        hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=50)
        hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=10)
        hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=50)

        # participants
        le = preprocessing.LabelEncoder()
        participants_idx = le.fit_transform(msg['prev_sender'])
        participants = le.classes_
        parti_num = len(participants)

        # parameters
        mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd, shape=parti_num)
        alpha = pm.Gamma('alpha', mu=hyper_alpha_mu,
                         sd=hyper_alpha_sd, shape=parti_num)

        # sampling
        y_esti = pm.NegativeBinomial('y_esti',
                                     mu=mu[participants_idx],
                                     alpha=alpha[participants_idx],
                                     observed=msg['time_delay_seconds'].values)
        y_pred = pm.NegativeBinomial('y_pred',
                                     mu=mu[participants_idx],
                                     alpha=alpha[participants_idx],
                                     shape=msg['prev_sender'].shape)
        start = pm.find_MAP()
        step = pm.Metropolis()
        hierarchical_trace = pm.sample(200000, step, progressbar=True)
        pm.traceplot(hierarchical_trace[120000:], varnames=['mu', 'alpha',
                                                            'hyper_mu_mu', 
                                                            'hyper_mu_sd', 
                                                            'hyper_alpha_mu',
                                                            'hyper_alpha_sd'])
        

if __name__ == '__main__':
    # visualOptimize()
    # testMC()
    # testMCMC(0.1, 0.1, 1800)
    # testMsgNum()
    # testModelsComparision()
    # testNegativeBinominal()
    # testGammaDistribution()
    trace = testNegativeBinominal()  # 获取所有数据的trace
    # print('trace:\n', trace.get_values('y_pred'))
    # indiv_traces = testSepModels()
    # print('indiv_traces:\n', indiv_traces)
    testModelFusion(trace, indiv_traces)
    # testPartialFusionModel()
    print('--Done.')


# ref:
# http://python.jobbole.com/85991/
# http://python.jobbole.com/88921/ 用 Python 实现一个大数据搜索引擎
# https://www.cnblogs.com/daniel-D/p/3388724.html
# https://www.cnblogs.com/ywl925/archive/2013/06/05/3118875.html
# http://blog.csdn.net/lanchunhui/article/details/50452515
# http://www.jianshu.com/p/63d7c6daefdc
# http://bbotte.com/more-info/upgrade-glibc-version-to-solve-glibc_2-14-not-found/ (升级glibc)
