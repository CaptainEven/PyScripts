# # -*- coding: utf-8 -*-
# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
# import pywt.data


# # Load image
# original = pywt.data.aero()
# print('original.shape:',original.shape)


# # Wavelet transform of image, and plot approximation and details
# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']

# coeffs2 = pywt.dwt2(original, 'bior1.3')
# LL, (LH, HL, HH) = coeffs2

# fig = plt.figure(figsize=(8,8))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(2, 2, i + 1)
#     ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=12)


# fig.suptitle("dwt2 coefficients", fontsize=14)

# # Now reconstruct and plot the original image
# reconstructed = pywt.idwt2(coeffs2, 'bior1.3')
# fig = plt.figure(figsize=(8 ,8))
# plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)


# # Check that reconstructed image is close to the original
# np.testing.assert_allclose(original, reconstructed, atol=1e-13, rtol=1e-13)


# # Now do the same with dwtn/idwtn, to show the difference in their signatures
# coeffsn = pywt.dwtn(original, 'bior1.3')
# fig = plt.figure(figsize = (8, 8))
# for i, key in enumerate(['aa', 'ad', 'da', 'dd']):
#     ax = fig.add_subplot(2, 2, i + 1)
#     ax.imshow(coeffsn[key], origin='image', interpolation="nearest",
#               cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=12)
# fig.suptitle("dwtn coefficients", fontsize=14)


# # Now reconstruct and plot the original image
# reconstructed = pywt.idwtn(coeffsn, 'bior1.3')
# fig = plt.figure(figsize = (8, 8))
# plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)


# # Check that reconstructed image is close to the original
# np.testing.assert_allclose(original, reconstructed, atol=1e-13, rtol=1e-13)
# plt.show()

# -*- coding: cp936 -*-
import pywt
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.robust import stand_mad

wavtag = 'db8'

# #===============================================================================
# # 图1：绘出Haar小波母函数
# #===============================================================================

# # 这里不是“函数调用”，二是“对象声明和创建”
# # 创建了一个pywt.Wavelet类，用以描述小波母函数的各种性质
# w = pywt.Wavelet('Haar')

# # 调用Wavefun()成员函数，返回：
# # phi - scaling function 尺度函数
# # psi - wavelet function 母函数
# phi, psi, x = w.wavefun(level=10)

# # 注意，此处采用“面对对象”的方式使用matplotlib
# # 而不是“状态机”的方式
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlim(-0.02, 1.02)
# ax.plot(x, psi)
# ax.grid(True)
# plt.show()


# #===============================================================================
# # 图2：Debauchies小波的尺度函数和母函数
# #===============================================================================

# db8 = pywt.Wavelet(wavtag)
# scaling, wavelet, x = db8.wavefun()

# fig = plt.figure(2)
# ax1 = fig.add_subplot(121)
# ax1.plot(x, scaling)
# ax1.set_title('Scaling function,' + wavtag)
# ax1.set_ylim(-1.2, 1.2)
# ax1.grid(True)

# ax2 = fig.add_subplot(122, sharey=ax1)
# ax2.set_title('Wavelet,' + wavtag)
# ax2.plot(x, wavelet)
# ax2.tick_params(labelleft=False)
# ax2.grid(True)

# plt.tight_layout()
# plt.show()

#===============================================================================
# 图3：小波去噪模拟，原始信号和混合噪声的信号
#===============================================================================


def Blocks(x):
    K = lambda x: (1.0 + np.sign(x)) / 2.0
    t = np.array(
        [[0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]]).T
    h = np.array([[4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]]).T
    return 3.655606 * np.sum(h * K(x - t), axis=0)


def bumps(x):
    K = lambda x: (1.0 + np.abs(x)) ** -4.0
    t = np.array([[.1, .13, .15, .23, .25, .4, .44, .65, .76, .78, .81]]).T
    h = np.array([[4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 2.1, 4.2]]).T
    w = np.array(
        [[.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005]]).T
    return np.sum(h * K((x - t) / w), axis=0)


# 构造原始数据
x = np.linspace(0, 1, 2**15)
blk = bumps(x)
print('blk:\n', blk)

# 构造含噪声的数据
np.random.seed(12345)
nblk = blk + stats.norm().rvs(2**15) * 0.3

fig = plt.figure(3)
ax31 = fig.add_subplot(211)
ax31.plot(x, blk)
ax31.grid(True)
ax31.set_title('Original Data')
ax31.tick_params(labelbottom=False)

ax32 = fig.add_subplot(212)
ax32.plot(x, nblk)
ax32.grid(True)
ax32.set_title('Noisy Data')

plt.show()

#===============================================================================
# 图4,5：小波分析，及数据展示
#===============================================================================


def coef_pyramid_plot(coefs, first=0, scale='uniform', ax=None):
    '''
    Parameters
    ----------
    coefs : array-like
        Wavelet Coefficients. Expects an iterable in order Cdn, Cdn-1, ...,
        Cd1, Cd0.
    first : int, optional
        The first level to plot.
    scale : str {'uniform', 'level'}, optional
        Scale the coefficients using the same scale or independently by
        level.
    ax : Axes, optional
        Matplotlib Axes instance

    Returns
    -------
    Figure : Matplotlib figure instance
        Either the parent figure of `ax` or a new pyplot.Figure instance if
        `ax` is None.
    '''
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, axisbg='lightgrey')
    else:
        fig = ax.figure
    n_levels = len(coefs)
    n = 2**(n_levels - 1)  # assumes periodic
    if scale == 'uniform':
        biggest = [np.max(np.abs(np.hstack(coefs)))] * n_levels
    else:
       # multiply by 2 so the highest bars only take up .5
        biggest = [np.max(np.abs(i)) * 2 for i in coefs]
    for i in range(first, n_levels):
        x = np.linspace(2**(n_levels - 2 - i), n - 2**(n_levels - 2 - i), 2**i)
        ymin = n_levels - i - 1 + first
        yheight = coefs[i] / biggest[i]
        ymax = yheight + ymin
        ax.vlines(x, ymin, ymax, linewidth=1.1)

        ax.set_xlim(0, n)
        ax.set_ylim(first - 1, n_levels)
        ax.yaxis.set_ticks(np.arange(n_levels - 1, first - 1, -1))
        ax.yaxis.set_ticklabels(np.arange(first, n_levels))
        ax.tick_params(top=False, right=False, direction='out', pad=6)
        ax.set_ylabel("Levels", fontsize=14)
        ax.grid(True, alpha=.85, color='white', axis='y', linestyle='-')
        ax.set_title('Wavelet Detail Coefficients',
                     fontsize=16, position=(.5, 1.05))
        fig.subplots_adjust(top=.89)
        return fig


fig = plt.figure(4)
ax4 = fig.add_subplot(111, axisbg='lightgrey')
fig = plt.figure(5)
ax5 = fig.add_subplot(111, axisbg='lightgrey')

# 调用wavedec()函数对数据进行小波变换
# mode指定了数据补齐的方式
#‘per’指周期延拓数据
true_coefs = pywt.wavedec(blk, wavtag, level=11, mode='per')
noisy_coefs = pywt.wavedec(nblk, wavtag, level=11, mode='per')

# 绘出‘coefficient pyramid’
# 注意，这里只绘出了detail coefficients
# 而没有展示approximation coefficient(s),该数据存在true_coefs[0]中
fig1 = coef_pyramid_plot(true_coefs[1:], scale='level', ax=ax4)
fig1.axes[0].set_title('Original Wavelet Detail Coefficients')

fig2 = coef_pyramid_plot(noisy_coefs[1:], scale='level', ax=ax5)
fig2.axes[0].set_title('Noisy Wavelet Detail Coefficients')

plt.show()

#===============================================================================
# 图6：降噪——全局阈值
# 图7：重构数据——对比效果
#===============================================================================


sigma = stand_mad(noisy_coefs[-1])
uthresh = sigma * np.sqrt(2.0 * np.log(len(nblk)))
denoised_coefs = noisy_coefs[:]
denoised_coefs[1:] = (pywt._thresholding.soft(data, value=uthresh)
                      for data in denoised_coefs[1:])

fig = plt.figure(6)
ax6 = fig.add_subplot(111, axisbg='lightgrey')
fig3 = coef_pyramid_plot(denoised_coefs[1:], scale='level', ax=ax6)
fig3.axes[0].set_title('Denoised Wavelet Detail Coefficients')

signal = pywt.waverec(denoised_coefs, wavtag, mode='per')

fig = plt.figure(7)
ax71 = fig.add_subplot(211)
ax71.plot(x, nblk)
ax71.grid(True)
ax71.set_title('Noisy Data')
ax71.tick_params(labelbottom=False)

ax72 = fig.add_subplot(212)
ax72.plot(x, signal, label='Denoised')
ax72.plot(x, blk, color='red', lw=0.5, label='Original')
ax72.grid(True)
ax72.set_title('Denoised Data')
ax72.legend()
plt.show()

# 安装opencv-python: http://www.lfd.uci.edu/~gohlke/pythonlibs/
