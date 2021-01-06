# coding: utf-8

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = -(X**2)
Z2 = -(Y**2)
Z = 1.0 * (Z1 + 3 * Z2 + 2 * X * Y) + 6.0

plt.figure()

CS = plt.contour(X, Y, Z)

a = []
b = []

a.append(2.0)
b.append(2.0)

Z_val = -2.0**2 - 3.0 * 2.0**2 + 2.0 * 2.0 * 2.0 + 6.0
print('=> z val: %.3f' %(Z_val))

j = 1

for i in range(200):
    a_tmp = b[j - 1]
    a.append(a_tmp)
    b.append(b[j - 1])  

    j += 1

    b_tmp = a[j - 1] / 3

    Z_val = -a_tmp**2 - 3.0 * b_tmp**2 + 2.0 * a_tmp * b_tmp + 6.0
    print('=> z val: %.3f' %(Z_val))

    a.append(a[j - 1])
    b.append(b_tmp)

plt.plot(a, b)

plt.title('Coordinate Ascent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
