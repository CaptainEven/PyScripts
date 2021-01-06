from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def func_1(x, a, b, c):
    return a * np.exp(b * x) + c


def func_2(x, a, b):
    return x**a + b


x_data = np.linspace(0, 4, 50)
# y = func_1(x_data, 2.5, 1.5, 0.5)
y = func_2(x_data, 2.5, 1.3)
print('origin a: 2.50, b: 1.30')
y_data = y + 0.5 * np.random.normal(size=len(x_data))
plt.plot(x_data, y_data, 'b-')
popt, pcov = curve_fit(func_1, x_data, y_data)
# y2 = [func_1(i, popt[0], popt[1], popt[2]) for i in x_data]
y2 = [func_2(x_val, popt[0], popt[1]) for x_val in x_data]
plt.plot(x_data, y2, 'r--')
print('fitted a: %.2f, b: %.2f' %(popt[0], popt[1]))
plt.show()
