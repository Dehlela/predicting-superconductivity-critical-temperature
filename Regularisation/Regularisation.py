#
# Kasim Terzic (kt54) Jan 2018

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Create a dictionary to pass to matplotlib
# This is an easy way to set many parameters at once
fontsize = "30"
params = {'figure.autolayout': True,
          'legend.fontsize': fontsize,
          'figure.figsize': (16, 8),
          'axes.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize}
plt.rcParams.update(params)

np.random.seed(3)
m = 15
X = 6 * np.random.rand(m, 1) - 3
y = 0.2 * X ** 2 + .1 * X + 2 + .3 * np.random.randn(m, 1)

print(X.min(), X.max())
# print(y)

# Create a new figure and an axes objects for the subplot
fig, ax = plt.subplots()

# Use a light grey grid to make reading easier, and put the grid
# behind the display markers
ax.grid(color='lightgray', linestyle='-', linewidth=1)
ax.set_axisbelow(True)
ax.set_ylim(1, 4)
ax.set_xlim(-4, 4)

# Draw a scatter plot of x vs y
ax.scatter(X, y, color='blue', alpha=.8, s=140, marker='^')
ax.set_xlabel('X')
ax.set_ylabel('y')

# Save as an image file
plt.savefig('plot_parabola.png')

# Linear regression
X_test = np.linspace(-3, 3, 100)
X_test = X_test.reshape((100, 1))
new_X = X_test

# Polynomial regression order 7, least squares
X7 = np.hstack((X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7))
new_X7 = np.hstack((new_X, new_X ** 2, new_X ** 3, new_X ** 4, new_X ** 5, new_X ** 6, new_X ** 7))
linreg = LinearRegression()
linreg.fit(X7, y)
new_y = linreg.predict(new_X7)
plt.plot(new_X, new_y, color='red', linewidth=3)
plt.savefig('plot_reg_none.png')

# Polynomial regression order 7
lasso = Lasso(alpha=.05, fit_intercept=True, normalize=True)
lasso.fit(X7, y)
new_y = lasso.predict(new_X7)
plt.plot(new_X, new_y, color='green', linewidth=3)
plt.savefig('plot_reg_lasso.png')

# Polynomial regression order 7
ridge = Ridge(alpha=1, fit_intercept=True, normalize=True)
ridge.fit(X7, y)
new_y = ridge.predict(new_X7)
plt.plot(new_X, new_y, color='purple', linewidth=3)
ax.legend(['Least squares', 'L1 (Lasso)', 'L2 (Ridge)'], loc='lower left')
plt.savefig('plot_reg_ridge.png')

# Display in a window
plt.show()
