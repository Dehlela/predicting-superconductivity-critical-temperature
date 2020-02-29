import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Loading data
data = np.loadtxt('l01-data.txt')

# fetching inputs and converting them into matrices
x = data[:, 1].reshape(-1, 1)
y = data[:, 2].reshape(-1, 1)

# performing linear regression
linreg = LinearRegression()
linreg.fit(x, y)
prediction = linreg.predict(x)
print(prediction)

# plotting the model vs data
fig, ax = plt.subplots()
ax.scatter(x, y, color='blue', alpha=.8, s=120, marker='x')
ax.scatter(x, prediction, color='black')
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
plt.show()
