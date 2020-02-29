import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Loading data
orgdata = pd.read_csv('org_data.csv')

# ------------------ Analysing & Visualising -----------------------
orgdata.hist(bins=50, figsize=(20, 15))
plt.show()



