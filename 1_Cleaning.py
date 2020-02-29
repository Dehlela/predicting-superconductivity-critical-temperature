
import pandas as pd

# Loading data
orgdata = pd.read_csv('org_data.csv')

# ------------------ Cleaning -----------------------
# Getting number of empty cells
counter = 0
for i in range(0, len(orgdata)):  # len(data) = 21263
    if orgdata.loc[i].empty:
        counter = counter + 1
print("Number of empty cells: " + str(counter))