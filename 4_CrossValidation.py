import numpy as np
import pandas as pd

mainTrainData = pd.read_csv(r'Splitting\train_set.csv')

# setting partitions for 10-fold cross validation
shuffled_indices = np.random.permutation(len(mainTrainData))
partition = int(len(mainTrainData) / 10)  # int to get 1701 instead of 1701.1

# Ideally, data is to be divided into 10 equal parts,
# but since 17011 is a prime number (80% of original data = 17011 rows),
# there will be 1 extra row of data.
# Hence we divide the data into 9 equal parts of size 1701
# and the last part will be of length 1702.
data_indices = {}
partitioned_data = {}
counter = 0
limit = len(mainTrainData) - partition - 1  # limits upto second-last partition (i.e., all equal partitions of len 1701)
for i in range(0, limit, partition):
    data_indices[counter] = shuffled_indices[i:i + partition]
    partitioned_data[counter] = mainTrainData.iloc[data_indices[counter]]
    counter = counter + 1

# storing the last partition of len 1702
data_indices[counter] = shuffled_indices[limit:]
partitioned_data[counter] = mainTrainData.iloc[data_indices[counter]]
