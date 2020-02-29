import numpy as np
import pandas as pd
# ------------------ Splitting -----------------------
reducedData = pd.read_csv('Reduced_Features.csv')


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(reducedData, 0.2)
print(len(train_set))
print(len(test_set))

train_set.to_csv(r'Splitting\train_set.csv', index=False)
test_set.to_csv(r'Splitting\test_set.csv', index=False)


