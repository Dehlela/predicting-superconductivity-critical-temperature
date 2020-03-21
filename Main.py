import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
import seaborn as sb


def print_heading(heading):
    print("\n---------------------------------------------------")
    print(heading)
    print("---------------------------------------------------")


# ----------------------------------------------------- Loading --------------------------------------------------------
print_heading("Loading")
print("Loading original data... ")
orgData = pd.read_csv("Data/org_data.csv")

# ---------------------------------------------------- Cleaning --------------------------------------------------------
print_heading("Cleaning")

# Getting number of empty cells
counter = 0
for i in range(0, len(orgData)):  # len(data) = 21263
    if orgData.loc[i].empty:
        counter = counter + 1

print("Empty cells: ", counter)
'''orgData.info()'''  # all correct data types

# ---------------------------------------------------- Analysing -------------------------------------------------------

# Histograms for every feature
# Check outliers in important features
'''orgData.hist(bins=50, figsize=(20, 15))
plt.show()'''

# Check mean vs max, min
'''orgData.describe()'''

# --------------------------------------------------- Splitting --------------------------------------------------------
print_heading("Splitting")


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# Training and testing sets
print("Splitting into train and test sets...")

# The following has been commented to avoid creating new test sets on each program execution,
# which might lead to inconsistencies with the assignment report.
# Splitting was done at this stage, to avoid looking further into given data.
'''
train_set, test_set = split_train_test(orgData, 0.2)
print(len(train_set))
print(len(test_set))

train_set.to_csv("Data/train_set.csv", index=False)
test_set.to_csv("Data/test_set.csv", index=False)'''

print("Train set and Test set: stored in separate files.")

# --------------------------------------------------- Visualizing ------------------------------------------------------
print_heading("Visualizing")


# Correlation ----------------------------------------------------------------------------------------------------------
print("Observing correlation...")

# Looking at train_set so that we don't analyse the test set.
corrAnalyseData = pd.read_csv("Data/train_set.csv")
'''
# Checking correlation of each feature wrt all other features
corr_matrix = corrAnalyseData.corr()
print(corr_matrix["wtd_std_FusionHeat"].sort_values(ascending=False))

# Checking correlation of selected sets of features (most collinear from above)
attribs1 = ["mean_atomic_mass", "gmean_atomic_mass", "wtd_mean_atomic_mass", "wtd_gmean_atomic_mass",
            "entropy_atomic_mass", "entropy_atomic_radius", "entropy_Valence", "wtd_entropy_atomic_mass",
            "wtd_entropy_atomic_radius", "range_atomic_mass", "std_atomic_mass"]
attribs2 = ["mean_atomic_radius", "wtd_mean_atomic_radius", "wtd_gmean_atomic_radius", "gmean_atomic_radius",
            "range_atomic_radius", "std_atomic_radius", "wtd_std_atomic_radius"]
attribs3 = ["range_ElectronAffinity", "std_ElectronAffinity", "range_ThermalConductivity", "std_ThermalConductivity",
            "wtd_std_ThermalConductivity"]
attribs4 = ["mean_Valence", "gmean_Valence", "wtd_mean_Valence", "wtd_gmean_Valence", "range_Valence", "std_Valence"]
attribs5 = ["mean_fie", "wtd_mean_fie", "range_ThermalConductivity", "std_ThermalConductivity",
            "wtd_std_ThermalConductivity"]

# Heat-map for sets
corr_matrix = corrAnalyseData[attribs4].corr()
sb.heatmap(corr_matrix, cmap="YlGnBu", xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)

# Scatter matrix for sets
scatter_matrix(corrAnalyseData[attribs5], figsize=(12, 8))
'''
# Checking correlation for selected pairs of features (most collinear from above)
plt.scatter(x=corrAnalyseData["range_ElectronAffinity"], y=corrAnalyseData["std_ElectronAffinity"], alpha=0.3, color="green")
plt.show()

# ------------------------------------------------- Data Preparation ---------------------------------------------------
print_heading("Preparing Data")


# Feature Selection ----------------------------------------------------------------------------------------------------
print("Performing feature selection from previous analysis...")

# This function removes some of the unimportant features, as specified in the paper,
# which include fie, density and fusion heat and all related attributes.
# The result of this selection is stored in respective "v1" file.
def remove_unimportant_features(data):
    list1 = ["mean_", "gmean_", "entropy_", "range_", "std_"]
    list2 = ["fie", "Density", "FusionHeat"]

    new_data = data.copy()
    for l1 in list1:
        for l2 in list2:
            feature = l1 + l2
            new_data = new_data.drop(feature, axis=1)

    list3 = []  # list of "wtd_mean_", "wtd_gmean_" etc.
    for l1 in list1:
        list3.append("wtd_" + l1)

    for l3 in list3:
        for l2 in list2:
            feature = l3 + l2
            new_data = new_data.drop(feature, axis=1)

    return new_data


splitTrainData = pd.read_csv("Data/train_set.csv")

train_features = remove_unimportant_features(splitTrainData)
print("Saving updated data to a new file...")
train_features.to_csv("Data/reduced_train_set_v1.csv", index=False)

print("Removing perfectly correlated features...")
# dropping attributes that are perfectly correlated to some important attributes.
# The result of this selection is stored in respective "v2" file.
def remove_correlated_features(data):
    # trial 1: removing 15 (+30 from v1)
    most_imp_features = ["gmean_atomic_mass", "wtd_gmean_atomic_mass", "entropy_atomic_radius", "entropy_Valence",
                         "wtd_entropy_atomic_radius", "std_atomic_mass", "wtd_gmean_atomic_radius",
                         "std_atomic_radius", "wtd_std_atomic_radius", "std_ElectronAffinity",
                         "std_ThermalConductivity", "wtd_std_ThermalConductivity", "gmean_Valence",
                         "wtd_gmean_Valence", "std_Valence"]

    # trial 2: removing important 7 (+30 from v1)
    least_imp_features = ["entropy_atomic_radius", "wtd_gmean_atomic_radius",
                          "std_ElectronAffinity", "std_ThermalConductivity",
                          "gmean_Valence", "wtd_gmean_Valence", "std_Valence"]

    new_data = data.copy()
    for attrib in most_imp_features:
        new_data = new_data.drop(attrib, axis=1)
    return new_data


# To remove 30 features + correlated features
# reducedTrainData = pd.read_csv("Data/reduced_train_set_v1.csv")

# To remove only correlated features
# reducedTrainData = pd.read_csv("Data/train_set.csv")

# train_features = remove_correlated_features(reducedTrainData)
# train_features.to_csv("Data/reduced_train_set_v2.csv", index=False)

# Preparation for Regression -------------------------------------------------------------------------------------------
print("Reading from final set of features...")

# Training with 0 features removed
# mainTrainData = pd.read_csv("Data/train_set.csv")

# Training with 30 features removed
mainTrainData = pd.read_csv("Data/reduced_train_set_v1.csv")

# Training with correlated features removed
# mainTrainData = pd.read_csv("Data/reduced_train_set_v2.csv")

# Separating variables
print("Separating independent and dependent variables...")
features = mainTrainData.drop("critical_temp", axis=1)
temperatures = mainTrainData["critical_temp"].copy()

# Scaling
print("Scaling...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ------------------------------------------------- Choosing Model -----------------------------------------------------
print_heading("Choosing model")


def cross_validation(model, feature_set):
    # Cross Validation RMSE and R2
    scores = cross_validate(model, feature_set, temperatures, scoring=("neg_mean_squared_error", "r2"),
                            return_train_score="true", cv=10)
    cv_train_scores = np.sqrt(-scores['train_neg_mean_squared_error'])
    cv_valid_scores = np.sqrt(-scores['test_neg_mean_squared_error'])
    print("RMSE:")
    print("Training RMSE Mean: ", cv_train_scores.mean())
    print("Standard Deviation: ", cv_train_scores.std())
    print("Cross Validation RMSE Mean: ", cv_valid_scores.mean())
    print("Standard Deviation: ", cv_valid_scores.std())

    print("\nR2 Score/Accuracy:")
    print("Training Accuracy Mean: ", scores['train_r2'].mean())
    print("Standard Deviation: ", scores['train_r2'].std())
    print("Cross Validation Accuracy Mean: ", scores['test_r2'].mean())
    print("Standard Deviation: ", scores['test_r2'].std())
    print("---------------------------------------------------")


# Simple Linear Regression
print("\nSimple Linear Regression cross-validation:\n")
lin_model = LinearRegression()
cross_validation(lin_model, scaled_features)

# Decision Tree Regression
'''print("Decision Tree Regression: ")
tree_model = DecisionTreeRegressor()
cross_validation(tree_model, scaled_features)'''

# Random Forest Regression
'''print("Random Forest Regression: ")
forest_model = RandomForestRegressor()
cross_validation(forest_model, scaled_features)'''

# Polynomial Regression
print("\nAdding polynomial features...")
poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(scaled_features)

print("\nPolynomial Regression cross-validation:\n")
lin_model = LinearRegression()
cross_validation(lin_model, poly_features)

# Regularization ---------------------------------------------
print("\nPerforming Regularization...")

# Ridge
print("\nRidge cross-validation:\n")
ridge_model = Ridge(alpha=1)
cross_validation(ridge_model, poly_features)

# Lasso
'''print("\nLasso cross-validation:\n")
lasso_model = Lasso(alpha=0.3, fit_intercept=True)
cross_validation(lasso_model, poly_features)'''

# ElasticNet
'''print("\nElasticNet cross-validation:\n")
elastic_model = ElasticNet(alpha=10, l1_ratio=0.3, fit_intercept=True)
cross_validation(elastic_model, poly_features)'''

# ---------------------------------------------------- Training --------------------------------------------------------
print_heading("Training")
print("Training chosen model: Ridge with Polynomial Features")
ridge_model.fit(poly_features, temperatures)

# --------------------------------------------------- Evaluating -------------------------------------------------------
print_heading("Evaluating")

# Applying feature selection to test set
print("Applying feature selection to test set...")

reducedTestData = pd.read_csv("Data/test_set.csv")
test_features = remove_unimportant_features(reducedTestData)
test_features.to_csv("Data/reduced_test_set_v1.csv", index=False)

'''reducedTestData = pd.read_csv("Data/reduced_test_set_v1.csv")
test_features = remove_correlated_features(reducedTestData)
test_features.to_csv("Data/reduced_test_set_v2.csv", index=False)'''

# Predicting ---------------------------------------------------------

test_data = pd.read_csv("Data/reduced_test_set_v1.csv")

# Separating variables
print("Separating independent and dependent variables...")
test_features = test_data.drop("critical_temp", axis=1)
test_temperatures = test_data["critical_temp"].copy()

# Scaling
print("Scaling test data...")
scaled_test_features = scaler.transform(test_features)

# Adding polynomial features
print("Adding polynomial features...")
test_poly_features = poly.fit_transform(scaled_test_features)

print("Predicting...")
test_predictions = ridge_model.predict(test_poly_features)

print_heading("Final Result")
mse = mean_squared_error(test_temperatures, test_predictions)
rmse = np.sqrt(mse)
print("RMSE: " + str(rmse))

r2 = r2_score(test_temperatures, test_predictions)
print("R2 Score/Accuracy: " + str(r2))
print("\n")
