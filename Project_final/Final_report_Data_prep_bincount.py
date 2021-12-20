# convert the categorical features to numerical features through bin count.
"""
Bincount details:

count the numbers of each feature values throughout the dataset, and divide the count by the number
of samples, use the probability to represent the corresponding categorical feature.

"""
# import the data
import numpy as np
from numpy import argmax
import pandas as pd
import copy
# import preprocessing from sklearn
from sklearn import preprocessing

"""
# ====================== pre-processing the training data=======================
# read the data as a data-frame
df = pd.read_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Project\Mid-term report/train_final.csv', header=None)

# assign the major label to substitute the question mark '?'
for j in [1, 6, 13]:
    major = df.iloc[:, j].value_counts()
    major_label = major.index[0]
    for i in range(df.shape[0]):
        if df.iloc[i, j] == '?':
            df.iloc[i, j] = major_label

# delete the header in the first row for convenience
df.drop(0, axis=0, inplace=True)
df.reset_index(inplace=True, drop=True)

num_set = [0, 2, 4, 10, 11, 12]  # indices of the numerical attributes
all_indices = list(range(df.shape[1]-1))

num_samples = df.shape[0]
# =================================== process the numerical features ===========================
# extract the numerical features
num_features = copy.deepcopy(df.iloc[:, num_set].astype(int))
num_features = num_features.values  # convert the numerical features to nd-array
# normalize the numerical features by (x(i)-min(x))/(max(x)-min(x))
min_max_scaler = preprocessing.MinMaxScaler()
num_features = min_max_scaler.fit_transform(num_features)
num_features = pd.DataFrame(num_features)

# extract the ground truth of the training data
y_gt = copy.deepcopy(df.iloc[:, -1].astype(int))

# ================================== process the categorical features ===========================
# extract the categorical features
cat_set = [i for i in all_indices if i not in num_set]
cat_features = copy.deepcopy(df.iloc[:, cat_set])
cat_features.columns = range(cat_features.shape[1])  # reset the column index

# count the values in the categorical feature part of dataframe and replace the
# categorical feature with its probability
for j in range(len(cat_set)):
    val_counts0 = cat_features.iloc[:, j].value_counts()
    val_counts = list(val_counts0)
    feature_name = list(val_counts0.index)
    relabels = [ii/num_samples for ii in val_counts]
    for i in range(num_samples):
        idx = feature_name.index(cat_features.iloc[i, j])
        cat_features.iloc[i, j] = relabels[idx]

frames = [num_features, cat_features, y_gt]
final_bincount = pd.concat(frames, axis=1)
# reset the column index
final_bincount.columns = range(final_bincount.shape[1])

# the final results are all the numerical features, now the regression method can be used.
# store the processed data, the last column is the label for the training data
final_bincount.to_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                      r'\Project\FINAL_REPORT\processed_data/train_bin_count.csv', index=False)
"""

# ========================= process the test data ======================================
# ====================== pre-processing the test data=======================
# read the data as a data-frame
df = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Project\FINAL_REPORT\original_data\test_final.csv')

# assign the major label to substitute the question mark '?'
for j in [1, 6, 13]:
    major = df.iloc[:, j].value_counts()
    major_label = major.index[0]
    for i in range(df.shape[0]):
        if df.iloc[i, j] == '?':
            df.iloc[i, j] = major_label

# delete the header in the first row for convenience
df.reset_index(inplace=True, drop=True)

print(df)

num_set = [0, 2, 4, 10, 11, 12]  # indices of the numerical attributes
all_indices = list(range(df.shape[1]))

num_samples = df.shape[0]
# =================================== process the numerical features ===========================
# extract the numerical features
num_features = copy.deepcopy(df.iloc[:, num_set].astype(int))
num_features = num_features.values  # convert the numerical features to nd-array
# normalize the numerical features by (x(i)-min(x))/(max(x)-min(x))
min_max_scaler = preprocessing.MinMaxScaler()
num_features = min_max_scaler.fit_transform(num_features)
num_features = pd.DataFrame(num_features)

# ================================== process the categorical features ===========================
# extract the categorical features
cat_set = [i for i in all_indices if i not in num_set]
cat_features = copy.deepcopy(df.iloc[:, cat_set])
cat_features.columns = range(cat_features.shape[1])  # reset the column index

# count the values in the categorical feature part of dataframe and replace the
# categorical feature with its probability
for j in range(len(cat_set)):
    val_counts0 = cat_features.iloc[:, j].value_counts()
    val_counts = list(val_counts0)
    feature_name = list(val_counts0.index)
    relabels = [ii/num_samples for ii in val_counts]
    for i in range(num_samples):
        idx = feature_name.index(cat_features.iloc[i, j])
        cat_features.iloc[i, j] = relabels[idx]

frames = [num_features, cat_features]
final_bincount = pd.concat(frames, axis=1)
# reset the column index
final_bincount.columns = range(final_bincount.shape[1])

# the final results are all the numerical features, now the regression method can be used.
# store the processed data, the last column is the label for the training data
final_bincount.to_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                      r'\Project\FINAL_REPORT\processed_data/test_bin_count.csv', index=False, header=False)
