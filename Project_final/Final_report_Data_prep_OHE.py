# convert the categorical features to numerical features through One-Hot-Encoding
"""
OneHotEncoder details:

Encode categorical integer features using a one-hot aka one-of-K scheme.

The input to this transformer should be a matrix of integers,
    denoting the values taken on by categorical (discrete) features.

The output will be a sparse matrix where each column corresponds to one possible value of one feature.

It is assumed that input features take on values in the range [0, n_values).

This encoding is needed for feeding categorical data to many scikit-learn estimators,
    notably linear models and SVMs with the standard kernels.
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
                 r'\Project\Mid-term report/train_final.csv')

# assign the major label to substitute the question mark '?'
for j in [1, 6, 13]:
    major = df.iloc[:, j].value_counts()
    major_label = major.index[0]
    for i in range(df.shape[0]):
        if df.iloc[i, j] == '?':
            df.iloc[i, j] = major_label

# delete the header in the first row for convenience
# df.drop(0, axis=0, inplace=True)
df.reset_index(inplace=True, drop=True)

num_set = [0, 2, 4, 10, 11, 12]  # indices of the numerical attributes
all_indices = list(range(df.shape[1]-1))

# extract the categorical features
cat_set = [i for i in all_indices if i not in num_set]
cat_features = df.iloc[:, cat_set]

mm0 = cat_features.iloc[:, -1].value_counts()
n0 = list(mm0.index)

# extract the numerical features
num_features = copy.deepcopy(df.iloc[:, num_set].astype(int))
num_features.reset_index(inplace=True, drop=True)
num_features = num_features.values  # convert the numerical features to nd-array

# normalize the numerical features by (x(i)-min(x))/(max(x)-min(x))
min_max_scaler = preprocessing.MinMaxScaler()
num_features = min_max_scaler.fit_transform(num_features)


# extract the ground truth of the training data
y_gt = (copy.deepcopy(df.iloc[:, -1].astype(int)).to_numpy()).reshape(-1, 1)

# dimension for categorical features
d_cat = []
for j in cat_set:
    d_cat.append(len(df.iloc[:, j].unique()))

print(d_cat)


# =============== use one-hot-encoding to convert the categorical feature into vectors =====================

# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()

# use df.apply() to apply le.fit_transform to all columns
x1 = cat_features.apply(le.fit_transform)

# creat a OneHotEncoder object, and fit it to all of the columns in cat_features
# 1. Instance
enc = preprocessing.OneHotEncoder()

# 2.Fit
enc.fit(x1)

# 3. Transform
onehotlabels = enc.transform(x1).toarray()

# ================================= concat the numerical features with the categorical features =============
# store the final results
final_train = np.concatenate((num_features, onehotlabels, y_gt), axis=1)

print(final_train.shape)
print(final_train)
final_train = pd.DataFrame(final_train)

# the final results are all the numerical features, now the regression method can be used.

# store the processed data, the last column is the label for the training data
final_train.to_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                   r'\Project\FINAL_REPORT\processed_data/train_p.csv', index=False)

"""


# ====================== pre-processing the training data=======================
# read the data as a data-frame
df = pd.read_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
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

num_set = [0, 2, 4, 10, 11, 12]  # indices of the numerical attributes
all_indices = list(range(df.shape[1]))

# extract the categorical features
cat_set = [i for i in all_indices if i not in num_set]
cat_features = df.iloc[:, cat_set]
cat_features.columns = range(cat_features.shape[1])  # reset the column index

mm1 = cat_features.iloc[:, -1].value_counts()
n1 = list(mm1.index)

# extract the numerical features
num_features = copy.deepcopy(df.iloc[:, num_set].astype(int))
num_features.reset_index(inplace=True, drop=True)
num_features = num_features.values  # convert the numerical features to nd-array

# normalize the numerical features by (x(i)-min(x))/(max(x)-min(x))
min_max_scaler = preprocessing.MinMaxScaler()
num_features = min_max_scaler.fit_transform(num_features)


# replace the Hotland-Netherlands, since it only appears once
cat_features = copy.deepcopy(cat_features)
for i in range(len(cat_features)):
    if cat_features.iloc[i, -1] == 'Holand-Netherlands':
        cat_features.iloc[i, -1] = 'United-States'

# dimension for categorical features
d_cat = []
for j in range(len(cat_set)):
    d_cat.append(len(cat_features.iloc[:, j].unique()))

# =============== use one-hot-encoding to convert the categorical feature into vectors =====================

# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()

# use df.apply() to apply le.fit_transform to all columns
x1 = cat_features.apply(le.fit_transform)

# creat a OneHotEncoder object, and fit it to all of the columns in cat_features
# 1. Instance
enc = preprocessing.OneHotEncoder()

# 2.Fit
enc.fit(x1)

# 3. Transform
onehotlabels = enc.transform(x1).toarray()

# ================================= concat the numerical features with the categorical features =============
# store the final results
final_test = np.concatenate((num_features, onehotlabels), axis=1)

final_test = pd.DataFrame(final_test)

# the final results are all the numerical features, now the regression method can be used.

# store the processed data, the last column is the label for the training data
final_test.to_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                   r'\Project\FINAL_REPORT\processed_data/test_p.csv', index=False, header=False)
