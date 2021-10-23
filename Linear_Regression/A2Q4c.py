# import the csv file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面'
                 r'\CS6350 Machine Learning\Assignments'
                 r'\Assignment2\A2Q4\concrete\train.csv', header=None)

# number of rows of the input data, i.e., the number of examples
rows = train_data.shape[0]
# number of columns of the input data, i.e., the number of features + 1
cols = train_data.shape[1]
# take the features
train_data_features = train_data.loc[:, 0:cols-2]
# take the values
y_values = train_data.loc[:, cols-1]
# convert the dataframe to array
train_features = train_data_features.values
y = y_values.values

# creat a column vector with ones to include the bias parameter b
bs = np.ones([rows, 1])

# stack the features and the ones column vector
augmented_features = np.concatenate((bs, train_features), axis=1)

# convert y, the values, to a column vector
y = np.mat(y).T

x = augmented_features.T

step1 = np.dot(x, x.T)
invstep1 = np.matrix(step1)
inv_step1 = invstep1.I
step2 = np.dot(x,y)

# compute the optimal weight vector.
optimal_w = np.dot(inv_step1,step2)
print(optimal_w)

np.savetxt('D:\optimal_w.txt', optimal_w)