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

# prepare the initial augmented weight vector [b, w1, w2,..,wd]=0
w = np.zeros([cols, 1])

# set the learning rate to be a small magnitude, i.e., r = 0.
r = 0.01
# set a threshold to identify a convergence, set it at 10^-6.
threshold = 10**-6
iteration_time = 0

# convert y, the values, to a column vector
y = np.mat(y).T

s1 = y - np.dot(augmented_features, w)

# compute the initials
delta = np.dot(augmented_features.T, r*s1)

# consider the cost function of the intial iteration
cost_function = []

while np.linalg.norm(delta) >= threshold:
    part_1 = augmented_features.T
    part_2 = y-np.dot(augmented_features, w)
    delta = np.dot(part_1, r*part_2)

    # compute the cost
    cost1 = np.linalg.norm(part_2) # the difference between y and w^T*xi
    cost2 = 0.5*cost1**2
    cost_function.append(cost2)
    w = w + delta

    # monitor the times of iteration
    iteration_time = iteration_time + 1
    if iteration_time == 1e10:
        break
        print('limitation for the time of iteration is reached')
#print(w)
xx = np.arange(len(cost_function))
plt.plot(xx, cost_function)
plt.show()
#print(len(cost_function))
#print(len(xx))

# save the cost_function for plotting
np.savetxt('D:\cost_function.txt', cost_function)
np.savetxt('D:\weightvector.txt', w)

#--------------------------------------------------------------------#
# compute the values of the test data.
test_data = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面'
                 r'\CS6350 Machine Learning\Assignments'
                 r'\Assignment2\A2Q4\concrete\test.csv', header=None)

# number of rows of the input data, i.e., the number of examples
rows1 = test_data.shape[0]
# number of columns of the input data, i.e., the number of features + 1
cols1 = test_data.shape[1]

# take the features
test_data_features = test_data.loc[:, 0:cols-2]

# take the values
y1_values = test_data.loc[:, cols-1]

# convert the dataframe to array
test_features1 = test_data_features.values
y1 = y1_values.values
# convert y, the values, to a column vector
y1 = np.mat(y1).T
# creat a column vector with ones to include the bias parameter b
bs1 = np.ones([rows1, 1])
# stack the features and the ones column vector
augmented_features1 = np.concatenate((bs1, test_features1), axis=1)
# compute the cost
s2 = y1 - np.dot(augmented_features1, w)
s3 = np.linalg.norm(s2)
cost_test = 0.5*s3**2

np.savetxt('D:\predicted.txt', s2)
print(cost_test)

