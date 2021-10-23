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

# convert y, the values, to a column vector
y = np.mat(y).T

# consider the cost function of the intial iteration
cost_function = []
# construct the sequence of the sampling
#arr = np.arange(rows)
#np.random.shuffle(arr)

count = 0
Times = 15000

while count <= Times:
    i = np.random.randint(0, rows)
    arr = np.arange(rows)
    np.random.shuffle(arr)
    # sample randomly unrepeatable
    idx = arr[i]
    aug_vector = np.mat(augmented_features[i, :])
    x1 = y[idx] - np.dot(aug_vector, w)

    # update the weight vector
    diff_w = np.multiply(r*x1, aug_vector.T)
    w = w + diff_w

    # compute the cost function
    difference = y-np.dot(augmented_features, w)
    length = np.linalg.norm(difference)
    cost = 0.5*length**2
    cost_function.append(cost)

    # set the iteration times
    count = count + 1

print(w)
print(cost_function[-1])

# plot the cost function versus iteration times
xx = np.arange(len(cost_function))
plt.plot(xx, cost_function)
plt.xlabel('Iteration times')
plt.ylabel('Cost function')
plt.title('Cost function vs Iteration times')
plt.show()
print(len(xx))
print(y-np.dot(augmented_features,w))
np.savetxt('D:\predicted.txt', w)

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
difference1 = y1 - np.dot(augmented_features1, w)
length = np.linalg.norm(difference1)
test_cost = 0.5 * length ** 2
print(test_cost)