# import the csv file
import numpy as np
import pandas as pd
import copy
from random import shuffle
# ========================== training data pre-processing ==================
# import the training data
df = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Assignments\Assignment_4\A4_lectures\bank-note/train.csv', header=None)
# set the last column to be either -1 and 1
for i in range(len(df)):
    if df.iloc[i, -1] == 0:
        df.iloc[i, -1] = -1

# fetch the ground truth of the train data
y_s = (copy.deepcopy(df.iloc[:, -1])).to_numpy()
y = (copy.deepcopy(df.iloc[:, -1])).to_numpy()

# fetch the first four columns of the train data and convert it to an array
x = (df.iloc[:, range(4)]).to_numpy()
num_of_samples = np.shape(x)[0]

# ========================== test data pre-processing ======================
# import the training data
df_t = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面\CS6350 Machine Learning'
                   r'\Assignments\Assignment_4\A4_lectures\bank-note/test.csv', header=None)

# set the last column to be either -1 and 1
for i in range(len(df_t)):
    if df_t.iloc[i, -1] == 0:
        df_t.iloc[i, -1] = -1


# fetch the ground truth of the test data
y_st = (copy.deepcopy(df_t.iloc[:, -1])).to_numpy()

# fetch the first four columns of the test data and convert it to an array
x_t = (df_t.iloc[:, range(4)]).to_numpy()

# error calculation between the predicted labels and the true labels
def err_calculation(true_label, predicted_label):
    count = 0  # correct predication count
    for k in range(len(true_label)):
        if true_label[k] != predicted_label[k]:
            count += 1
    return count / float(len(true_label))


def sgn_func(numbers):
    """
    :param numbers: a number to be determined its sign
    :return:
    """
    if float(numbers) >= 0.0:
        return 1.0
    else:
        return -1.0


df1 = copy.deepcopy(df.to_numpy())
df2 = copy.deepcopy(df_t.to_numpy())
for i in range(len(df1)):
    df1[i, -1] = 1

r = [0.1, 0.5, 1, 5, 100]


def gaussian_kernel_maker(x1,x2,gamma):
    """
    Description: to make a gaussian kernel between two vectors, in form of nd array
    :param x1: the training dataset, a nd array
    :param x2: the training dataset or test dataset, a nd array
    :param gamma: the variance in gaussian kernel, sigma^2
    :return:
    """
    k_mat = np.ndarray([x1.shape[0], x2.shape[0]])
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            norm2 = (np.linalg.norm(x1[i]-x2[j]))**2
            k_mat[i, j] = np.exp(-norm2/gamma)
    return k_mat


def dual_nl_perceptron(dataset, gamma, epoch):
    """
    :param dataset: training data in form of DataFrame
    :param gamma: the variance in gaussian kernel, sigma^2
    :param epoch: total shuffle times
    :return: the number of mistakes made for predicting each sample, which is
    stored in a list
    """

    df1 = copy.deepcopy(dataset.to_numpy())
    # fetch the ground truth
    y_gt =(copy.deepcopy(dataset.iloc[:, -1])).to_numpy()

    for i in range(len(df1)):
        df1[i, -1] = 1
    # make gaussian kernel matrix
    kernel_g = gaussian_kernel_maker(dataset.to_numpy(),
                                     dataset.to_numpy(), gamma)

    num_samp = dataset.shape[0]
    counter = np.zeros([num_samp])

    # shuffle the indices
    indices = []
    for times in range(epoch):
        # shuffle the row indices
        row_idx = [r_idx for r_idx in range(df1.shape[0])]
        shuffle(row_idx)
        indices.append(row_idx)

    h = []  # the shuffled row indices max_epoch*len(df)
    for zz in range(len(indices)):
        h = h + indices[zz]

    for jj in h:
        cy = counter * y_gt
        decision = sgn_func(y_gt[jj]*np.dot(cy, kernel_g[jj]))
        if decision != y_gt[jj]:
            counter[jj] = counter[jj] + 1

    return counter


def dual_perceptron_predict(train_data, test_data, gamma):
    """
    Description: Use the Gaussian kernel to predict test data
    :param train_data:  the data to train, in form of DataFrame
    :param test_data:  the data to be predicted, in form of DataFrame
    :param gamma: the denominator for the Gaussian kernel, a scalar value
    :return: the predicted label for the test data, stored in a list
    """

    df1 = copy.deepcopy((train_data.to_numpy()))
    df2 = copy.deepcopy((test_data.to_numpy()))

    cc = dual_nl_perceptron(train_data, gamma, 1)
    weights = cc

    # make the kernel matrix between the training data and the test data
    kk = gaussian_kernel_maker(df1, df2, gamma)

    # fetch the ground truth of the training data
    y_gt = (copy.deepcopy(train_data.iloc[:, -1])).to_numpy()

    cy1 = weights * y_gt
    predict = []

    for j in range(len(test_data)):
        results = sgn_func(np.dot(cy1, kk[:, j]))
        predict.append(results)

    return predict


# store the errors
err1 = []
err2 = []
for ga in r:
    # training errors
    pre_train = dual_perceptron_predict(df, df, ga)
    err_train = err_calculation(y_s, pre_train)
    err1.append(str(err_train*100) + '%')

    # test errors
    pre_test = dual_perceptron_predict(df, df_t, ga)
    err_test = err_calculation(y_st, pre_test)
    err2.append(str(err_test*100) + '%')

# print the training errors
print('The training errors for r = 0.1, 0.5, 1, 5, 100, respectively, are:')
print(err1)
# print the test errors
print('The test errors for r = 0.1, 0.5, 1, 5, 100, respectively, are:')
print(err2)