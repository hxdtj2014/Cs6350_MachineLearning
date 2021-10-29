# Data preprocessing
# import the csv file
import numpy as np
import pandas as pd
from random import shuffle
import copy


# ========================== training data preprocessing ======================
# import the training data
df = pd.read_csv(r'C:\Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Assignments\A3/bank-note/train.csv', header=None)


# set the last column to be either -1 and 1
for i in range(len(df)):
    if df.iloc[i, -1] == 0:
        df.iloc[i, -1] = -1

# fetch the ground truth of the train data
y_s = (copy.deepcopy(df.iloc[:, -1])).values
# ========================== test data preprocessing ======================
# import the training data
df_t = pd.read_csv(r'C:\Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Assignments\A3/bank-note/test.csv', header=None)

# set the last column to be either -1 and 1
for i in range(len(df_t)):
    if df_t.iloc[i, -1] == 0:
        df_t.iloc[i, -1] = -1


# fetch the ground truth of the test data
y_st = (copy.deepcopy(df_t.iloc[:, -1])).values
# ========================== define functions ================================

def sgn_func(numbers):
    """
    :param numbers: a number to be determined its sign
    :return:
    """
    if float(numbers) >= 0:
        return 1
    else:
        return -1


def standard_perceptron(data, epoch, y_s):
    """
    :param data: the training dataframe
    :param epoch: the shuffle times
    :param y_s: the ground truth of the training data
    :return: the weight vector w = [w_1, w_2, ..., w_n, b] with the bias parameter
    b included.
    """
    # fetch the ground truth
    # y_s = (copy.deepcopy(data.iloc[:, -1])).values
    # set the augmented space
    for ii in range(len(data)):
        data.iloc[ii, -1] = 1
    # set the initial weight vector to be zero vector
    w_length = data.shape[1]
    w = (np.zeros([1, w_length]))[0]
    # r is the learning rate and is set at 0.1
    r = 0.1

    # shuffle the data
    indices = []
    for times in range(epoch):
        # shuffle the row indices
        row_idx = [r_idx for r_idx in range(data.shape[0])]
        shuffle(row_idx)
        indices.append(row_idx)
    h = []  # the shuffled row indices max_epoch*len(df)
    for zz in range(len(indices)):
        h = h + indices[zz]

    for j in h:
        x_j = (data.loc[j, :]).values
        p = x_j.dot(w)  # take the dot product
        delta_w = y_s[j] * p

        if delta_w <= 0:
            w = w + r*y_s[j]*x_j

    return w


def standard_perceptron_predict(data, w):
    """
    :param data: a dataframe to predict
    :param w: the weight vector after training the train data
    :return: the prediction in a list
    """

    # set the augmented space
    if data.shape[1] == len(w):
        for i in range(len(data)):
            data.iloc[i, -1] = 1
    elif data.shape[1] < len(w):
        data[data.shape[1]] = np.ones((len(data), 1))

    predict_standard = []
    for k in range(len(data)):
        b = (data.loc[k, :]).values
        sgn = w.dot(b)
        predict_standard.append(sgn_func(sgn))

    return predict_standard


# error calculation between the predicted labels and the true labels
def err_calculation(true_label, predicted_label):
    count = 0  # correct predication count
    for k in range(len(true_label)):
        if true_label[k] != predicted_label[k]:
            count += 1
    return count / float(len(true_label))


# ============================ test error calculation ========================
errors = []
weights = []
for i in range(1, 11):
    weight = standard_perceptron(df, i, y_s)
    c = standard_perceptron_predict(df_t, weight)
    err_test = err_calculation(y_st, c)
    weights.append(weight)
    errors.append(err_test)

err = pd.DataFrame(errors, index=['epoch = 1', 'epoch = 2', 'epoch = 3', 'epoch = 4',
                                   'epoch = 5', 'epoch = 6', 'epoch = 7', 'epoch = 8',
                                   'epoch = 9', 'epoch = 10'], columns=['Errors'])
wei = pd.DataFrame(weights, index=['epoch = 1', 'epoch = 2', 'epoch = 3', 'epoch = 4',
                                   'epoch = 5', 'epoch = 6', 'epoch = 7', 'epoch = 8',
                                   'epoch = 9', 'epoch = 10'],
                   columns=['w_1', 'w_2', 'w_3', 'w_4', 'b'])

frame = [wei, err]
output = (pd.concat(frame, axis=1)).round(4)
print(output)





