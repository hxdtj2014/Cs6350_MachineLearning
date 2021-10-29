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


def voted_perceptron(data, epoch, y_s):
    """
    :param data: the training dataframe
    :param epoch: the shuffle times
    :param y_s: the ground truth of the training data
    :return: weights: the weight vector w = [w_1, w_2, ..., w_n, b] with the bias parameter
    b included at the last column,
    correct_predict_times: the counts of the correct prediction of each unique vectors on the training
    data, stored in a list

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

    m = 0
    c = 1

    cm = []
    weights = []
    idx = []
    for j in h:
        x_j = (data.loc[j, :]).values
        p = x_j.dot(w)  # take the dot product
        delta_w = y_s[j] * p

        if delta_w <= 0:
            w = w + r*y_s[j]*x_j  # update w
            m = m + 1
            c = 1  # not correct prediction

            weights.append(w.tolist())
        else:
            c = c + 1 # correct prediction
        cm.append(c)
        idx.append(m)

    unique_m = list(set(idx))
    correct_predict_times = []  # i.e., the passing times of each unique weight vector deduct the first time
    for jk in unique_m:
        time = idx.count(jk) - 1
        correct_predict_times.append(time)

    return weights, correct_predict_times


def voted_perceptron_predict(data, w, c_m):
    """
    :param data: a dataframe to predict
    :param w: the weight vectors stored in matrix [w1_vect, w2_vect, ....]
    :return: final: the prediction in a list
    """
    # set the augmented space
    if data.shape[1] == len(w[0]):
        for i in range(len(data)):
            data.iloc[i, -1] = 1
    elif data.shape[1] < len(w[0]):
        data[data.shape[1]] = np.ones((len(data), 1))

    # matrix multiplication
    weight_matrix = (np.mat(w)).T
    dataset_matrix = data.values
    product = np.dot(dataset_matrix, weight_matrix)
    # apply the sign function
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            product[i, j] = sgn_func(product[i, j])
    
    c_m_vector = np.mat(c_m)
    predict = np.dot(product, c_m_vector.T)

    final = []
    for i in range(len(predict)):
        h = sgn_func(predict[i][0])
        final.append(h)

    return final


# error calculation between the predicted labels and the true labels
def err_calculation(true_label, predicted_label):
    counts = 0  # correct predication count
    for k in range(len(true_label)):
        if true_label[k] != predicted_label[k]:
            counts += 1
    return counts / float(len(true_label))


# ============================ test error calculation ==================================
errors = []
weights_final = []
for i in range(1, 11):
    weight = voted_perceptron(df, i, y_s)

    # the unique weight vectors, including bias parameter at the last column
    w_voted = weight[0]
    c_m = weight[1]
    c = voted_perceptron_predict(df_t, w_voted, c_m)
    err_test = err_calculation(y_st, c)
    errors.append(err_test)

    count = pd.DataFrame(c_m)
    # round off the decimals after four digits
    ww = (pd.DataFrame(w_voted)).round(4)

    frame = [ww,  count]
    outputs = pd.concat(frame, axis=1)
    df2 = outputs.set_axis(['w_1', 'w_2', 'w_3', 'w_4', 'b', 'count'], axis=1, inplace=False)
    weights_final.append(df2)

# save the results into multiple sheets in a single excel file
with pd.ExcelWriter('A:\Data_Processing_for_experiments/' +
                    'A3Q2b' + '.xlsx', mode="w", engine="openpyxl") as writer:
    for i in range(len(weights_final)):
        weights_final[i].to_excel(writer, sheet_name='epoch_' + str(i+1), index=True, index_label=None)


# print the errors at each maximum epoch, i.e, 1, 2, .., 10
names = []
for i in range(len(errors)):
    a_err = 'epoch_' + str(i+1)
    names.append(a_err)

errors_final = pd.DataFrame(errors, index=names, columns=['Error'])
print(errors_final)



