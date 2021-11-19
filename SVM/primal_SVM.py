# import the csv file
import numpy as np
import pandas as pd
from random import shuffle
import copy
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

# ========================== training data pre-processing ==================
# import the training data
df = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Assignments\Assignment_4\A4_lectures\bank-note/train.csv', header=None)

# set the last column to be either -1 and 1
for i in range(len(df)):
    if df.iloc[i, -1] == 0:
        df.iloc[i, -1] = -1

# fetch the ground truth of the train data
y_s = (copy.deepcopy(df.iloc[:, -1])).values
# ========================== test data pre-processing ======================
# import the training data
df_t = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Assignments\Assignment_4\A4_lectures\bank-note/test.csv', header=None)

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
    :return: the sgn of the number
    """
    if float(numbers) >= 0:
        return 1
    else:
        return -1


def label_predict(data, w):
    """
    :param data: a dataframe to predict
    :param w: the weight vector after training the train data,
              the bias parameter b is folded into the end of the weight vector.
    :return: the prediction in a list
    """

    # set the augmented space
    if data.shape[1] == len(w):
        for i in range(len(data)):
            data.iloc[i, -1] = 1
    elif data.shape[1] < len(w):
        data[data.shape[1]] = np.ones((len(data), 1))

    predict = []
    for k in range(len(data)):
        b = (data.loc[k, :]).values
        sgn = w.dot(b)
        predict.append(sgn_func(sgn))

    return predict


# error calculation between the predicted labels and the true labels
def err_calculation(true_label, predicted_label):
    ct = 0  # correct predication count
    for k in range(len(true_label)):
        if true_label[k] != predicted_label[k]:
            ct += 1
    return ct / float(len(true_label))


# set the hyper-parameters
C = [100/873, 500/873, 700/873]


def ssgd_svm(data, g_t, c, shuffle_times, r_t_type, z=0):
    """
    Description: the primal domain with stochastic sub-gradient descent algorithm
                 the maximum epoch is set at 100.
    :param data: the training data-frame
    :param g_t: the ground truth of the training data
    :param c: the hyper parameter c
    :param shuffle_times: times to shuffle the dataset
    :param r_t_type: the schedule of select r_t, if r_t_type == 1, r_t = r0/(1 + r0/d*t);
                    if r_t_type == 2, r_t = r0/(1 + t)
    :param z: whether to compute the training error at each iteration
    :return: the weight vector w = [w_1, w_2, ..., w_n, b] with the bias parameter
    b included.
    """
    # set the augmented space, the last column of the dataframe is assigned 1.

    for ii in range(len(data)):
        data.iloc[ii, -1] = 1
    # set the initial weight vector to be zero vector
    w_length = data.shape[1]
    w = (np.zeros([1, w_length]))[0]

    # shuffle the data 100 times at one time to avoid for loops
    indices = []
    for times in range(shuffle_times):
        # shuffle the row indices
        row_idx = [r_idx for r_idx in range(data.shape[0])]
        shuffle(row_idx)
        indices.append(row_idx)

    # number of training examples in each shuffle, i.e., the rows of the dataframe
    n = data.shape[0]
    # N = len(indices)
    
    h = []  # the shuffled row indices max_epoch*len(df)
    for zz in range(len(indices)):
        h = h + indices[zz]

    # r0 and d needs to be tuned, which is accomplished through trials
    r0 = 2.3
    d = 1

    # loss function sum(r_t^2)
    loss = []

    t = 0
    for j in h:
        # count the total iteration times
        t = t + 1
        # r_t
        if r_t_type == 1:       # for (a)
            r_t = r0/(1 + r0/d*t)
        elif r_t_type == 2:     # for (b)
            r_t = r0 / (1 + t)
        else:
            r_t = r0 / (1 + r0 / d * t)

        x_j = (data.loc[j, :]).values
        p = x_j.dot(w)  # take the dot product
        delta_w = g_t[j] * p

        w_0 = w[0:w_length - 1]
        # substitute the last element of w by 0
        w_last0 = np.append(w_0, 0)

        if delta_w <= 1:
            w = w - r_t*w_last0 + r_t*c*n*g_t[j]*x_j

        else:
            coe_0 = (np.ones([1, w_length]))[0]
            coe = (1-r_t)*coe_0
            coe[-1] = 1  # the lase element was set at 1
            w = coe*w

        if z == 1:
            # compute the error in each step
            b0 = label_predict(data, w)
            jk = err_calculation(g_t, b0)
            loss.append(jk)
    if z == 1:
        return w, loss
    else:
        return w


# ========================== A4 part II, question 2(a) convergence validation==========================================
# validate the convergence of the selected r_0 and d for the step size r_t
# check the training errors when setting, r0 = 2.3, d = 1.

# epoch T = 1, C = 100/873
a2 = (ssgd_svm(df, y_s, C[0], 1, 1, z=1))[1]
horizontal = [i+1 for i in range(len(a2))]
plt.plot(horizontal, a2, label='1/873')
plt.xlabel('Iteration times')
plt.ylabel('training error')
plt.title('T = 1')
plt.show()

# epoch T = 5, C = 100/873
a2 = (ssgd_svm(df, y_s, C[0], 5, 1, z=1))[1]
horizontal = [i+1 for i in range(len(a2))]
plt.plot(horizontal, a2, label='1/873')
plt.xlabel('Iteration times')
plt.ylabel('training error')
plt.title('T = 5')
plt.show()


# ========================================= A4 part II, question 2(a) and (b)===========================================
# compute the training error, test error for different c
train_err_2a = []
test_err_2a = []
weight_vector = []
count = 0

# choose q = 1 to generate results for 2(a), set q = 2 to generates results for 2(b)
q = 1  # IMPORTANT

for i in range(len(C)):
    a = ssgd_svm(df, y_s, C[i], 100, q)  # fetch the learned weight vector
    b = label_predict(df, a)
    c = err_calculation(y_s, b)
    train_err_2a.append(str(round(c*100, 2)) + "%")

    d = label_predict(df_t, a)
    e = err_calculation(y_st, d)
    test_err_2a.append((str(round(e*100, 2)) + "%"))

    weight_vector.append(a)

    count = count + 1
    progress = round(100 * count / 7, 2)
    print(str(progress) + ' has completed')

# construct a DataFrame to show the predicted results
row_names = ['100/873', '500/873', '700/873']
col_names = ['w_1', 'w_2', 'w_3', 'w_4', 'b']

train_errors = pd.DataFrame(train_err_2a, index=row_names, columns=['train error'])
test_errors = pd.DataFrame(test_err_2a, index=row_names, columns=['test error'])
weight_vectors = pd.DataFrame(weight_vector, index=row_names, columns=['w_1', 'w_2', 'w_3', 'w_4', 'b'])

frame = [train_errors, test_errors, weight_vectors]
output = pd.concat(frame, axis=1)  # round the data to four decimal places

# print out the results
print(output)

# save the results
# output.to_csv('C:/Users/nanji/OneDrive\桌面/CS6350 Machine Learning/Assignments/A4/A4Q2a.csv')



