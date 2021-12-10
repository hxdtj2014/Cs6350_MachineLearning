# import the csv file
import numpy as np
import pandas as pd
import copy
from random import shuffle
import matplotlib.pyplot as plt

# enhance the resolution of the figure
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000

# ========================= training data pre-processing ==================
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

# ========================== Logistic regression ========================


def label_predict(data, w):
    """
    :param data: a dataframe to predict
    :param w: the weight vector after training the train data,
              the bias parameter b is folded into the end of the weight vector.
    :return: the prediction in a 2d array (n,1)
    """
    h = np.dot(data, w.T)
    predict = (np.sign(h))
    return predict


# error calculation between the predicted labels and the true labels
def err_calculation(true_label, predicted_label):
    """
    :param true_label: the ground truth, in an array
    :param predicted_label:
    :return:
    """
    err = np.sum((true_label != predicted_label))/len(predicted_label)
    return err


# define the sigmoid function
def sigmoid(in_x):
    return .5 * (1 + np.tanh(.5 * in_x))


def lre_estimation(train_data, test_data, v, mode='MAP', err_plot='inactive'):
    """
    Description: The logistic regression estimation algorithm, composed of:
                (a) the maximum a posteriori estimation logistic regression algorithm
                (b) the maximum likelihood estimation logistic regression algorithm

    :param train_data: the training dataset in form of DataFrame, each row is a sample, the last column
    corresponds to the labels
    :param test_data: the test dataset in form of DataFrame, each row is a sample, the last column
    corresponds to the labels
    :param v: the variance of Gaussian distribution for p(w)
    :param mode: the type of algorithm, maximum a posteriori (by default), or when mode = 'ML', it is
                the Maximum Likelihood algorithm.
    :param err_plot: a string argument to determine whether to plot the training errors and test errors for
        each itertion. By default, it is 'inactive'. When 'active', errors in each iteration will be ploted.

    :return:return (augmented weight vect, final train error, final test err)
    """
    # ======================== prepare the test data ===========================
    # fetch the first four columns of the train data and convert it to an array
    input_x0 = (copy.deepcopy(test_data.iloc[:, range(4)]).to_numpy())

    # insert 1  at the beginning of each row of input_x
    x0 = np.insert(input_x0, 0, 1, axis=1)

    # fetch the last column, y, and convert it to a 2d array
    y0 = (copy.deepcopy(test_data.iloc[:, -1])).to_numpy()
    y0 = y0.reshape(-1, 1)

    # ========================prepare the training data ===========================
    # fetch the first four columns of the train data and convert it to an array
    input_x = (copy.deepcopy(train_data.iloc[:, range(4)]).to_numpy())

    # insert 1  at the beginning of each row of input_x
    x = np.insert(input_x, 0, 1, axis=1)

    # fetch the last column, y, and convert it to a 2d array
    y = (copy.deepcopy(train_data.iloc[:, -1])).to_numpy()
    y = y.reshape(-1, 1)

    # ======================== logistic regression =================================

    # initialize w0, the augmented vector, w = [b, w1, w2, w3, w4]
    w = np.zeros((1, 5))

    # number of training examples in each shuffle, i.e., the rows of the dataframe
    m = train_data.shape[0]

    # shuffle the data 100 times at one time to avoid for loops
    epochs = 100
    indices = []
    for times in range(epochs):
        # shuffle the row indices
        row_idx = [r_idx for r_idx in range(m)]
        shuffle(row_idx)
        indices.append(row_idx)

    h = []  # the shuffled row indices max_epoch*len(df)
    for zz in range(len(indices)):
        h = h + indices[zz]

    # construct the learning rate by r_t = r0/(1 + r0/d*t)
    # r0 and d needs to be tuned, which is accomplished through trials
    r0 = 0.001
    d = 1

    # set the initial iteration times
    t = 0
    errs0 = []
    errs = []
    for j in h:
        # count the total iteration times
        t = t + 1

        if t/m/epochs*100 % 10 <= 0.0001:
            print(str(round(t/m/epochs, 2)*100) + ' % is completed')

        # set the learning rate r_t
        r_t = r0 / (1 + r0 / d * t)

        if mode == 'ML':
            # compute the gradient for maximum likelihood estimation logistic regression algorithm
            grad = m * (sigmoid(y[j] * np.dot(w, x[j])) - 1) * y[j] * x[j]
        else:
            # compute the gradient for he maximum a posteriori estimation logistic regression algorithm
            grad = m * (sigmoid(y[j] * np.dot(w, x[j])) - 1) * y[j] * x[j] + 1 / v * w

        # update the augmented weight vector
        w = w - r_t*grad

        if err_plot == 'active':
            # plot the training errors vs iteration times
            train_predict = label_predict(x, w)
            err_train = err_calculation(train_predict, y)
            errs.append(round(err_train, 4))

            # plot the test errors vs iteration times
            test_predict = label_predict(x0, w)
            err_test = err_calculation(test_predict, y0)
            errs0.append(round(err_test, 4))

    if err_plot == 'active':
        # plot the two error curves
        plt.plot(errs, label='training', c='k', linewidth=1.0)
        plt.plot(errs0, '--', label='test', c='r', linewidth=1.0)
        plt.xlabel('Iteration times', fontname="serif", weight="bold")
        plt.ylabel('Errors', fontname="serif", weight="bold")
        plt.xticks(fontname="serif", weight="bold")
        plt.yticks(fontname="serif", weight="bold")
        if mode == 'ML':
            plt.title('ML', fontname="serif", weight="bold")
        else:
            plt.title('MAP', fontname="serif", weight="bold")
        plt.legend()
        plt.show()
        # return (augmented weight vect, final train error, final test err)
        return w, errs[-1], errs0[-1]
    else:
        train_predict = label_predict(x, w)
        err_train = err_calculation(train_predict, y)

        test_predict = label_predict(x0, w)
        err_test = err_calculation(test_predict, y0)

        # return (augmented weight vect, final train error, final test err)
        return w, round(err_train, 4), round(err_test, 4)


# convergence checking when r0 = 0.001, d=1.0 for the learning rate, if necessary.
map_checking = lre_estimation(df, df_t, 100, err_plot='active')
ml_checking = lre_estimation(df, df_t, 100, mode='ML', err_plot='active')

"""
# ======================= MAP error calculation 3(a)=============================
variance = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
map_train_err = []
map_test_err = []
for var in variance:
    results_map = lre_estimation(df, df_t, var)
    map_train_err.append(results_map[1])
    map_test_err.append(results_map[2])


error_list = [map_train_err, map_test_err]
map_df = pd.DataFrame(np.array(error_list), columns=['0.01', '0.1', '0.5', '1', '3', '5', '10', '100'],
                      index=['training errors', 'test errors'])
print('For MAP, the errors are:')
print(map_df)
"""
"""
# ======================== ML error calculation 3(b)==============================
variance = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
ml_train_err = []
ml_test_err = []
for var in variance:
    results_ml = lre_estimation(df, df_t, var, mode='ML')
    ml_train_err.append(results_ml[1])
    ml_test_err.append(results_ml[2])

error_list = [ml_train_err, ml_test_err]
ml_df = pd.DataFrame(np.array(error_list), columns=['0.01', '0.1', '0.5', '1', '3', '5', '10', '100'],
                      index=['training errors', 'test errors'])
print('\nFor ML, the errors are:')
print(ml_df)
"""




