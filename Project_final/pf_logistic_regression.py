# import the csv file
import numpy as np
import pandas as pd
import copy
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# enhance the resolution of the figure
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000

# ========================= training data pre-processing ==================
# import the training data
df = pd.read_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Project\FINAL_REPORT\processed_data/train_p.csv', header=None)

y_raw = (copy.deepcopy(df.iloc[:, -1])).to_numpy()
# set the last column to be either -1 and 1
for i in range(len(df)):
    if df.iloc[i, -1] == 0:
        df.iloc[i, -1] = -1

# fetch the ground truth of the train data
y = (copy.deepcopy(df.iloc[:, -1])).to_numpy()


# ========================== test data pre-processing ======================
# import the training data
df_t = pd.read_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                   r'\Project\FINAL_REPORT\processed_data/test_p.csv', header=None)

# set the last column to be either -1 and 1
for i in range(len(df_t)):
    if df_t.iloc[i, -1] == 0:
        df_t.iloc[i, -1] = -1


# fetch the ground truth of the test data
y_st = (copy.deepcopy(df_t.iloc[:, -1])).to_numpy()


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


# creat the training subset (20000 samples) and test subset (5000 samples) from the training data.
def subset_creator(dataset, k):
    num = dataset.shape[0]

    total_idx = np.arange(0, num)

    xx = np.arange(0, 5000)
    test_idx = xx + (k-1)*5000

    test_data = dataset.iloc[test_idx, :]

    test_idx = test_idx.tolist()
    total_idx = total_idx.tolist()
    fi = [i for i in total_idx if i not in test_idx]
    train_data = dataset.iloc[fi, :]

    return train_data, test_data


def lre_estimation(train_data, test_data, v, r0, d, mode='MAP', err_plot='inactive', xcorr='inactive'):
    """
    Description: The logistic regression estimation algorithm, composed of:
                (a) the maximum a posteriori estimation logistic regression algorithm
                (b) the maximum likelihood estimation logistic regression algorithm

    :param train_data: the training dataset in form of DataFrame, each row is a sample, the last column
    corresponds to the labels
    :param test_data: the test dataset in form of DataFrame, each row is a sample, the last column
    corresponds to the labels
    :param v: the variance of Gaussian distribution for p(w)
    :param r0：a hyper parameter for the learning rate r_t = r0/(1 + r0/d*t)
    :param d：a hyper parameter for the learning rate r_t = r0/(1 + r0/d*t)
    :param mode: the type of algorithm, maximum a posteriori (by default), or when mode = 'ML', it is
                the Maximum Likelihood algorithm.
    :param err_plot: a string argument to determine whether to plot the training errors and test errors for
        each itertion. By default, it is 'inactive'. When 'active', errors in each iteration will be ploted.

    :return:return (augmented weight vect, final train error, final test err)
    """
    # ======================== prepare the test data ===========================

    if xcorr == 'active':

        n0 = test_data.shape[1] - 1
        # fetch the first four columns of the train data and convert it to an array
        input_x0 = (copy.deepcopy(test_data.iloc[:, range(n0)]).to_numpy())

        # insert 1  at the beginning of each row of input_x
        x0 = np.insert(input_x0, 0, 1, axis=1)

        # fetch the last column, y, and convert it to a 2d array
        y0 = (copy.deepcopy(test_data.iloc[:, -1])).to_numpy()
        y0 = y0.reshape(-1, 1)

    else:
        # fetch the first four columns of the train data and convert it to an array
        x_test = copy.deepcopy(test_data)
        x_test1 = x_test.to_numpy()

        # insert 1  at the beginning of each row of input_x
        x0 = np.insert(x_test1, 0, 1, axis=1)

    # ========================prepare the training data ===========================
    # fetch the first four columns of the train data and convert it to an array
    n = train_data.shape[1]-1
    input_x = (copy.deepcopy(train_data.iloc[:, range(n)]).to_numpy())

    # insert 1  at the beginning of each row of input_x
    x = np.insert(input_x, 0, 1, axis=1)

    # fetch the last column, y, and convert it to a 2d array
    y = (copy.deepcopy(train_data.iloc[:, -1])).to_numpy()
    y = y.reshape(-1, 1)

    # ======================== logistic regression =================================

    # initialize w0, the augmented vector, w = [b, w1, w2, w3, w4]
    w = np.zeros((1, df.shape[1]))

    # number of training examples in each shuffle, i.e., the rows of the dataframe
    m = train_data.shape[0]

    # shuffle the data 50 times at one time to avoid for loops
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
    # r0 = 1
    # d = 0.001

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

    if err_plot == 'active':
        # plot the two error curves
        plt.plot(errs, label='training', c='k', linewidth=1.0)
        # plt.plot(errs0, '--', label='test', c='r', linewidth=1.0)
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
        return w, errs[-1],  # errs0[-1]
    else:
        train_predict = label_predict(x, w)
        err_train = err_calculation(train_predict, y)

        # predict the test results
        test_predict = label_predict(x0, w)
        if xcorr == 'active':
            err_test = err_calculation(test_predict, y0)
            return w, test_predict, round(err_train, 4), round(err_test, 4), train_predict
        else:
            return w, test_predict, round(err_train, 4), train_predict


"""
# perform the cross correlation, setting the values for the hyper parameter
val = [0.001, 0.01, 0.1, 1]

train_errors = np.zeros((4, 4))
test_errors = np.zeros((4, 4))

for r0 in val:
    for d in val:
        train_e = []
        test_e = []
        for k in np.arange(1, 6):
            res = subset_creator(df, k)
            sub_train = res[0]
            sub_test = res[1]
            results_map = lre_estimation(sub_train, sub_test, 1, r0, d, xcorr='active')
            train_e.append(results_map[2])
            test_e.append(results_map[3])

        mean_train_error = np.mean(np.array(train_e))
        mean_test_error = np.mean(np.array(test_e))

        r0_idx = val.index(r0)
        d_idx = val.index(d)

        train_errors[r0_idx, d_idx] = mean_train_error
        test_errors[r0_idx, d_idx] = mean_test_error


tr_err = pd.DataFrame(train_errors, columns=['0.001', '0.01', '0.1', '1'], index=['0.001', '0.01', '0.1', '1'])
te_err = pd.DataFrame(test_errors, columns=['0.001', '0.01', '0.1', '1'], index=['0.001', '0.01', '0.1', '1'])

print('The train errors are: ')
print(tr_err)
print('The test errors are: ')
print(te_err)
"""

"""
# choose the right variance
var = [0.01, 0.1, 1, 3, 5, 10, 100]
tr = []
te = []
tr0 = []
te0=[]
for v in var:
    for k in np.arange(1, 6):
        res = subset_creator(df, k)
        sub_train = res[0]
        sub_test = res[1]
        results_map = lre_estimation(sub_train, sub_test, 10, 0.01, 0.1, xcorr='active')  # change the r0, and d.
        tr0.append(results_map[2])
        te0.append(results_map[3])

    mean_train_error = np.mean(np.array(tr0))
    mean_test_error = np.mean(np.array(te0))
    tr.append(mean_train_error)
    te.append(mean_test_error)

print(tr)
print(te)
err_v = pd.DataFrame(np.array([tr, te]), columns=[0.01, 0.1, 1, 3, 5, 10, 100], 
                     index=['Training error', 'Test error'])
print(err_v)
"""

# ==================================================================================================
"""
# confirm convergence by ploting the training error from the whole training dataset
results_map = lre_estimation(df, df_t, 0.01, 0.001, 0.1, err_plot='active')
# the training error
print(results_map[1])
"""

"""
# compute the confusion matrix by sklearn
# training_confusion = (lre_estimation(df, df_t, 0.01, 0.001, 0.1))[3] # for OHE dataset
training_confusion = (lre_estimation(df, df_t, 0.01, 0.01, 0.1))[3] # for bin counting dataset
for i in range(df.shape[0]):
    if training_confusion[i] == -1:
        training_confusion[i] = 0

tr_conf_mx = confusion_matrix(y_raw, training_confusion)
print(tr_conf_mx)
disp = ConfusionMatrixDisplay(confusion_matrix=tr_conf_mx)
disp.plot()
plt.show()
"""


# ============================ predicting the labels for test data =====================================
# Predict the labels for the test dataset using the optimal hyper parameters

#results_map = lre_estimation(df, df_t, 0.01, 0.01, 0.1)  # for the bin counting dataset
results_map = lre_estimation(df, df_t, 0.01, 0.001, 0.1)  # for the OHE dataset

predict_test = results_map[1]
# convert the -1 to 0
for i in range(df_t.shape[0]):
    if predict_test[i] == -1:
        predict_test[i] = 0
print('The prediction for test data is: ')
print(results_map[1])
print('The training error is: ')
print(results_map[2])
predict_final = pd.DataFrame(predict_test)


"""
# save prediction for the BC case
predict_final.to_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                       r'\Project\FINAL_REPORT\processed_data/test_BC_lr.csv', index=False, header=False)

"""

# save prediction for the OHE case
predict_final.to_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                       r'\Project\FINAL_REPORT\processed_data/test_OHE_lr.csv', index=False, header=False)






