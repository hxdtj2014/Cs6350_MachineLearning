# import the csv file
import numpy as np
import pandas as pd
import copy
import scipy.optimize as sop

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

# fetch the first four columns of the dataframe and convert it to an array
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
# =============================== optimization ==========================================
# the hyperparameter c is set as below
c = [100/873, 500/873, 700/873]


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
    count = 0  # correct predication count
    for k in range(len(true_label)):
        if true_label[k] != predicted_label[k]:
            count += 1
    return count / float(len(true_label))


# ==================== define the objective function ========================

# make a linear kernel, composed of the inner product between two row vectors
k_mat = np.ndarray([num_of_samples, num_of_samples])
for i in range(num_of_samples):
    for j in range(num_of_samples):
        k_mat[i, j] = np.dot(x[i], x[j])


def obj_func(alpha):
    """
    :param alpha: a 1D array, which needs to be optimized
           x is an nd array, y is a 1D array. Both x and y are fixed
    :return: a scalar value of the obj_func
    """
    ay = alpha*y  #  alpha[i]*y[i] in a vector
    # obj_func = 0.5*(ay^T)k_mat*ay - sum(alpha)
    result_obj = 0.5 * np.dot(ay.T, np.dot(k_mat, ay)) - np.sum(alpha)
    return result_obj


# ============================ define the constraint ==========================
def eq_constraint(alpha):
    """
    :param alpha: a 1D array
            y is a 1D array, which is fixed
    :return: the dot product of alpha and y
    """
    res_constraint = np.dot(alpha, y)
    return res_constraint


# Equality constraint means that the constraint function result is to be zero
cons = {'type': 'eq', 'fun': eq_constraint}


# ============== call the minimization function from scipy package ===============
def dual_svm(hp, objective_function, dataframe):
    """
    :param hp: the hyperparameter, a number
    :param objective_function: the function to be optimized
    :param dataframe: the n samples in form of a dataframe
    :return: w_aug: [w_1, w_2, ..., w_n, b] with the bias parameter
    """

    x1 = (dataframe.iloc[:, range(4)]).to_numpy()
    y1 = (copy.deepcopy(df.iloc[:, -1])).to_numpy()
    num_of_samples1 = np.shape(x1)[0]

    # specify the bounds of alpha, num_of_samples of them in tuple
    bnds = [(0, hp)]*num_of_samples1

    # set the initial guess to be a zero vector
    alpha_0 = np.array([0]*num_of_samples1)

    # call the minimization from scipy
    result = sop.minimize(objective_function, alpha_0,
                          method='SLSQP', bounds=bnds, constraints=cons)

    # fetch the optimized alpha, an 1D array
    a = result.x

    # fetch the weight vector based on the minimization results
    w = np.sum([(a[i1]*y1[i1])*x1[i1] for i1 in range(num_of_samples1)], axis=0)

    # fetch the biased parameter b
    bias = [(y1[j] - np.dot(w, x1[j])) for j in range(num_of_samples1)]
    b = np.mean(bias)

    # fold the bias parameter into the weight vector such that w = [w_1, w_2, ..., w_n, b]
    w_aug = np.append(w, b)

    return w_aug


# ========================== compute the training errors and test errors with different c =============
weight_vect = []
train_errs = []
test_errs = []

w0 = dual_svm(c[0], obj_func, df)
weight_vect.append(w0)
w1 = dual_svm(c[1], obj_func, df)
weight_vect.append(w1)
w2 = dual_svm(c[2], obj_func, df)
weight_vect.append(w2)
# print(weight_vect)


# training error
lp0 = label_predict(df, w0)
errs0 = err_calculation(y_s, lp0)
train_errs.append(errs0)
# print(errs0)

lp1 = label_predict(df, w1)
errs1 = err_calculation(y_s, lp1)
train_errs.append(errs1)

lp2 = label_predict(df, w2)
errs2 = err_calculation(y_s, lp2)
train_errs.append(errs2)

# print(train_errs)

# test error
lp3 = label_predict(df_t, w0)
errs0 = err_calculation(y_st, lp3)
test_errs.append(errs0)
# print(errs0)


lp4 = label_predict(df_t, w1)
errs1 = err_calculation(y_st, lp4)
test_errs.append(errs1)

lp5 = label_predict(df_t, w2)
errs2 = err_calculation(y_st, lp5)
test_errs.append(errs2)


# construct a dataframe to show the predicted results
row_names = ['100/873', '500/873', '700/873']
col_names = ['w_1', 'w_2', 'w_3', 'w_4', 'b']

train_errors = pd.DataFrame(train_errs, index=row_names, columns=['train error'])
test_errors = pd.DataFrame(test_errs, index=row_names, columns=['test error'])
weight_vectors = pd.DataFrame(weight_vect, index=row_names, columns=['w_1', 'w_2', 'w_3', 'w_4', 'b'])

frame = [train_errors, test_errors, weight_vectors]
output = (pd.concat(frame, axis=1)).round(4)  # round the data to four decimal places
print(output)






