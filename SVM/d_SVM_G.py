# import the csv file
import numpy as np
import copy
import scipy.optimize as sop
import pandas as pd
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

# =============================== optimization ==========================================
# the hyper parameter c is set as below
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


def label_predict_g(k_matrix, a, b):
    """
    :param k_matrix: the kernel matrix, n by n
    :param a: the product of alpha[i]*y[i] in an array, a vector
    :param b: the bias parameter from the optimization results
    :return: the prediction in a list
    """

    predict = []
    for j2 in range(k_matrix.shape[1]):
        part1 = np.dot(k_matrix[:, j2], a)
        sgn = part1 + b
        predict.append(sgn_func(sgn))
    return predict


# error calculation between the predicted labels and the true labels
def err_calculation(true_label, predicted_label):
    count = 0  # correct predication count
    for k in range(len(true_label)):
        if true_label[k] != predicted_label[k]:
            count += 1
    return count / float(len(true_label))


# ==================== construct the kernel matrix for the training data ========================
r = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]

# pick a gamma from the list r
def gaussian_kernel_maker(dataset1, dataset2, gamma1):
    """
    Description: to make a gaussian kernel between two vectors, in form of nd array
    :param dataset1: the training dataset, a DataFrame
    :param dataset2: the training dataset or test dataset, a DataFrame
    :param gamma1: the variance in gaussian kernel, sigma^2
    :return:
    """
    cols = dataset1.shape[1] - 1
    x1 = copy.deepcopy((dataset1.iloc[:, range(cols)]).to_numpy())
    x2 = copy.deepcopy((dataset2.iloc[:, range(cols)]).to_numpy())
    k_mat1 = np.ndarray([x1.shape[0], x2.shape[0]])
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            norm2 = (np.linalg.norm(x1[i]-x2[j]))**2
            k_mat1[i, j] = np.exp(-norm2/gamma1)
    return k_mat1


# ==================== define the objective function ========================
def obj_func(alpha, k_mat2):
    """
    :param alpha: a 1D array, which needs to be optimized
           x is an nd array, y is a 1D array. Both x and y are fixed
    :return: a scalar value of the obj_func
    """
    ay = alpha*y  #  alpha[i]*y[i] as a vector
    # obj_func = 0.5*(ay^T)k_mat*ay - sum(alpha)
    result_obj = 0.5 * np.dot(ay.T, np.dot(k_mat2, ay)) - np.sum(alpha)
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
def dual_svm(hp, objective_function, dataframe, gamma0):
    """
    :param hp: the hyperparameter, a number
    :param objective_function: the function to be optimized
    :param dataframe: the n samples in form of a dataframe
    :param gamma0: the variance in gaussian kernel, sigma^2
    :return: a: the optimized alpha[i]*y[i] in a array,
             b: the bias (a mean)
    """
    x1 = (dataframe.iloc[:, range(4)]).to_numpy()
    y1 = (copy.deepcopy(df.iloc[:, -1])).to_numpy()
    num_of_samples1 = np.shape(x1)[0]

    # specify the bounds of alpha, num_of_samples of them in tuple
    bnds = [(0, hp)]*num_of_samples1

    # set the initial guess to be a zero vector
    alpha_0 = np.array([0]*num_of_samples1)

    k_mat0 = gaussian_kernel_maker(dataframe, dataframe, gamma0)
    # call the minimization from scipy
    result = sop.minimize(lambda alpha: objective_function(alpha, k_mat0),
                          alpha_0, method='SLSQP', bounds=bnds, constraints=cons)

    # fetch the optimized alpha, a 1D array
    a = result.x

    # fetch the biased parameter b based on the results of minimization
    ay1 = a*y1
    kernel_mat = gaussian_kernel_maker(dataframe, dataframe, gamma0)
    bia = [(y1[j] - np.dot(ay1, kernel_mat[j])) for j in range(num_of_samples1)]
    b = np.mean(bia)
    # return the vector of alpha[i]*y[i], the mean of bias parameter b
    return ay1, b


# ================== compute the training errors and test errors with different c and gamma =============
results0 = np.zeros((len(r), len(c)))
results_b = pd.DataFrame(results0)
results_c = pd.DataFrame(results0)

p = 0
total_times = len(r)*len(c)
for i in range(len(r)):
    for j in range(len(c)):
        gamma = r[i]
        hyper_p = c[j]
        m = dual_svm(hyper_p, obj_func, df, gamma)
        a1 = m[0]
        b1 = m[1]

        k_mat = gaussian_kernel_maker(df,df, gamma)
        hh = label_predict_g(k_mat, a1, b1)
        errs = err_calculation(y_s, hh)
        # print the training error

        # print the test error
        k_mat_t = gaussian_kernel_maker(df,df_t, gamma)
        hh_t = label_predict_g(k_mat_t, a1, b1)
        errs1 = err_calculation(y_st, hh_t)

        results_b.iloc[i,j] = '( ' + str(100 * np.round(errs, 4)) + '% ,' \
                            + str(100 * np.round(errs1,4)) +'%)'

        # count the non-zero alpha[i]
        results_c.iloc[i, j] = (abs(a1)>1/10**10
                                ).sum() #np.count_nonzero(a1)

        # monitor the computing progress
        p = p + 1
        progress = 100 * np.round(p/total_times, 3)
        print(str(progress) + '% is completed')

row_names = ['r = 0.01', '0.1', '0.5', '1',
             '2', '5', '10', '100']
col_names = ['c = 100/873', 'c = 500/873', 'c = 700/873']

# change the column names and row names
results_b.index = row_names
results_b.columns = col_names

results_c.index = row_names
results_c.columns = col_names

print('The (train error, test error) for different cases are:')
print(results_b)

print('The number of support vectors is:')
print(results_c)
# ===================================2c part 2==========================
# compute the overlapped support vectors
def overlap_calculation(x1, x2):
    count = 0  # correct predication count
    for k in range(len(x1)):
        if x1[k]*x2[k] !=0 and x1[k] == x2[k]:
            count += 1
    return count

alpha_y = []
for j in range(len(r)):
    m = dual_svm(c[1], obj_func, df, r[j])
    alpha_y.append(m[0])

xx = pd.DataFrame(alpha_y)
ov = []
for j in range(len(r)-1):
    a1 = (xx.iloc[j, :]).round(decimals=6) # control the accuracy
    a2 = (xx.iloc[j+1, :]).round(decimals=6)
    ov0 = overlap_calculation(a1, a2)
    ov.append(ov0)

print('When c = 500/873, the number of overlapped vectors with r_i and r_i+1 are')
print(ov)
