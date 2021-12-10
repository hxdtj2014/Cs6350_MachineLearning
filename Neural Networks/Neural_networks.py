# import the csv file
import numpy as np
import pandas as pd
import copy
from random import shuffle
import matplotlib.pyplot as plt

# enhance the resolution of the figure
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000

# ========================== training data pre-processing ===================================
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

# ========================== test data pre-processing =========================================
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

# ========================== define sigmoid function and its derivative ========================
def sigmoid(x0):
    return 1 / (1 + np.exp(-x0))
# define the derivative of sigmoid function


def sigmoid_d(sigma0):
    return sigma0*(1 - sigma0)
# ==============================define the forward pass algorithm ==============================
def sgn_func(numbers):
    """
    :param numbers: a number to be determined its sign
    :return: the sgn of the number
    """
    if float(numbers) >= 0:
        return 1
    else:
        return -1

# error calculation between the predicted labels and the true labels
def err_calculation(true_label, predicted_label):
    """
    :param true_label: the ground truth, in an array
    :param predicted_label:
    :return:
    """
    err = np.sum((true_label != predicted_label))/len(predicted_label)
    return err


def forward_pass(weights_array, input_x):
    """
    Description: to calculate the node-value, i.e., z at each node
    :param weights_array: the weight matrices for all the layers, of the form
    of a 2D array, stored in a list, in order of the layer, i.e., layer 1, layer 2, and layer 3.

    :param input_x: a single sample stored in a 1D array of dimension of 1 by M.
    The sample is augmented, [1 x1 x2 ... x_M-1]

    :return: z, a matrix of size of 2 by M [[1, z11, z12, ...], [1, z21, z22, ...]]
             y0, the computed output
    """
    # take the size of weight array
    m= weights_array[0].shape[1]

    # set the output matrix, put like [z10 z11 z12]
    z = np.ones((2, m+1))


    # compute the nodes at layer 1
    product_1 = np.dot(input_x, weights_array[0])

    # print(sigmoid(product_1))
    z[0, 1:] = sigmoid(product_1)

    # compute the nodes at layer 2
    product_2 = np.dot(z[0, :], weights_array[1])
    z[1, 1:] = sigmoid(product_2)


    y0 = np.dot(z[1, :], weights_array[2])

    # the node for the hidden layers, i.e., layer 1 and layer 2 and the computed label y
    return z, y0


# compute the derivatives of L w.r.t all the weights w_ij
def backward_prop(weights_array, input_x,  y_star):
    """
    :param weights_array: [weights matrix for layer 1, weights matrix for layer 2]
    :param input_x:a single sample stored in a 1D array of dimension of 1 by M.
                    The sample is augmented, [1 x1 x2 ... x_M-1]
    :param y_star: the ground truth label
    :return: the derivative of Loss w.r.t the weights, stored in the same format as
            the weight_array.
    """
    forward_output = forward_pass(weights_array, input_x)
    y_computed = forward_output[1]
    z_nodes = forward_output[0]

    # create a list to store the derivatives for all the layers
    deriv_weights = [i0 for i0 in range(3)]
    # compute the derivative of loss w.r.t the output
    ly = y_computed - y_star
    # compute the derivative of loss w.r.t the weights at the third layer
    deri_layer3 = ly*z_nodes[1, :]

    # compute the derivative of loss w.r.t the weights at the second layer
    z_layer2 = z_nodes[1, 1:]
    w_3 = weights_array[2][1:]
    coeff_2 = ly*w_3*sigmoid_d(z_layer2)
    deri_layer2 = np.multiply((z_nodes[0, :])[:, None], coeff_2[None, :])

    # compute the derivative of loss w.r.t the weights at the first layer
    c0 = ly*weights_array[2][1:]*sigmoid_d(z_nodes[1, 1:])
    c1 = (weights_array[1])[1:, :].T
    c2 = np.dot(c0, c1)
    c3 = c2*sigmoid_d(z_nodes[0, 1:])
    deri_layer1 = np.multiply(input_x[:, None], c3[None, :])

    # store the deri_layer1 and deri_layer2 in order in a list

    deriv_weights[0], deriv_weights[1], deriv_weights[2] = \
         deri_layer1,      deri_layer2,      deri_layer3

    # return the derivative of loss w.r.t weights in a list, corresponds to layer 1, 2, and 3
    return deriv_weights

'''
# verify problem 2
ix = np.array([1, 1, 1])
# weights for layer 3
wy = np.array([-1, 2, -1.5])
weights = []
# weights for layer 1
weights.append( (np.array([[-1, -2, -3], [1, 2, 3]])).T)
# weights for layer 2
weights.append((np.array([[-1, -2, -3], [1, 2, 3]])).T)
# weights for layer 2
weights.append(wy)
hh = forward_pass(weights, ix)
print(hh)
# verify problem 3
dv = backward_prop(weights, ix, 1)
print(dv)
'''


def neural_networks_trainer(parameter_vect, dataset, plot_option='NO', weights_0_type='NO'):
    """
    Descriptions: This function uses three layers for training the weights, layer 1 and layer 2
    are fixed to have the same number of nodes, layer 3 is the output layer.

    :param parameter_vect: a list containing the parameters in order of

                            [hl_width, r0, d, shuffle_times],

                            hl_width: the width of hidden layers, to be specified
                            r0: gamma0, a parameter in the formula for the learning rate
                            d: a parameter in the formula for the learning rate
                            shuffle_times: the times to shuffle the dataset.

    :param dataset: the input, training data in a DataFrame [x1, x2, x3, x4,y] for each row.
    :param weights_0_type: an integer to select the way of initializing the weights
                            weights_0_type = 'NO' is by default, using probability density of
                                             Gaussian distribution for the weights_0.
                            weights_0_type = 'YES'  assuming the weights_0 to be zeros

    :param plot_option: if plot_option == 'YES', the function will plot the loss and training error for each iteration.
                        else:, the function will not plot nor compute the loss and training error for every step.

    :return: the trained weights, in the same form of weight_arrays defined in the
            forward pass algorithm and back propagation algorithm.
    """
    hl_width = parameter_vect[0]
    r0 = parameter_vect[1]
    d = parameter_vect[2]
    shuffle_times = parameter_vect[3]

    # count the number of samples in the dataset, i.e., the number of rows
    num_samples = dataset.shape[0]
    # count the number of inputs
    num_x = dataset.shape[1]

    if weights_0_type == 'YES':
        # initialize the weights by assuming all to be 0
        weights_0 = [j0 for j0 in range(3)]
        weights_0[0], weights_0[1], weights_0[2] = \
            np.zeros((num_x, hl_width - 1)), \
            np.zeros((hl_width, hl_width - 1)), \
            np.zeros((hl_width,))
    else:
        # initialize the weights from Gaussian distribution, which is set by default
        weights_0 = [j0 for j0 in range(3)]
        weights_0[0], weights_0[1], weights_0[2] = \
        np.random.randn(num_x, hl_width - 1), \
        np.random.randn(hl_width, hl_width - 1), \
        np.random.randn(hl_width, )

    x_input0 = (df.iloc[:, range(4)]).to_numpy()
    y_gt = (df.iloc[:, -1]).to_numpy()

    # insert a column with ones in front of the [x1 x2 x3 x4] dataframe
    x_input = np.insert(x_input0, 0, 1, axis=1)

    # shuffle the samples in the dataset
    indices = []
    for times in range(shuffle_times):
        # shuffle the row indices
        row_idx = [r_idx for r_idx in range(num_samples)]
        shuffle(row_idx)
        indices = indices + row_idx

    # train the weights by stochastic gradient descent method
    iteration_times = 0

    # compute the training errors as well as the loss at each iteration
    errors = []
    loss = []
    for j in indices:
        # compute the gradients for all the weights
        gradients_1 = backward_prop(weights_0, x_input[j], y_gt[j])
        # learning rate
        iteration_times = iteration_times + 1
        r = r0 / (1 + r0 / d * iteration_times)

        if iteration_times/shuffle_times/num_samples*100 % 10 <= 0.0001:
            print(str(round(iteration_times/shuffle_times/num_samples, 1)*100) + '% is completed')

        # update the weights for each layer through sgd
        weights_0[0] = weights_0[0] - r * gradients_1[0]
        weights_0[1] = weights_0[1] - r * gradients_1[1]
        weights_0[2] = weights_0[2] - r * gradients_1[2]

        if plot_option == 'YES':
            # compute the loss and training errors at each iteration, if required
            y_predicted0 = [forward_pass(weights_0, x_input[k])[1] for k in range(num_samples)]
            y_predicted = np.array(y_predicted0)
            ys = np.sign(y_predicted0)
            loss0 = np.sum((y_gt - y_predicted)**2)/2
            loss.append(loss0)
            err = err_calculation(y_gt, ys)
            errors.append(err)

    if plot_option == 'YES':
        plt.plot(loss)
        plt.xlabel('Iteration times')
        plt.ylabel('Loss')
        plt.title('Loss vs iteration times')
        plt.show()
        plt.plot(errors)
        plt.xlabel('Iteration times')
        plt.ylabel('Training errors')
        plt.title('Training errors vs iteration times')
        plt.show()
        return weights_0, errors[-1]

    else:
        y_predicted0 = [forward_pass(weights_0, x_input[k])[1] for k in range(num_samples)]
        ys1 = np.sign(y_predicted0)
        train_err = err_calculation(y_gt, ys1)
        return weights_0, train_err


# ================== check the convergence of assuming r0 = 0.01, d = 10 ====================================
# compute the test errors
trained_results = neural_networks_trainer([25, 0.01, 10, 10], df, weights_0_type='NO', plot_option='YES')


'''
# ================== calculate the training and test errors under different width 2(b) ======================
width = [5, 10, 25, 50, 100]
training_errors = []
test_errors = []
for wid in width:
    trained_results = neural_networks_trainer([wid, 0.01, 10, 100], df)  # set the shuffle time to be 100
    trained_weights = trained_results[0]
    training_errors.append(round(trained_results[1],4))
    x_input1 = (df_t.iloc[:, range(4)]).to_numpy()
    x_input2 = np.insert(x_input1, 0, 1, axis=1)
    ys = []
    y2 = []
    for i in range(len(x_input2)):
        y_predicted = forward_pass(trained_weights, x_input2[i])[1]
        ys = ys + [sgn_func(y_predicted)]
        y2.append(y_predicted)
    err = err_calculation(y_st, ys)
    test_errors.append(round(err,4))

error_list = [training_errors, test_errors]
err_df = pd.DataFrame(np.array(error_list), columns=['5', '10', '25', '50', '100'],
                      index=['training errors', 'test errors'])
print('\nThe errors are:')
print(err_df)
'''

'''
# ================== calculate the training and test errors under different width 2(c) ======================

width = [5, 10, 25, 50, 100]
training_errors = []
test_errors = []
for wid in width:
    # set the w0 to be zero vector
    trained_results = neural_networks_trainer([wid, 0.01, 10, 100], df, weights_0_type='YES', plot_option='NO')
    trained_weights = trained_results[0]
    training_errors.append(round(trained_results[1],4))
    x_input1 = (df_t.iloc[:, range(4)]).to_numpy()
    x_input2 = np.insert(x_input1, 0, 1, axis=1)
    ys = []
    y2 = []
    for i in range(len(x_input2)):
        y_predicted = forward_pass(trained_weights, x_input2[i])[1]
        ys = ys + [sgn_func(y_predicted)]
        y2.append(y_predicted)
    err = err_calculation(y_st, ys)
    test_errors.append(round(err,4))

error_list = [training_errors, test_errors]
err_df = pd.DataFrame(np.array(error_list), columns=['5', '10', '25', '50', '100'],
                      index=['training errors', 'test errors'])
print('\nThe errors are:')
print(err_df)
'''

'''
# PLOT the loss and training errors for each iteration, setting width = 5, all intial weights  = 0
w0_initial_checking = neural_networks_trainer([5, 0.01, 10, 50], df, weights_0_type='YES', plot_option='YES')
'''
