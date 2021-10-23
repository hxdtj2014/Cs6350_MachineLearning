# AdaBoost Algorithm A2Q2(a)
# import the csv file
import numpy as np
import pandas as pd
from numpy import ma
import pprint
import copy
import matplotlib.pyplot as plt

# =================== Pre-processing the training data=================================

# read the data as a dataframe
df = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面'
                 r'\CS6350 Machine Learning\Assignments'
                 r'\Assignment2\A2Q2\train.csv', header=None)

num_set = [0, 5, 9, 11, 12, 13, 14] # indices of the numerical attributes

# convert the numerical values to the binary values, i.e., 'yes', or 'no'.
rows = df.shape[0] # rows of the dataframe
cols = df.shape[1] # cols of the dataframe
median_of_num = [] # the median of the numerical features, serving as the threshold
# replace the numerical features by the binary ones
for i in num_set:
    t = df.loc[:, i].median()
    median_of_num.append(t)
    for j in range(rows):
        if df.loc[j, i] >= t:
            df.loc[j, i] = 'yes'
        else:
            df.loc[j, i] = 'no'

# convert the output (the last column of the df) of the dataframe
# to be either 1 or -1 (binaries), corresponding to 'yes' or 'no', respectively.
for i in range(rows):
    if df.loc[i, 16] == 'yes':
        df.loc[i, 16] = 1
    else:
        df.loc[i, 16] = -1

# print(df) # check the dataframe

# ============================ pre-processing the test data============================

# read the data as a dataframe
df_t = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面'
                 r'\CS6350 Machine Learning\Assignments'
                 r'\Assignment2\A2Q2\test.csv', header=None)

num_set = [0, 5, 9, 11, 12, 13, 14] # indices of the numerical attributes

# convert the numerical values to the binary values, i.e., 'yes', or 'no'.
rows = df_t.shape[0] # rows of the dataframe
cols = df_t.shape[1] # cols of the dataframe
median_of_num = [] # the median of the numerical features, serving as the threshold
# replace the numerical features by the binary ones
for i in num_set:
    t = df_t.loc[:, i].median()
    median_of_num.append(t)
    for j in range(rows):
        if df_t.loc[j, i] >= t:
            df_t.loc[j, i] = 'yes'
        else:
            df_t.loc[j, i] = 'no'

# convert the output (the last column of the df) of the dataframe
# to be either 1 or -1 (binaries), corresponding to 'yes' or 'no', respectively.
for i in range(rows):
    if df_t.loc[i, 16] == 'yes':
        df_t.loc[i, 16] = 1
    else:
        df_t.loc[i, 16] = -1


# =========================decision trees========================================

# The attribute names to the imported dataframe
attr_names = ['age', 'job', 'martial', 'education',
              'default', 'balance', 'housing','loan',
              'contact', 'day', 'month', 'duration',
              'campaign', 'pdays', 'previous', 'poutcome']

Attr_dict = {'age': ['yes', 'no'],
             'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                     'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
             'martial': ['married', 'divorced', 'single'],
             'education': ['unknown', 'secondary', 'primary', 'tertiary'],
             'default': ['yes', 'no'],
             'balance': ['yes', 'no'],
             'housing': ['yes', 'no'],
             'loan': ['yes', 'no'],
             'contact': ['unknown', 'telephone', 'cellular'],
             'day': ['yes', 'no'],
             'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
             'duration': ['yes', 'no'],
             'campaign': ['yes', 'no'],
             'pdays': ['yes', 'no'],
             'previous': ['yes', 'no'],
             'poutcome': ['unknown', 'other', 'failure', 'success']}


# fetch the column number by attr_name
def attr_name_idx(idx):
    attr_names1 = ['age', 'job', 'martial', 'education',
                  'default', 'balance', 'housing', 'loan',
                  'contact', 'day', 'month', 'duration',
                  'campaign', 'pdays', 'previous', 'poutcome']
    func_val = attr_names1.index(idx)  # column number for each attribute
    return func_val


def adaboost_ig(df, D_t):

    """
    Description: the ig calculation considers the weights in AdaBoost algorithm
        On input:
        df:  a M by N matrix, the last column is the labels
        other columns are attributes.
        D_t: the weights for the all the examples in round t iteration

        On output:
        a list contains the calculated information gain
        in the order that the attributes locate in the
        columns of the input matrix.
        In such form [weighted entropy_based IG]
    """
    # consider the slice of the a dataframe where the indices are not continuous
    df.reset_index(inplace = True, drop = True)
    # calculate the number of the examples
    length = len(df)
    # compute the weighted average entropy
    wa_entropy = []
    # find the column index of the label
    df_cols = df.shape[1] - 1

    # positive portion of the labels
    pos_label = [i for i in df.loc[:, df_cols] if i > 0]
    pos = sum(pos_label)/length
    neg = 1 - sum(pos_label)/length

    H_total = -pos*ma.log2(pos) - neg*ma.log2(neg)

    for ix in range(df_cols):
        b = df.loc[:, [ix, df_cols]]  # fetch the each attribute column together with the output

        attr_val_col = b.loc[:, ix]
        true_labels = b.loc[:, df_cols]  # true_labels for all examples
        attr_val = set(attr_val_col)  # the attribute values in an attribute

        weights = [c*d for c, d in zip(true_labels, D_t)]  # all the weights with + or -

        h1 = []
        for ixx in attr_val:
            weights_idx = attr_val_col[attr_val_col == ixx].index.tolist()

            weights_val= [weights[i] for i in weights_idx]

            port = [abs(i) for i in weights_val]  # portion of ixx in the whole dataset
            p =sum(port) # sum of all weights of examples in attr_val, ixx

            pos_val_list = [x for x in weights_val if x > 0]
            pos_val = sum(pos_val_list)

            neg_val_list = [x for x in weights_val if x <= 0]
            neg_val = abs(sum(neg_val_list))  # weights for negative part

            pos_portion = pos_val/(pos_val + neg_val)
            neg_portion = 1 - pos_portion

            # compute the entropy based on the weights under an attribute value
            entropy = -pos_portion*ma.log2(pos_portion) - neg_portion*ma.log2(neg_portion)
            weighted_entropy = p*entropy
            h1.append(weighted_entropy)
        wa_entropy.append(sum(h1))  # the entropy for the attribute ix
    IG_entropy = [H_total - i for i in wa_entropy]

    return IG_entropy


def get_split(dataset, ig):
    """ On input:
                dataset: a M by N matrix
                ig: a list of  the information gain for each column in dataset
        On output:
                mm: all the indices of the best attribute, in a dict
                attr_name: the name of attribute with the largest gain.
                returns (best_attr, {attr_value:[indices]})
    """
    index_1a = np.argsort(ig)
    index_1b = list(index_1a)
    index_1b.reverse()  # The index of the largest IG is listed first
    best_attr = index_1b[0]

    b = dataset.loc[:, [index_1b[0]]]  # b is a series, index_1b[0] is the purest attribute
    # construct the list of the features in the attribute with largest IG
    c = b.drop_duplicates().values.tolist()
    d = []
    [d.append(c[i][0])for i in range(0, len(c))]
    d.sort()  # element stored in alphabetical order in list d

    kinds = len(d)  # kinds of features in a given attribute
    mm = []
    for i in range(0, kinds):
        h = b[b.values == d[i]].index
        # convert series of index to list
        e = h.values.tolist()
        mm.append({d[i]: e})
    return attr_names[best_attr], mm    # return a tuple, (best attr name, a list of indices in dicts for attr_value)

#v2 = get_split(df, hh)
#print(len(v2[1]))


def stumps_decision_tree(dataset, att_dic, tree_depth, D):
    """
    On input:
            dataset: a dataframe with dimension of M by N, its last column is the labels.
            att_dic: The dictionary contains all attributes as keys and the corresponding attribute
            values in
            tree_depth: A number which specifies the layer of the dictionary, should be smaller than
            its maximum
            D: weights of all examples
    On output:
            tree:the decision tree with a tree_depth layer,which is a multilayer of dictionary
    """
    ig1 = adaboost_ig(dataset, D)  # get the information gain list
    split1 = get_split(dataset, ig1)  # a tuple, (best attr name, a list of indices in dicts for attr_value)
    best_attr = split1[0] # the attribute name with largest IG, the key

    tree = {best_attr:{}} # create a root

    # update the dataset
    # index_max_GI = igcalculation(df)[0].index(max(igcalculation(df)[0]))

    # reduce tree_depth number each time when calling
    tree_depth = tree_depth - 1

    for j in range(len(split1[1])):
        split_idx = split1[1]  # take the indices, which is a list
        idx = list(split_idx[j].values())[0]    # fetch the indices
        s = dataset.loc[idx, :]  # fetch the elements in the dataframe

        D_sub1 = [D[i] for i in idx] # fetch part of the weights D
        ig2 = adaboost_ig(s, D_sub1)  # a list of information gain

        # select the purest attr_value to label. Otherwise,
        s_1 = dataset.loc[idx, df.shape[1] - 1]
        s_2 = s_1.value_counts()

        if len(s_2) == 1:
            attr_value = list(split_idx[j].keys())[0] # fetch the key in dict, the attribute value
            tree[best_attr][attr_value] = dataset.loc[idx[0], dataset.shape[1]-1] # right side is the label
        else:
            split2 = get_split(s, ig2)
            sub_attr_name = split2[0]    # the attribute name for next round of splitting
            attr_value = list(split_idx[j].keys())[0]  # fetch the key in dict, the attribute value
            # tree[attr_name][attr_value] = {subattr_name: idx} # idx is the row_number of the subdataset

            # tree[attr_name][attr_value] = {sub_attr_name: {}}
            tree[best_attr][attr_value] = {sub_attr_name: att_dic[sub_attr_name]} # a single layer of decision tree

            reduced_data = dataset.loc[idx, :]  # fetch the impure sub_data set for further splitting

            if tree_depth > -1:
                sub_tree = stumps_decision_tree(reduced_data, att_dic, tree_depth, D_sub1)  # call the tree recursively
                tree[best_attr][attr_value] = sub_tree
            else:
                return
    return tree

# tt = stumps_decision_tree(df, Attr_dict, 1, D_t)
# pprint.pprint(tt)

# deep copy the true label to be immune to later operations
y_s = copy.deepcopy(df.iloc[:, -1]) # fetch the true labels free of later operations
y_s = y_s.values
y_ts = copy.deepcopy(df_t.iloc[:, -1]) # fetch the true labels free of later operations
y_ts = y_ts.values



def label_predict(dt, inst):
    """
    Description: predict a label for an instance according to its attributes and the given decision tree
    On input:
            dt: the generated decision trees
            inst: a single instance in a list
    On output:
            predicted label, according to the attributes of the instance via the given decision tree
    """
    k = list(dt.keys())[0]  # take the key in the dict and let it serve as a variable.
    idx = attr_name_idx(k)  # take the column idx of the attribute.
    try:
        if isinstance(dt[k][inst[idx]], dict):
            dt_sub = dt[k][inst[idx]]  # recursively fetching
            label = label_predict(dt_sub, inst)
        else:
            label = dt[k][inst[idx]]
    except KeyError:  # exclude the situation where the key does not exist, i.e., the dict fails to predict
        label = None
    return label


def prediction_dataframe(dt, data_set):
    """
    :param dt: the given decision trees
    :param data_set: the set of examples, rows corresponds to the example, columns to the attributes
    :return: a list for the predicted labels for each sample
    """
    predictions = []
    num_of_examples = len(data_set)
    for n in range(num_of_examples):
        predicted_label = label_predict(dt, data_set.loc[n,:])
        predictions.append(predicted_label)
    return predictions


# error calculation between the predicted labels and the true labels
def err_calculation(true_label, predicted_label):
    count = 0  # correct predication count
    for k in range(len(true_label)):
        if true_label[k] != predicted_label[k]:
            count += 1
    return count / float(len(true_label))


# ========================AdaBoost algorithm=================================
# determine the label to be either 1 or -1
def sgn_func(numbers):
    if numbers > 0:
        return 1
    else:
        return -1


def stump_tree_modification(training_data,  stump_tree):
    """
    :param training_data: a dataframe, last column corresponds to the label
    :param stump_tree: a decision tree with only two layers, tree-depth = 1
    :return: a modified stump_tree with the major label for each attribute value
    """
    # assign the label for each attribute_val in the training data
    lable_idx = training_data.shape[1]-1
    attribute = list(stump_tree)[0] # fetch the attribute name in the stump tree
    attribute_idx = attr_name_idx(attribute)
    b1 = stump_tree[attribute]
    attr_values = list(b1.keys())
    num_of_attrval = len(attr_values)
    # label vals
    for j in range(num_of_attrval):
        a1 = training_data.loc[:, [attribute_idx, lable_idx]]
        sub_idx = a1[a1.loc[:, attribute_idx] == attr_values[j]].index.tolist()
        sub_label = [a1.loc[i, lable_idx] for i in sub_idx]
        total = np.sum(sub_label) # determine the sgn of each type of attribute_val
        major_label = sgn_func(total)
        if stump_tree[attribute][attr_values[j]] is None:
            stump_tree[attribute][attr_values[j]] = major_label # take the major label
    return stump_tree


def AdaBoost(T, df):
    """
    :param T: total iteration times, a positive integer
    :param df: a dataset in the form of dataframe
    :return:  H_final hypothesis dataframe, which shows the alpha_t*ht in each iteration at each sample,
            the generated stump tree at each iteration and its corresponding vote.
    """
    # iteration times is adjustable, create a dataframe to store the results
    H_final = pd.DataFrame(np.zeros((len(df), T)))
    # set initial weights equally for all examples
    D_1 = np.ones(len(df)) / len(df)

    trees = []
    votes = []
    progress = 0
    for itera in range(T):
        # monitor the progress
        progress = progress + 1
        ratio  = progress/T
        print(ratio)

        # fetch the data group by the decision tree stump, tree_depth = 1
        tree_stump = stumps_decision_tree(df, Attr_dict, 1, D_1)

        modified_stump = stump_tree_modification(df,  tree_stump)

        trees.append(modified_stump)

        attr_ky = list(tree_stump.keys())[0] # fetch the key as a variable
        stump_attr_vals = list(Attr_dict[attr_ky]) # a list storing all the best attribute values

        attr_col_number = attr_name_idx(attr_ky) # get the column number of the best attribute (in the stump tree)
        # take the row index number of each sample in a sub dataset
        length = len(stump_attr_vals) # the number of the best attribute values
        # hypothesis at round t
        hs = np.zeros(len(df))
        dd = df.iloc[:, -1].values
        # the weights with + or -
        ee = D_1*dd

        for ix in range(length):
            row_idx = df[(df.loc[:, attr_col_number] == stump_attr_vals[ix])].index.tolist()
            val = []
            for iz in row_idx:
                val.append(ee[iz]) # discard val.append(df.iloc[iz, -1])

            # determine the majority label
            major_sgn = sgn_func(np.sum(val))
            # update the value of the dataframe

            # assign the major label to the output column of the h_s
            for j in row_idx:
                hs[j] = major_sgn
            # check_val.append(df.iloc[i, -1])
        # compute the weights
        # compute the error epsilon_t
        diff_idx = []  # find the index of the label where h_s != y_s
        for i in range(len(y_s)):
            if hs[i] != y_s[i]:
                diff_idx.append(i)
        D_diff = D_1[diff_idx] # the value of D where h_s != y_s
        epsilon_t = sum(D_diff)
        # compute the vote alpha_t, and store it at every iteration
        alpha_t = 1/2*np.log((1-epsilon_t)/epsilon_t) # ln function
        votes.append(alpha_t)   # store the votes

        # compute the weights for next round of iteration
        # D_1 = D_1.tolist()

        d_next = []
        for i in range(len(y_s)):
            t = D_1[i]*np.exp(-alpha_t*y_s[i]*hs[i])
            d_next.append(t)

        # compute the normalization constant Z_t
        d1 = np.array(d_next)
        Z_t = np.sum(d1)

        # normalize the weight for weights of the next round of iteration
        D_next = d_next/Z_t

        # update the weights for next round of iteration
        D_1 = D_next
        # save alpha_t*h_t(xi) at a matrix
        for ia in range(len(y_s)):
            H_final.iloc[ia, itera] = alpha_t*hs[ia] # D_1[ia]

    return H_final, trees, votes


# =================================error calculation================================
#k = AdaBoost(3, df)[0] # fetch the alpha_t*h matrix
# the stump tree at each iteration, stored in a list, in the form of dictionary
#kk = AdaBoost(3, df)[1]
#tree1 = kk[1]
#tree1_modified = stump_tree_modification(df, tree1)
# ===============================calculates the error for test data
#results = AdaBoost(3, df)
#stumps = results[1]  # fetch all the generated stump trees
#votes = results[2]  # fetch the votes for each stump trees


def adaboost_prediction(df1, df2, times):
    """
    :param df1: training data, in form of dataframe
    :param df2: test data to be predicted, in form of dataframe,
    :param times: iteration times
    :return: predicted labels in a list, in the order of examples, stored in a matrix
           a_test: the hypothesis by all the classifiers through major label.
           v_test: vote for these classifiers in a list.

    """

    results = AdaBoost(times, df1) # process the training data
    stumps = results[1]  # fetch all the generated stump trees
    votes = results[2]

    num_stumps = len(stumps)  # also the iteration times
    predictions_test = []
    print(stumps)

    for j in range(num_stumps):
        predicted_labels = prediction_dataframe(stumps[j], df2) # predict via the modified stumps
        predictions_test.append(predicted_labels)

    aaa = (pd.DataFrame(predictions_test)).T # predicted labels for each trees
    a_test = aaa.values      # hypothesis of all the examples through prediction.
    v_test = np.array(votes) # votes for all the stumps
    print(np.sum(a_test, axis = 0))
    #print(v_test)

    final = np.dot(a_test, v_test) # multiply the weights*h_s and sum for each sample
    print(sum(final))
    # final results
    test_prediction = []
    for kk in range(len(final)):
        pred = sgn_func(final[kk])
        test_prediction.append(pred)
    print(sum(test_prediction))
    return test_prediction, a_test, v_test

# =================================compute the test error
a = adaboost_prediction(df, df_t, 100) # set the iteration times to 500

h_ts = a[1]

votes = np.array(a[2])
r = len(votes)

alp = np.zeros([h_ts.shape[0], h_ts.shape[1]])
for i in range(h_ts.shape[0]):
    for j in range(h_ts.shape[1]):
        alp[i, j]= votes[j]

ah = alp*h_ts

upper_ones = np.ones([r, r])
uptri = np.triu(upper_ones, k = 0)

product_test = np.dot(ah, uptri)
print(product_test)

prediction = np.zeros([product_test.shape[0],product_test.shape[1]])
num_r = product_test.shape[0]
num_c = product_test.shape[1]

# iterative sum the columns
for i1 in range(num_r):
    for i2 in range(num_c):
        prediction[i1, i2] = sgn_func(product_test[i1, i2])


print(np.sum(prediction, axis=0))
# compute the error
error_test = []
for i in range(ah.shape[1]):
    xxx = err_calculation(y_ts, prediction[:, i])
    error_test.append(xxx)

iteration_times = [i+1 for i in range(ah.shape[1])]
print(error_test)


# plot the test errors
plt.plot(iteration_times, error_test, label = 'test data')
plt.title("AdaBoost test Errors")
plt.xlabel('iteration times')
plt.ylabel('error_test')
plt.legend()
plt.show()


"""
# =================== calculate the error for the test data at different maximum iteration times =======================
# compute the test error and training error
max_t = 500
lag = 1
rt = [(i+1)*lag for i in range(int(max_t/lag))]
first = [1]
iteration_times = first + rt
print(iteration_times)


err_test = []
err_train = []

for t in range(4):

    # compute the error for test data
    b = adaboost_prediction(df, df_t, t+1)
    test_true_label = (df_t.loc[:, df_t.shape[1] - 1]).values
    error_b = err_calculation(y_ts, b)
    err_test.append(error_b)

print(iteration_times)
print(err_test)

# plot the errors
plt.plot(iteration_times, err_test, label = 'test data')
plt.title("AdaBoost Errors")
plt.xlabel('iteration times')
plt.ylabel('error')
plt.legend()
plt.show()
"""

"""
# ======================== calculate the training error for different maximum iteration times ==========================
# k.to_csv('HH.csv')[0]
#print((AdaBoost(4, df))[2])
k = (AdaBoost(50, df))[0]
k_mat = k.values
upper_ones = np.ones([k.shape[1], k.shape[1]])
uptri = np.triu(upper_ones, k = 0) # an upper triangle matrix for summation

# compute the summation of the previous t iteration of alpha_t*h_t
product = np.dot(k_mat, uptri)
prediction = np.zeros([k.shape[0],k.shape[1]])
num_r = product.shape[0]
num_c = product.shape[1]

# iterative sum the columns
for i1 in range(num_r):
    for i2 in range(num_c):
        prediction[i1, i2] = sgn_func(product[i1, i2])

print(np.sum(prediction, axis=0))
# compute the error
error = []
for i in range(k.shape[1]):
    xxx = err_calculation(y_s, prediction[:, i])
    error.append(xxx)


iteration_times = [i+1 for i in range(k.shape[1])]
print(error)


# draw the errors along with iteration times
plt.plot(iteration_times, error, label = 'training data')
plt.xlabel('iteration times')
plt.ylabel('error')
plt.legend()
plt.show()
"""

"""
# ============================= calculate the stump trees errors ================================
result = AdaBoost(500, df)
stumps_training = result[1]

# errors for training data
err = []
for i in range(len(stumps_training)):
    predicts = prediction_dataframe(stumps_training[i], df)
    errors = err_calculation(y_s, predicts)
    err.append(errors)

# errors for test data
err_t = []
for i in range(len(stumps_training)):
    predicts = prediction_dataframe(stumps_training[i], df)
    errors = err_calculation(y_ts, predicts)
    err_t.append(errors)

# plot together
tree_sequence = [i+1 for i in range(len(stumps_training))]
plt.plot(tree_sequence, err, label = 'stumps_error_training')
plt.plot(tree_sequence, err_t, label = 'stumps_error_test')
plt.title("Errors for stumps at each iteration")
plt.xlabel('iteration times')
plt.ylabel('error')
plt.legend()
plt.show()
"""
