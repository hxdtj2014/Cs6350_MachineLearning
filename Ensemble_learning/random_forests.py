# Random forests Algorithm A2Q2(d) and (e)
# import the csv file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import pprint as pp

# =================== pre-processing the training data=================================

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


def igcalculation(data,random_feature_size):
    """
    Description: randomly select a subset of features to calculate the information gain for splitting

    On input:
        data:
            a M by N matrix, the last column is the labels other columns are attributes.
        random_feature_size:
            The size of features that are randomly selected

    On output:
        a list contains the calculated information gain
        in the order that the attributes locate in the
        columns of the input matrix.
        In such form ([entropy_based IG], [Gini_based IG], [ME_based IG])
    """
    # calculate the number of the examples
    length = len(data)
    # compute the weighted average entropy, Gini index, and majorityerror
    wa_entropy = []
    wa_GI = []
    wa_ME = []

    # randomly select a subset of attributes
    idx = np.random.randint(df.shape[1]-1, size=random_feature_size)
    set_idx = set(idx)
    unrepeated_idx = list(set_idx)
    # zero other attributes
    all_col_idx = [i for i in range(df.shape[1]-1)]
    not_selected_idx = list(set(all_col_idx).difference(unrepeated_idx))
    for k in not_selected_idx:
        data.loc[:,k] = 0

    # find the column index of the label
    df_cols = data.shape[1] - 1

    for ix in range(df_cols):
        b = data.loc[:, [ix, df_cols]]
        # count the features in each attribute and its corresponding values
        c = b.value_counts(sort=False)
        label = list(c.index)
        c1, c2 = map(list, zip(*label))
        # eliminate the repeated elements through listing
        c1 = list(set(c1))
        # list the features in attribute in ascending order
        c1.sort()
        # compute the entropy for each feature in the attribute
        H = []
        G_1 = []
        MajorityError = []
        proportion = []
        for idx in c1:
            m0 = c[idx].to_numpy()
            # calculate the proportion of each attribute in the whole sample set
            proportion.append(np.sum(m0) / length)
            # proportion of labels in each feature
            m1 = m0 / np.sum(m0)
            e = np.log2(m1)
            # compute the entropy
            sum_log_product = np.dot(-1 * e, m1)
            H.append(sum_log_product)
            # compute the Gini
            gini = 1 - np.dot(m1, m1)
            G_1.append(gini)
            # compute ME-information gain
            # portion of the labels in each feature of an attribute
            if len(m0) > 1:  # exclude the single label situation for ME calculation
                majority_error = 1 - np.max(m0) / np.sum(m0)
            else:
                majority_error = 0
            MajorityError.append(majority_error)
        entropy_g = np.dot(H, proportion)
        Gini = np.dot(G_1, proportion)
        me = np.dot(MajorityError, proportion)
        wa_entropy.append(entropy_g)
        wa_GI.append(Gini)
        wa_ME.append(me)

    # compute the entropy for the whole example sets
    values = data.loc[:, [df_cols]]
    # count the number of elements in the value column
    counted_values = values.value_counts(sort=False, normalize=True)

    value_ratio = counted_values.to_numpy()
    s1 = np.log2(value_ratio)
    s = np.dot(-1 * s1, value_ratio)
    # entropy for the whole examples
    # compute the entropy-based information gain
    IG_entropy = s * np.ones(len(wa_entropy)) - wa_entropy


    # Compute the Gini_index-based information gain
    g_1 = 1 - np.dot(value_ratio, value_ratio)
    IG_gini = g_1 * np.ones(len(wa_GI)) - wa_GI

    # Compute the MajorityError
    me1 = 1 - max(value_ratio) / np.sum(value_ratio)
    ME_gain = me1 * np.ones(len(wa_ME)) - wa_ME

    return list(IG_entropy), list(IG_gini), list(ME_gain)


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


v1 = igcalculation(df,6)[0]
v2 = get_split(df, v1)

print(v1)
# print(v2)
# attribute value: print(list(Attr_dict.values()))
# attribute name: print(list(Attr_dict.keys()))


def decision_tree(dataset,random_feature_size,  att_dic, tree_depth, ig_type):
    """
    On input:
            dataset: a dataframe with dimension of M by N, its last column is the labels.
            att_dic: The dictionary contains all attributes as keys and the corresponding attribute
            values in
            tree_depth: A number which specifies the layer of the dictionary, should be smaller than
            its maximum
            ig_type: the type of gain to use, 0,1,2, stands for entropy, Gini index, and majorityerror
            gain.
    On output:
            tree:the decision tree with a tree_depth layer,which is a multilayer of dictionary
    """
    ig1 = igcalculation(dataset,random_feature_size)[ig_type]  # select the type of IG to split (entropy), 0,1,2 corresponds to the
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
        ig2 = igcalculation(s, random_feature_size)[ig_type]

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
                sub_tree = decision_tree(reduced_data, random_feature_size, att_dic, tree_depth, ig_type)  # call the tree recursively
                tree[best_attr][attr_value] = sub_tree
            else:
                return
    return tree

# tr = decision_tree(df,4,  Attr_dict, 5, 0)
# pp.pprint(tr)

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
        label = 0  #'None'
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


def err_estimation(data, d_tree):
    """
    Description: to calculate the error caused by the failures of prediction
    :param data: a data frame. m examples, listed in rows, columns correspond to attributes.
    :param d_tree: the given decision trees in the form of a multi layers of dictionary.
    :return: the prediction error.
    """
    y = []
    for i in range(len(data)):
        inst = data.loc[i, :].tolist()
        k = label_predict(d_tree, inst)
        if k == data.loc[i, data.shape[1]-1]: # the predicted label matches with the true label
            y.append(1)
        else:
            y.append(0)
    error = 1 - np.sum(y)/len(data)
    return error  # a value


# ==================================Random forests Algorithm==================================
def random_forests(total_iteration, random_feature_size, df1, df2, num_examples):
    """
    :param total_iteration: total_iteration: the number of trees
    :param random_feature_size: The size of features that are randomly selected
    :param df1: training dataset
    :param df2: dataset that are used to serve as an input for predicting the labels
    :param num_examples: num_examples: the amount of examples to draw
    :return: (errors, and num_examples decision trees in a list)
    """
    # the generated decision trees, a dict, stored in the list, in the sequence of iteration
    classifier_trees = []
    errors = []
    for i in range(total_iteration):
        print(i / total_iteration)  # Monitor the computing progress
        # generate repeatable numbers serving as row indices
        idx = np.random.randint(num_examples, size=num_examples)
        # slice randomly of the original dataframe
        random_part = df1.loc[idx, :]
        # reset the sliced index
        random_part.reset_index(inplace=True, drop=True)
        # fully grow the decision trees
        trees = decision_tree(random_part,random_feature_size, Attr_dict, 16, 0)
        # store the learned decision trees
        classifier_trees.append(trees)
    # ==================== error calculation for all the decision trees =============================== #
    num_trees = len(classifier_trees)
    for j in range(num_trees):
        tree = classifier_trees[j]
        err = err_estimation(df2, tree)
        errors.append(err)
    return errors, classifier_trees


# ===========================A2Q2(d)=========================================
""" vary the maximum iterationtimes from 0 to 500
tt = 500 # set the tree size to be 500 to reduce the computation time
lag = 25
rt = [(i+1)*lag for i in range(int(500/lag))]
first = [1]
run_times = first + rt
print(run_times)

err_train = []
err_test =[]
for i in run_times:
    print(i/tt)  # monitor the progress
    results_train = random_forests(tt, 2, df, df, 5000) # set the random_feature_size to 2,4,6
    err_train = results_train[0]
    mean_err_train = np.mean(err_train)
    print(mean_err_train)
    err_train.append(mean_err_train)

    results_test = random_forests(tt, 2, df, df_t, 5000)
    err_test = results_test[0]
    mean_err_test = np.mean(err_test)
    err_test.append(mean_err_test)
    print(mean_err_test)

print(err_train)
print(err_test)
plt.plot(run_times, err_train, label='training_error')
plt.plot(run_times, err_test, label='test_error')
# details

plt.legend()
plt.xlabel('tree_size')
plt.ylabel('mean_err')
plt.show()


"""

"""
# ======================================A2Q2(d)=============================================
tt = 500
lo_tril = np.ones([tt, tt]) # construct a lower triangle with 1s and 0s
lo = np.tril(lo_tril, k = 0)


results_train = random_forests(tt, 2, df, df, 5000) # set the random_feature_size to 2,4,6
err_train = results_train[0]

results_test = random_forests(tt, 2, df, df_t, 5000)
err_test = results_test[0]


a1 = np.array(err_train)
a2 = np.array(err_test)

frac = np.array([1/(i+1) for i in range(tt)])
b1 = np.dot(lo, a1)
acc_cum_train = b1*frac # average the error
print(acc_cum_train)


b2 = np.dot(lo, a2)
acc_cum_test = b2*frac # average the error
print(acc_cum_test)

iterations = [(i+1) for i in range(tt)]


plt.plot(iterations, err_train, label='training_error')
plt.plot(iterations, err_test, label='test_error')
# details

plt.legend()
plt.xlabel('tree_size')
plt.ylabel('mean_err')
plt.show()
"""



"""
# ============================== run 100 times =========================
run_times = 10*np.ones([10,1]) # set the run times
output_predicted_labels = [] # a list of list, each predicted labels are stored in a list.
for run_time in run_times:
    dec_tr = random_forests(10, df, df, 1000)
    first_dec_tree = dec_tr[1][0] # get the first decision tree in each run time
    predicted_labels = prediction_dataframe(first_dec_tree, df_t)
    output_predicted_labels.append(predicted_labels)

print(output_predicted_labels)
"""

# ================================A2Q2e======================================
# bias and variance decomposition

tt = 40 # repeated times
run_times = [100*i**0 for i in range(tt)] # set the run times
mv = []
mb =[]
gse = []
count = 0
for run_time in run_times:
    count = count + 1
    print(count) # monitor the progress
    results_train = random_forests(tt, 6, df, df_t, 5000)
    trees = results_train[1] # 100 learned trees

    # fetch the ground truth of all the test examples
    y_s = df_t.loc[:,df_t.shape[1]-1]

    # get the hypothesis for all the examples in the test data
    H_s = []
    for tre in trees:
        h_s = prediction_dataframe(tre, df_t)
        H_s.append(h_s)

    # convert it to dataframes
    H_s = pd.DataFrame(H_s)

    # compute the sample variance
    variance_h = H_s.var(axis=0,ddof=1)
    mean_variance = variance_h.mean() # take average of the variance of all the examples


    average_value = H_s.mean(axis = 0) # compute the average of the hypothesis
    # compute the bias
    bias =[]
    for i in range(len(average_value)):
        t = (average_value[i]-y_s[i])**2
        bias.append(t)
    bias = pd.DataFrame(bias)
    mean_bias = bias.mean()

    general_squared_error = mean_bias + mean_variance
    print(mean_variance)
    print(mean_bias)
    print(general_squared_error)
    mv.append(mean_variance)
    mb.append(mean_bias)
    gse.append(general_squared_error)

print(mv)
print(mb)
print(gse)
print(np.mean(mv))
print(np.mean(mb))
print(np.mean(gse))
