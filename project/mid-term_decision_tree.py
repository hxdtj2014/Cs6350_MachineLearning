# Data preprocessing
# import the csv file
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import pprint

from sklearn import metrics

# read the training data as a dataframe
df = pd.read_csv(r'C:\Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Project\Mid-term report\New folder/train_bin6.csv', header=None)
df.drop(0, axis=0, inplace=True)
df.reset_index(inplace=True, drop=True)

# print(df)
# read the test data as a dataframe
df_t = pd.read_csv(r'C:\Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                   r'\Project\Mid-term report\New folder/test_bin6.csv', header=None)
df_t.drop(0, axis=0, inplace=True)
df_t.reset_index(inplace=True, drop=True)
# print(df_t)


# The attribute names to the imported dataframe
attr_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
              'marital-status', 'occupation', 'relationship', 'race', 'sex',
              'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
'''
Attr_dict = {'age': ['no', 'yes'],
            'workclass': ['Self-emp-inc', 'State-gov', 'Self-emp-not-inc',
                       'Never-worked', 'Local-gov', 'Private', '?',
                       'Without-pay', 'Federal-gov'],
            'fnlwgt': ['no', 'yes'],
            'education': ['Doctorate', '9th', '5th-6th', '10th', 'Preschool',
                       'Bachelors', '1st-4th', 'Masters', '7th-8th', 'Assoc-voc',
                       '12th', 'Prof-school', 'HS-grad', 'Assoc-acdm',
                       '11th', 'Some-college'],
            'education-num': ['no', 'yes'],
            'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced',
                            'Married-spouse-absent', 'Widowed', 'Separated',
                            'Married-AF-spouse'],
            'occupation': ['Machine-op-inspct', 'Tech-support', 'Craft-repair',
                        'Adm-clerical', 'Protective-serv', 'Farming-fishing',
                        'Armed-Forces', 'Sales', '?', 'Transport-moving',
                        'Prof-specialty', 'Handlers-cleaners', 'Exec-managerial',
                        'Other-service', 'Priv-house-serv'],
            'relationship': ['Husband', 'Own-child', 'Not-in-family', 'Wife',
                          'Unmarried', 'Other-relative'],
            'race': ['Black', 'Asian-Pac-Islander', 'Other', 'White', 'Amer-Indian-Eskimo'],
            'sex': ['Male', 'Female'],
            'capital-gain': ['yes'],
            'capital-loss': ['yes'],
            'hours-per-week': ['no', 'yes'],
            'native-country': ['Germany', 'El-Salvador', 'Haiti', 'Cambodia',
                            'Poland', 'Hungary', 'United-States', 'Portugal',
                            'Nicaragua', 'Puerto-Rico', 'Peru', 'South', 'Scotland',
                            'Greece', 'Jamaica', 'Vietnam', 'India', 'Columbia',
                            'Ireland', 'Hong', 'Canada', 'France', 'Dominican-Republic',
                            'Mexico', 'Yugoslavia', 'Italy', 'Honduras',
                            'Outlying-US(Guam-USVI-etc)', 'Guatemala', 'Japan',
                            'England', 'Thailand', 'Ecuador', 'Cuba', 'Laos',
                            'Philippines', '?', 'Trinadad&Tobago', 'Iran', 'China', 'Taiwan']}
'''

Attr_dict = {}
for i in range(df.shape[1] - 1):
    Attr_dict[attr_names[i]] = list(set(df.loc[:, i]))

# print(Attr_dict)
# =========================================================================

# fetch the column number by attr_name
def attr_name_idx(idx):
    attr_names1 = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                   'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    func_val = attr_names1.index(idx)  # column number for each attribute
    return func_val


def igcalculation(data):
    """ On input:
        a M by N matrix, the last column is the labels
        other columns are attributes.

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
        me = []
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

    ky = ['age', 'job', 'martial', 'education',
            'default', 'balance', 'housing', 'loan',
            'contact', 'day', 'month', 'duration',
            'campaign', 'pdays', 'previous', 'poutcome']

    # Final answer for the entropy-based information gain
    EN = dict(zip(ky, IG_entropy))

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


def decision_tree(dataset, att_dic, tree_depth, ig_type):
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
    ig1 = igcalculation(dataset)[ig_type]  # select the type of IG to split (entropy), 0,1,2 corresponds to the
    split1 = get_split(dataset, ig1)  # a tuple, (best attr name, a list of indices in dicts for attr_value)
    best_attr = split1[0] # the attribute name with largest IG, the key

    tree = {best_attr:{}} # create a root

    # update the dataset
    # reduce tree_depth number each time when calling
    tree_depth = tree_depth - 1

    best_attr_col = attr_name_idx(best_attr)
    label_idx = df.shape[1] - 1
    b1 = df.loc[:, [best_attr_col, label_idx]]
    best_attr_vals = att_dic[best_attr]

    for j in range(len(split1[1])):
        split_idx = split1[1]  # take the indices, which is a list
        idx = list(split_idx[j].values())[0]    # fetch the indices
        s = dataset.loc[idx, :]  # fetch the elements in the dataframe
        ig2 = igcalculation(s)[ig_type]

        # select the purest attr_value to label. Otherwise,
        s_1 = dataset.loc[idx, df.shape[1] - 1]
        s_2 = s_1.value_counts()

        if len(s_2) == 1:
            attr_value = list(split_idx[j].keys())[0] # fetch the key in dict, the attribute value
            tree[best_attr][attr_value] = dataset.loc[idx[0], dataset.shape[1]-1] # right side is the label
        else:
            split2 = get_split(s, ig2)
            attr_value = list(split_idx[j].keys())[0]  # fetch the key in dict, the attribute value
            # tree[attr_name][attr_value] = {subattr_name: idx} # idx is the row_number of the subdataset

            # tree[attr_name][attr_value] = {sub_attr_name: {}}
            attr_val = best_attr_vals[j]
            sub_idx = b1[b1.loc[:, best_attr_col] == attr_val].index.tolist()  # the row index
            attr_val_label = b1.loc[sub_idx, label_idx]
            major_label = attr_val_label.value_counts()  # fetch the major label
            hh = pd.DataFrame(list(major_label.index))
            major = hh[0][0]

            tree[best_attr][attr_value] = major  # a single layer of decision tree
            reduced_data = dataset.loc[idx, :]   # fetch the impure sub_data set for further splitting

            if tree_depth > -1:
                sub_tree = decision_tree(reduced_data, att_dic, tree_depth, ig_type)  # call the tree recursively
                tree[best_attr][attr_value] = sub_tree
            else:
                return tree[best_attr][attr_value]
    return tree


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
        label = 0  # 'None'
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
        predicted_label = label_predict(dt, data_set.loc[n, :])
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

# ================================= Generate prediction for Kaggle submission ===============================
"""
full_grow_tree = decision_tree(df, Attr_dict, 14, 0)
# error_train = err_estimation(df, full_grow_tree)
# print(error_train)
b = prediction_dataframe(full_grow_tree, df_t)

# replace -1 with 0 for Kaggle submission
for i in range(len(b)):
    if b[i] == -1:
        b[i] = 0

p_df = pd.DataFrame(b)
p_df.to_csv(r'C:/Users/nanji/OneDrive/桌面/CS6350 Machine Learning'
            r'/Project/Mid-term report/predicted_decision_tree.csv', index=False)
print(p_df)
"""

# ================================= training error calculation ===============================

"""
full_grow_tree = decision_tree(df, Attr_dict, 14, 0)
error_train = err_estimation(df, full_grow_tree)
print(error_train)
pp = prediction_dataframe(full_grow_tree, df_t)
print(pp)
for i in range(len(pp)):
    if pp[i] == -1:
        pp[i] = 0
"""
# pp1 = pd.DataFrame(pp)

# pp1.to_csv('C:/Users/nanji/OneDrive\桌面\CS6350 Machine Learning/Project/Mid-term report/bin6.csv', index=False)

# ================================ training error analysis ========================================
tree_depth = [i+1 for i in range(14)]
y_s = copy.deepcopy(df.iloc[:, -1])  # fetch the true labels free of later operations
y_s = y_s.values
error_set = []
progress = 0
for j in tree_depth:
    progress = progress + 1
    print(progress/len(tree_depth))  # Monitor the computing progress
    full_grow_tree1 = decision_tree(df, Attr_dict, j, 0)
    error_train = err_estimation(df, full_grow_tree1)
    error_set.append(error_train)

print(tree_depth)
print(error_set)

plt.plot(tree_depth, error_set, label='training data(bin6)')
plt.xlabel('tree depth')
plt.ylabel('error')
plt.legend()
plt.show()

"""
# ================================= classifier evaluation  ===============================
# computing the confusion matrix, the FPR, and the TPR, plot the ROC curve.

tree_depth = [i+1 for i in range(14)]
error_set = []

progress = 0
TP = []
TN = []
FN = []
FP = []
y_s = copy.deepcopy(df.iloc[:, -1])  # fetch the true labels free of later operations
y_s = y_s.values

for j in tree_depth:
    progress = progress + 1
    print(progress/len(tree_depth))  # Monitor the computing progress
    full_grow_tree1 = decision_tree(df, Attr_dict, j, 0)
    error_train = err_estimation(df, full_grow_tree1)
    error_set.append(error_train)

    y_predict = prediction_dataframe(full_grow_tree1, df)
    # computing the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_s, y_predict)
    TP.append(confusion_matrix[1, 1])
    TN.append(confusion_matrix[0, 0])
    FN.append(confusion_matrix[1, 0])
    FP.append(confusion_matrix[0, 1])

print(tree_depth)
print(error_set)
plt.plot(tree_depth, error_set, label = 'training data')
plt.xlabel('tree depth')
plt.ylabel('error')
plt.legend()
plt.show()


# plot the AUC-ORC curve
TPR = []  # True Positive Rate
FPR = []
for j in range(len(TP)):
    a = TP[j]/(TP[j]+FN[j])
    b = FP[j]/(TN[j]+FP[j])
    TPR.append(a)
    FPR.append(b)

print(TPR)
print(FPR)
plt.plot(FPR, TPR, label = 'AUC-ORC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
"""

# ============================= computing the confusion matrix ==========================
"""
# deep copy the true label to be immune to later operations
y_s = copy.deepcopy(df.iloc[:, -1])  # fetch the true labels free of later operations
y_s = y_s.values
full_grow_tree2 = decision_tree(df, Attr_dict, 14, 0)
y_predict = prediction_dataframe(full_grow_tree2, df)
print(y_predict)
print(np.sum(y_predict))


y_pred = ["a", "b", "c", "a", "b"]
# Actual values
y_act = ["a", "b", "c", "c", "a"]
# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.

confusion_matrix = metrics.confusion_matrix(y_s, y_predict)
print(confusion_matrix)
print(confusion_matrix[0,0])

# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_s, y_predict)) """
