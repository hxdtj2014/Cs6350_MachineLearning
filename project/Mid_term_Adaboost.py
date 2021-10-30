# Data preprocessing
# import the csv file
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import pprint

# read the training data as a dataframe
df = pd.read_csv(r'C:/Users/nanji/OneDrive/桌面/CS6350 Machine Learning'
                 r'/Project/Mid-term report/train_1.csv', header=None)
df.drop(0, axis=0, inplace=True)
df.reset_index(inplace=True, drop=True)

#print(df)
# read the test data as a dataframe
df_t = pd.read_csv(r'C:/Users/nanji/OneDrive/桌面/CS6350 Machine Learning'
                 r'/Project/Mid-term report/test_1.csv', header=None)
df_t.drop(0, axis=0, inplace=True)
df_t.reset_index(inplace=True, drop=True)
print(df_t)


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


def adaboost_ig(dataset_ig, D_t):
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
    dataset_ig.reset_index(inplace=True, drop=True)
    # calculate the number of the examples
    length = len(dataset_ig)
    # compute the weighted average entropy
    wa_entropy = []
    # find the column index of the label
    df_cols = dataset_ig.shape[1] - 1

    # positive portion of the labels
    pos_label = [z for z in dataset_ig.loc[:, df_cols] if z > 0]
    pos = np.sum(pos_label) / length
    neg = 1 - np.sum(pos_label) / length

    if pos == 0:
        H_total = - neg * np.log2(neg)
    elif neg == 0:
        H_total = -pos * np.log2(pos)
    else:
        H_total = -pos * np.log2(pos) - neg * np.log2(neg)

    for ix in range(df_cols):
        b = dataset_ig.loc[:, [ix, df_cols]]  # fetch the each attribute column together with the output

        attr_val_col = b.loc[:, ix]
        true_labels = b.loc[:, df_cols]  # true_labels for all examples
        attr_val = set(attr_val_col)  # the attribute values in each attribute

        weights = [c * d for c, d in zip(true_labels, D_t)]  # all the weights with + or -

        h1 = []
        for ixx in attr_val:
            weights_idx = attr_val_col[attr_val_col == ixx].index.tolist()

            weights_val = [weights[jn] for jn in weights_idx]

            port = [abs(jm) for jm in weights_val]  # portion of ixx in the whole dataset
            p = np.sum(port)  # sum of all weights of examples in attr_val, ixx

            pos_val_list = [x for x in weights_val if x > 0.0]
            pos_val = np.sum(pos_val_list)

            neg_val_list = [x for x in weights_val if x <= 0.0]
            neg_val = abs(np.sum(neg_val_list))  # weights for negative part

            pos_portion = pos_val / (pos_val + neg_val)
            # print((pos_val + neg_val)*1)

            neg_portion = 1 - pos_portion

            # compute the entropy based on the weights under an attribute value

            # exclude the 0 situation
            if pos_portion == 0:
                entropy = - neg_portion * np.log2(neg_portion)
            elif neg_portion == 0:
                entropy = -pos_portion * np.log2(pos_portion)
            else:
                entropy = -pos_portion * np.log2(pos_portion) - neg_portion * np.log2(neg_portion)

            weighted_entropy = p * entropy
            h1.append(weighted_entropy)

        wa_entropy.append(np.sum(h1))  # the entropy for the attribute ix
    IG_entropy = [H_total - x for x in wa_entropy]
    return IG_entropy


def get_split(dataset, ig):
    """
    On input:
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
    [d.append(c[hn][0]) for hn in range(0, len(c))]
    d.sort()  # element stored in alphabetical order in list d

    kinds = len(d)  # kinds of features in a given attribute
    mm = []
    for gb in range(0, kinds):
        h = b[b.values == d[gb]].index
        # convert series of index to list
        e = h.values.tolist()
        mm.append({d[gb]: e})
    return attr_names[best_attr], mm  # return a tuple, (best attr name, a list of indices in dicts for attr_value)


def stumps_decision_tree(dataset, att_dic, D):
    """
    Description: to calculate a decision tree considering the weighted entropy, and the tree_depth is just 1.
    On input:
            dataset: a dataframe with dimension of M by N, its last column is the labels.
            att_dic: The dictionary contains all attributes as keys and the corresponding attribute
            values in
            D: weights of all examples
    On output:
            tree:the decision tree with a tree_depth layer,which is a multilayer of dictionary
    """
    ig1 = adaboost_ig(dataset, D)  # get the information gain list
    split1 = get_split(dataset, ig1)  # a tuple, (best attr name, a list of indices in dicts for its attr_values)
    best_attr = split1[0]  # the attribute name with largest IG, the key
    tree = {best_attr: {}}  # create a root
    label_idx = df.shape[1]- 1
    best_attr_col = attr_name_idx(best_attr)

    best_attr_vals = att_dic[best_attr]    # the attribute values in a list

    b1 = df.loc[:, [best_attr_col, label_idx]]
    for i in range(len(best_attr_vals)):
        attr_val = best_attr_vals[i]
        sub_idx = b1[b1.loc[:, best_attr_col] == attr_val].index.tolist() # the row index
        attr_val_label = b1.loc[sub_idx, label_idx]
        major_label = attr_val_label.value_counts() # fetch the major label
        hh = pd.DataFrame(list(major_label.index))
        major = hh[0][0]

        tree[best_attr][attr_val] = major
    return tree


# determine the label to be either 1 or -1
def sgn_func(numbers):
    if float(numbers) > 0:
        return 1
    else:
        return -1

"""
D_1 = np.ones(len(df)) / len(df)
tt = stumps_decision_tree(df, Attr_dict, D_1)
pprint.pprint(tt)
new = stump_tree_modification(df, tt)
pprint.pprint(new)
"""

# deep copy the true label to be immune to later operations
y_s = copy.deepcopy(df.iloc[:, -1])  # fetch the true labels free of later operations
y_s = y_s.values
# print(y_s)


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
        label = 0
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


# error calculation between the predicted labels and the true labels
def err_calculation(true_label, predicted_label):
    count = 0  # correct predication count
    for k in range(len(true_label)):
        if true_label[k] != predicted_label[k]:
            count += 1
    return count / float(len(true_label))


# ========================AdaBoost algorithm===========================================================
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
        ratio = progress / T
        print(ratio)

        # fetch the data group by the decision tree stump, tree_depth = 1
        tree_stump = stumps_decision_tree(df, Attr_dict, D_1)

        # store the trained stumps
        trees.append(tree_stump)
        attr_ky = list(tree_stump.keys())[0]  # fetch the key as a variable

        stump_attr_vals = list(Attr_dict[attr_ky])  # a list storing all the best attribute values

        attr_col_number = attr_name_idx(attr_ky)  # get the column number of the best attribute (in the stump tree)
        # take the row index number of each sample in a sub dataset
        length = len(stump_attr_vals)  # the number of the best attribute values

        dd = df.iloc[:, -1].values
        # the weights with + or -
        ee = D_1 * dd

        # hypothesis at round t
        hs = np.zeros(len(df))
        for ix in range(length):
            row_idx = df[(df.loc[:, attr_col_number] == stump_attr_vals[ix])].index.tolist()
            val = []
            for iz in row_idx:
                val.append(ee[iz])  # discard val.append(df.iloc[iz, -1])
            # determine the majority label
            major_sgn = sgn_func(np.sum(val))
            # update the value of the dataframe
            # assign the major label to the output column of the h_s
            for j in row_idx:
                hs[j] = major_sgn
            # check_val.append(df.iloc[i, -1])

        # hs = prediction_dataframe(tree_stump, df)

        # compute the weights
        # compute the error epsilon_t
        diff_idx = []  # find the index of the label where h_s != y_s
        for xs in range(len(y_s)):
            if hs[xs] != y_s[xs]:
                diff_idx.append(xs)
        D_diff = D_1[diff_idx]  # the value of D where h_s != y_s
        epsilon_t = sum(D_diff)

        if epsilon_t > 0.5:
            epsilon_t = 1 - epsilon_t

        # compute the vote alpha_t, and store it at every iteration
        alpha_t = 1 / 2 * np.log((1 - epsilon_t) / epsilon_t)  # ln function
        votes.append(alpha_t)  # store the votes

        # compute the weights for next round of iteration
        # D_1 = D_1.tolist()

        d_next = []
        for cd in range(len(y_s)):
            t = D_1[cd] * np.exp(-alpha_t * y_s[cd] * hs[cd])
            d_next.append(t)

        # compute the normalization constant Z_t
        d1 = np.array(d_next)
        Z_t = np.sum(d1)
        # normalize the weight for weights of the next round of iteration
        D_next = d_next / Z_t
        # update the weights for next round of iteration
        D_1 = D_next
        # save alpha_t*h_t(xi) at a matrix
        for ia in range(len(y_s)):
            H_final.iloc[ia, itera] = alpha_t * hs[ia]  # D_1[ia]

    return H_final, trees, votes


def adaboost_prediction(df1, df2, times):
    """
    :param df1: training data, in form of dataframe
    :param df2: test data to be predicted, in form of dataframe,
    :param times: iteration times
    :return: predicted labels in a list, in the order of examples, stored in a matrix
           a_test: the hypothesis by all the classifiers through major label.
           v_test: vote for these classifiers in a list.
    """
    results = AdaBoost(times, df1)  # process the training data
    stumps = results[1]  # fetch all the generated stump trees
    votes = results[2]  # fetch the votes

    num_stumps = len(stumps)  # also the total iteration times
    predictions_test = []
    # print(stumps)

    for j in range(num_stumps):
        predicted_labels = prediction_dataframe(stumps[j], df2)  # predict via the modified stumps
        predictions_test.append(predicted_labels)

    aaa = (pd.DataFrame(predictions_test)).T  # predicted labels for each trees
    a_test = aaa.values  # hypothesis of all the examples through prediction.
    v_test = np.array(votes)  # votes for all the stumps
    print(np.sum(a_test, axis=0))
    # print(v_test)

    final = np.dot(a_test, v_test)  # multiply the votes*h_s and sum for each sample
    print(final)
    # print(sum(final))
    # final results
    test_prediction = []
    for kk in range(len(final)):
        predict = sgn_func(final[kk])
        # print(final[kk])
        test_prediction.append(predict)
    print(sum(test_prediction))
    return test_prediction, a_test, v_test

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


# =================================================
#
# ==== prediction result analysis =================
a = adaboost_prediction(df, df, 500)
prediction = a[0]
print(prediction)
# print(a[1]) # the votes


'''
b = AdaBoost(50, df)
print(b[1])
print(b[2]) # the votes
'''