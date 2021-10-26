# import the csv file
import numpy as np
import pandas as pd
import pprint

df = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面'
                 r'\CS6350 Machine Learning\Assignments'
                 r'\Assignment1\car\train.csv', header=None)


# The attribute names to the imported dataframe
attr_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'labels']

Attr_dict = {'buying': ['vhigh', 'high', 'med', 'low'],
             'maint': ['vhigh', 'high', 'med', 'low'],
             'doors': ['2', '3', '4', '5more'],
             'persons': ['2', '4', 'more'],
             'lug_boot': ['small', 'med', 'big'],
             'safety': ['low', 'med', 'high']}


# fetch the column number by attr_name
def attr_name_idx(idx):
    attr_names1 = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    func_val = attr_names1.index(idx)  # column number for each attribute
    return func_val

print(attr_name_idx('persons')) # get the index of the

def igcalculation(df):
    """ On input:
        a M by N matrix, the last column is the labels
        other columns are attributes.

        On output:
        a list contains the calculated information gain
        in the order that the attributes locate in the
        columns of the input matrix.
        In such form ([entropy_based IG], [Gini_based IG], [ME_based IG])
    """
    # calculate the length of the examples
    length = len(df)
    # compute the weighted average entropy, Gini index, and majorityerror
    wa_entropy = []
    wa_GI = []
    wa_ME = []
    # find the column index of the label
    df_cols = df.shape[1] - 1

    for i in range(0, df_cols):
        b = df.loc[:, [i, df_cols]]
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
            proportion.append(sum(m0) / length)
            # proportion of labels in each feature
            m1 = m0 / sum(m0)
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
                majority_error = 1 - max(m0) / sum(m0)
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
    values = df.loc[:, [df_cols]]
    # count the number of elements in the value column
    counted_values = values.value_counts(sort=False, normalize=True)

    value_ratio = counted_values.to_numpy()
    s1 = np.log2(value_ratio)
    s = np.dot(-1 * s1, value_ratio)
    # entropy for the whole examples
    # compute the entropy-based information gain
    IG_entropy = s * np.ones(len(wa_entropy)) - wa_entropy

    ky = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    # Final answer for the entropy-based information gain
    EN = dict(zip(ky, IG_entropy))

    # Compute the Gini_index-based information gain
    g_1 = 1 - np.dot(value_ratio, value_ratio)
    IG_gini = g_1 * np.ones(len(wa_GI)) - wa_GI

    # Compute the MajorityError
    me1 = 1 - max(value_ratio) / sum(value_ratio)
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
    index_1b.reverse() # The index of the largest IG is listed first
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
    return attr_names[best_attr], mm    # return a tuple, (best attr name, a list of indicies in dicts for attr_value)


v1 = igcalculation(df)[0]
v2 = get_split(df, v1)

print(v2)
# attribute value: print(list(Attr_dict.values()))
# attribute name: print(list(Attr_dict.keys()))


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
            sub_attr_name = split2[0]    # the attribute name for next round of splitting
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


# Test the decision tree building function
trial_of_decision_tree = decision_tree(df, Attr_dict, 2, 0)
pprint.pprint(trial_of_decision_tree)


#-------------------# error estimation and prediction#--------------------------------#
# test if an instance (a row vector, in the form of a list) belongs to a decision recursively in a given decision tree

def single_tree_modification(training_data,  stump_tree):
    """
    :param training_data: a dataframe, last column corresponds to the label
    :param stump_tree: a decision tree with only two layers, tree-depth = 1
    :return: a modified stump_tree with the major label for  attribute value which is None
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
        sub_label.sort()

        major_label = sub_label[0]
        if stump_tree[attribute][attr_values[j]] is None:
            stump_tree[attribute][attr_values[j]] = major_label # take the major label
    return stump_tree


def decision_tree_modification(dt, df):
    ky1 = list(dt.keys())
    if len(ky1) == 1:
        dt[ky1[0]] = single_tree_modification(df, dt)
    else:
        for ky2 in ky1:
            dt[ky2] = decision_tree_modification(dt[ky2], df)

    return dt


m = decision_tree_modification(trial_of_decision_tree, df)
pprint.pprint(m)


#a = single_tree_modification(df,  trial_of_decision_tree)
#pprint.pprint(a)


def label_check(dt, inst):
    '''
    Description: check whether an instance belong to a decision tree.
    On input:
            dt: the generated decision trees
            inst: a single instance in a list
    On output:
            count: count = 1, if the instance matches with the decision tree. Otherwise, count = 0.
    '''
    count = 0              # set the initial count to be 0.
    k = list(dt.keys())[0] # take the key in the dict and let it serve as a variable.
    idx = attr_name_idx(k) # take the column idx of the attribute.

    if dt[k][inst[idx]] == inst[-1]:
        count = 1

    elif isinstance(dt[k][inst[idx]], dict):
        dt_sub = dt[k][inst[idx]]  # recursively fetching
        count = label_predict(dt_sub, inst)

    return count


# predict the label of an instance

def label_predict(dt, inst):
    '''
    Description: check whether an instance belong to a decision tree.
    On input:
            dt: the generated decision trees
            inst: a single instance in a list
    On output:
            predicted label, according to the attributes of the instance via the given decision tree
    '''
    k = list(dt.keys())[0] # take the key in the dict and let it serve as a variable.
    idx = attr_name_idx(k) # take the column idx of the attribute.

    if isinstance(dt[k][inst[idx]], dict):
        dt_sub = dt[k][inst[idx]]  # recursively fetching
        label = label_predict(dt_sub, inst)
    else:
        label = dt[k][inst[idx]]

    return label


instance = ['1', '2', '3', '4', '5', 'low', 'unacc']
instance1 = ['med', 'low', '2', '2', 'small', 'high', 'unacc']

ccc = label_check(trial_of_decision_tree, instance)
# print(ccc)

# predict the label
predicted = []
for j in range(len(df)):
    itx = df.loc[j, :].tolist()
    num = label_predict(trial_of_decision_tree, itx)
    predicted.append(num)

# get the true lable of the data
true = df.loc[:, df.shape[1]-1].tolist()

# error calculation between the predicted labels and the true labels


def err_calculation(true_label, predicted_label):
    count = 0  # correct predication count
    for i in range(len(true_label)):
        if true_label[i] != predicted_label[i]:
            count += 1
    return count / float(len(true_label))


error = err_calculation(true, predicted)
print(error)

# =========================


def err_estimation(df, decision_tree):
    y = []
    for i in range(len(df)):
        inst = df.loc[i, :].tolist()
        k = label_predict(decision_tree, inst)
        if k == df.loc[i, df.shape[1]-1]: # the predicted label matches with the true label
            y.append(1)
        else:
            y.append(0)
    error = 1-sum(y)/len(df)
    return error # a value

# decision_tree1 = decision_tree(df, Attr_dict, 7, 0)
# error = err_estimation(df, decision_tree1)
# print(error)
