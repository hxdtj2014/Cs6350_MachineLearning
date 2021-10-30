# Data preprocessing
# import the csv file
import numpy as np
import pandas as pd


# ==================================pre-processing the training data=====================
# read the data as a dataframe
df = pd.read_csv(r'C:\Users\nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Project\Mid-term report\train_final.csv', header=None)

# delete the header in the first row for convenience
df.drop(0, axis=0, inplace=True)
df.reset_index(inplace=True, drop=True)

num_set = [0, 2, 4, 10, 11, 12] # indices of the numerical attributes


# replace the numerical features by the binary ones
# ========================== creat a bin_size function===========================

def bin_num(dataset,column_num, bin_size):
    """
    Description: bin the numeric datas into bins with the size of bin_size.
    :param dataset: the input, a dataframe
    :param column_num (an integer): the column indices of the column with numerical values
    :param bin_size(a positive integer): the size of bin
    :return: the modified df
    """
    digitized = (dataset.loc[:, column_num]).values.tolist()
    new = [int(i) for i in digitized]
    new.sort()   # sort the data in ascending order

    # bin size, an integer
    step = round(len(dataset)/(1+bin_size))

    numbers = [int(i) for i in digitized]
    numbers.sort()
    bins = [numbers[step*i] for i in range(1+bin_size)]
    num = [int(dataset.loc[i, column_num]) for i in range(len(dataset))]

    for i in range(len(dataset)):
        for k in range(len(bins)-1):
            if bins[k+1] > num[i] >= bins[k]:
                dataset.loc[i, column_num] = 'bin' + str(k+1)
                # print(dataset.loc[i, column_num])
            elif num[i] >= bins[k+1]:
                dataset.loc[i, column_num] = 'bin' + str(k+1)

    return dataset

# convert all the numeric values in the training data into bin numbers, which is adjustable
for j in num_set:
    bin_num(df, j, 10)


# convert the output (the last column of the df) of the dataframe
# to be either 1 or -1 (binaries), corresponding to 'yes' or 'no', respectively.
for i in range(df.shape[0]):
    if float(df.loc[i, df.shape[1]-1]) == 1.0:
        df.loc[i, df.shape[1]-1] = 1  # use integer dtype
    else:
        df.loc[i, df.shape[1]-1] = -1

# print(df)


#  =================== convert the numerical values to the binary values, i.e., 'yes', or 'no'.
rows = df.shape[0] # rows of the dataframe
cols = df.shape[1] # cols of the dataframe
median_of_num = [] # the median of the numerical features, serving as the threshold
for i in num_set:
    t = df.loc[:, i].mean()  # can be changed
    median_of_num.append(t)
    for j in range(rows):
        if float(df.loc[j, i]) >= t:

            df.loc[j, i] = 'yes'
        else:
            df.loc[j, i] = 'no'


# ==================================pre-processing the test data=====================
# read the data as a dataframe
test = pd.read_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Project\Mid-term report/test_final.csv', header=None)

# delete the header in the first row for convenience

test.drop(0, axis = 1, inplace=True)
test.drop(0, axis = 0, inplace=True)
test.reset_index(inplace = True, drop = True)
# print(test)

df_t = pd.DataFrame(np.zeros((len(test), test.shape[1])))
# let the column index starts from 0
for i in range(test.shape[0]):
    for j in range(test.shape[1]):
        df_t.loc[i,j]=test.loc[i,j+1]

num_set = [0, 2, 4, 10, 11, 12] # indices of the numerical attributes
# print(df_t)

'''
# convert the numerical values to the binary values, i.e., 'yes', or 'no'.
rows = df_t.shape[0] # rows of the dataframe
cols = df_t.shape[1] # cols of the dataframe
median_of_num = [] # the median of the numerical features, serving as the threshold
# replace the numerical features by the binary ones
for i in num_set:
    t = df_t.loc[:, i].mean() #can be changed
    median_of_num.append(t)
    for j in range(rows):
        if float(df_t.loc[j, i]) >= t:

            df_t.loc[j, i] = 'yes'
        else:
            df_t.loc[j, i] = 'no'

'''

'''
# convert all the numeric values in the training data into bin numbers
for j in num_set:
    bin_num(df_t, j, 10)
'''
# for i in range(df.shape[0]):
    # df.loc[i,df.shape[1]-1] = int(df.loc[i,df.shape[1]-1])


print(df_t)


# save df as training data
df.to_csv(r'C:/Users/nanji/OneDrive/桌面/CS6350 Machine Learning'
          r'/Project/Mid-term report/train_median.csv', index=False)
# save df_t as test data
df_t.to_csv(r'C:/Users/nanji/OneDrive/桌面/CS6350 Machine Learning'
            r'/Project/Mid-term report/test_median.csv', index=False)

"""
# ====================================attribute names and attribute values===================
attr_names = ['age', 'workclass','fnlwgt','education','education-num',
              'marital-status', 'occupation','relationship','race','sex',
              'capital-gain','capital-loss','hours-per-week','native-country']

Attr_dict = {}
for i in range(df.shape[1] - 1):
    Attr_dict[attr_names[i]] = list(set(df.loc[:, i]))

"""
