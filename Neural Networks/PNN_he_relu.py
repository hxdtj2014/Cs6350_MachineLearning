# import modules
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# prepare the data from csv file into the Dataset form in PyTorch
class BankNote:
    def __init__(self, data_path, mode):

        super(BankNote, self).__init__()

        raw_tr = np.loadtxt(os.path.join(data_path, 'train.csv'), delimiter=',')
        raw_te = np.loadtxt(os.path.join(data_path, 'test.csv'), delimiter=',')

        Xtr, ytr, Xte, yte = \
            raw_tr[:, :-1], raw_tr[:, -1].reshape(-1, 1), raw_te[:, :-1], raw_te[:, -1].reshape(-1, 1)

        if mode == 'train':
            self.X, self.y = Xtr, ytr
        elif mode == 'test':
            self.X, self.y = Xte, yte
        else:
            raise Exception("Error: Invalid mode option!")

    def __getitem__(self, index):
        # fetch the feature vectors and label of any sample
        return self.X[index, :], self.y[index, :]

    def __len__(self, ):
        # Return total number of samples.
        return self.X.shape[0]


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Initialize the weights with Xavier initialization and use tanh activation function
        # torch.nn.init.xavier_uniform_(self.weight)

        # Initialize the weights with he's initialization and use relu activation function
        torch.nn.init.kaiming_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_features,))

    def forward(self, X):
        y = torch.matmul(X, self.weight.T) + self.bias
        # y = F.linear(X, self.weight, self.bias)
        return y


class Net(nn.Module):
    def __init__(self, config):

        super(Net, self).__init__()

        layers_list = []

        for l in range(len(config) - 1):
            in_dim = config[l]
            out_dim = config[l + 1]
            layers_list.append(MyLinear(in_features=in_dim, out_features=out_dim))
        #
        self.net = nn.ModuleList(layers_list)

    def forward(self, X):
        h = X
        for l in range(len(self.net) - 1):
            layer = self.net[l]
            h = torch.relu(layer(h))   # switch to tanh or relu
        #
        h = self.net[-1](h)
        return h


def error_estimation(model, dataset, device):
    data_loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(data_loader))

    X = X.float().to(device)
    y = y.long().to(device).squeeze()

    with torch.no_grad():
        logits = model(X)
        yhat = logits.argmax(1)
        corr = torch.eq(yhat, y).int().sum()
        acc = 1 - corr.item() / len(dataset)
        return acc, yhat


# ================================= scripts for running ========================================
# change the config to change the size and the width [input_layer, hidden_layers, output_layer(two labels)]
config0 = [4, 5, 5, 2]


# consider different width and
layers = [3, 5, 9]
width = [5, 10, 25, 50, 100]
all_config = []
for j in layers:
    for i in width:
        hidden_layers = [i]*(j-1)
        config = [4] + hidden_layers + [2]
        all_config.append(config)

# print(all_config)

dataset_train = BankNote(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                         r'\Assignments\Assignment_4\A4_lectures/bank-note/', mode='train')
dataset_test = BankNote(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                         r'\Assignments\Assignment_4\A4_lectures/bank-note/', mode='test')

epochs = 50
learning_rate = 1e-3
reg = 1e-5
bz = 4

train_loader = DataLoader(dataset=dataset_train, batch_size=bz, shuffle=True)
train_prediction = []
test_prediction = []

training_errors = []
test_errors = []

for configuration in all_config:
    model = Net(configuration).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    for ie in range(epochs):

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.float().to(device)
            y = y.long().to(device).squeeze()  # the ground truth

            logits = model(X)
            loss = F.cross_entropy(logits, y)
            # loss = F.mse_loss(logits, y.float())
            # loss = torch.sum((logits.argmax(1)-y)**2)
            # print(logits)
            # print(logits.argmax(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # store the last prediction of the train data
            if ie == epochs - 1:
                train_prediction.append(logits.argmax(1))
                test_predict = error_estimation(model, dataset_test, device)[1]
                test_prediction.append(test_predict)

        if ie == epochs - 1:
            err_tr = error_estimation(model, dataset_train, device)[0]
            err_te = error_estimation(model, dataset_test, device)[0]
            training_errors.append(err_tr)
            test_errors.append(err_te)

"""
    if ie % 1 == 0:
        err_tr = error_estimation(model, dataset_train, device)[0]
        err_te = error_estimation(model, dataset_test, device)[0]
        print('r', 'Epoch #{}: '.format(ie+1), end='')
        print('bce={:.5f}, train_err={:.3f}, test_err={:.3f}'.format(loss.item(), err_tr, err_te))
# store the last prediction of the train dataset and test dataset into lists, respectively
final_results_train = torch.cat(train_prediction, dim=0)
final_results_test = (torch.cat(test_prediction, dim=0)).tolist()
"""

training_errors = np.array(training_errors)
err_tr0 = np.round(training_errors.reshape((3, 5)), 3)
err_tr = pd.DataFrame(err_tr0)
err_tr.set_axis(['3', '5', '9'], axis=0, inplace=True)
err_tr.set_axis(['5', '10', '25', '50', '100'], axis=1, inplace=True)


test_errors = np.array(test_errors)
err_te0 = np.round(test_errors.reshape((3, 5)), 3)
err_te = pd.DataFrame(err_te0)
err_te.set_axis(['3', '5', '9'], axis=0, inplace=True)
err_te.set_axis(['5', '10', '25', '50', '100'], axis=1, inplace=True)
print('The training errors are:')
print(err_tr)
print('\nThe test errors are:')
print(err_te)
