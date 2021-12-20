""""
The Neural networks algorithm is used on PyTorch to attack the project.

"""
# import essential modules
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# preferred to use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# prepare the data from csv file into the Dataset form in PyTorch
class InputData:
    def __init__(self, data_path, mode):

        super(InputData, self).__init__()

        raw_tr = np.loadtxt(data_path, delimiter=',')
        raw_te = np.loadtxt(data_path, delimiter=',')

        Xtr, ytr = raw_tr[:, :-1], raw_tr[:, -1].reshape(-1, 1)
        Xte, yte = raw_te, raw_tr[:, -1].reshape(-1, 1)  # yte is dummy

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
        h = self.net[-1](h)
        return h


def error_estimation(model, dataset, device, mode):

    data_loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(data_loader))
    X = X.float().to(device)
    y = y.long().to(device).squeeze()

    with torch.no_grad():
        logits = model(X)
        yhat = logits.argmax(1)

        if mode == 'train':
            corr = torch.eq(yhat, y).int().sum()
            err = 1 - corr.item() / len(dataset)
            return err, yhat
        elif mode == 'test':
            return yhat

# ================================= scripts for running ========================================
# change the config to change the size and the width [input_layer, hidden_layers, output_layer(two labels)]
# config0 = [104, 105, 105, 2]
# consider different width and depth
# layers = [3, 5, 9]


# plot the training error curve , epoch=1
"""
# for prediction in OHE case
layers = [3]   # adjustable
width = [105]   # 5 times of the input 

# for prediction in bin counting case
layers = [3]   # adjustable
width = [15]  # 10 times of the input
"""


# try different layers and width to compute the training error
"""
# OHE case
layers = [3, 6, 9]   # adjustable
width = [105, 210, 525, 1050]  # adjustable
"""
"""
layers = [3, 6, 9]   # adjustable
width = [15, 30, 75, 150]  # adjustable 1, 2, 5, 10 times of the input size
"""

# use the best parameters to predict the test data
"""
# for prediction in OHE case
layers = [10]   # adjustable
width = [525]   # 5 times of the input 
"""
# for prediction in bin counting case
layers = [9]   # adjustable
width = [150]  # 10 times of the input


# construct the neural networks configuration
all_config = []
for j in layers:
    for i in width:
        hidden_layers = [i]*(j-1)
        config = [14] + hidden_layers + [2]  # the input size + hidden layer width + types of labels
        # config = [104] + hidden_layers + [2] # OHE case
        all_config.append(config)


# training data with labels
dataset_train = InputData(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                          r'\Project\FINAL_REPORT\processed_data/train_bin_count.csv', mode='train')
# test data is without labels
dataset_test = InputData(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                         r'\Project\FINAL_REPORT\processed_data/test_bin_count.csv', mode='test')

epochs = 50   # adjustable
learning_rate = 1e-3
reg = 1e-5
bz = 4  # batch size, set at 4.
train_loader = DataLoader(dataset=dataset_train, batch_size=bz, shuffle=True)
train_prediction = []
test_prediction = []
training_errors = []
test_errors = []

err_tr = 0
train_err_step = []  # training error for each step

predict_tr = 0
predict_te = 0
count = 0
total = len(layers)*len(width)
for configuration in all_config:
    model = Net(configuration).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    count = count + 1

    for ie in range(epochs):
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.float().to(device)
            y = y.long().to(device).squeeze()  # the ground truth

            logits = model(X)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """
            # remove the quotes for this part to plot the training error 
            # compute the training error for each iteration
            res_tr = error_estimation(model, dataset_train, device, mode='train')
            err_tr = res_tr[0]
            train_err_step.append(err_tr)
            """

        if ie == epochs - 1:
            res_tr = error_estimation(model, dataset_train, device, mode='train')
            err_tr = res_tr[0]
            training_errors.append(err_tr)
            predict_tr = res_tr[1].tolist()

            # store the predicted test results
            res_te = error_estimation(model, dataset_test, device, mode='test')
            predict_te = res_te.tolist()

    # monitoring the computing progress
    print(str(round(count/total, 2)*100) + "% completed")

# ============================ plot the training error for the training dataset ================
"""
print(train_err_step)
plt.plot(train_err_step, label='training', c='r', linewidth=1.0)
plt.xlabel('Iteration times', fontname="serif", weight="bold")
plt.ylabel('Errors', fontname="serif", weight="bold")
plt.xticks(fontname="serif", weight="bold")
plt.yticks(fontname="serif", weight="bold")
plt.title('Convergence check for OHE dataset', fontname="serif", weight="bold")
plt.legend()
plt.show()
"""


"""
# ================================ only compute the training error ===========================
training_errors = np.array(training_errors)
row = len(layers)
col = len(width)
err_tr0 = training_errors.reshape((row, col))
err_tr = pd.DataFrame(err_tr0)
err_tr.set_axis([str(i) for i in layers], axis=0, inplace=True)
err_tr.set_axis([str(i) for i in width], axis=1, inplace=True)
print('The training errors are:')
print(err_tr)
"""


# ================= compute the confusion matrix for the training dataset =====================
# compute the confusion matrix for the training dataset, change the file, either train_p(OHE) or
# train_bin_count(BC)
df = pd.read_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                 r'\Project\FINAL_REPORT\processed_data/train_bin_count.csv', header=None)

y_raw = (copy.deepcopy(df.iloc[:, -1])).to_numpy()
predict_tr = np.array(predict_tr)
tr_conf_mx = confusion_matrix(y_raw, predict_tr)
print(tr_conf_mx)
disp = ConfusionMatrixDisplay(confusion_matrix=tr_conf_mx)
disp.plot()
plt.show()


# ================================ store the prediction for the test data ========================
test_prediction = pd.DataFrame(predict_te)
print(err_tr)  # print the training error
# print(test_prediction)
test_prediction.to_csv(r'C:/Users/nanji\OneDrive\桌面\CS6350 Machine Learning'
                       r'\Project\FINAL_REPORT\processed_data/test_predict_bc9.csv', index=False, header=False)


