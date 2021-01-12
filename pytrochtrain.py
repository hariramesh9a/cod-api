import pandas as pd
import torch
import torch.nn as nn
from math import sqrt
import numpy as np
import torch.utils.data as data_utils
from sklearn import preprocessing

file_path = "/Users/harumughan/Downloads/bench_work/Hackathon_cod/User-1-data.csv"
df = pd.read_csv(file_path, header=None, usecols=[1, 2, 3, 4, 5, 6, 7],
                 names=['starting_balance', 'credit', 'amount', 'current_total', 'day1', 'day2', 'day3'])
df.drop(df.tail(3).index, inplace=True)

# Test and train data this si for Day 1

target = pd.DataFrame(df['day1'])
print(df[:1])


#
# train = data_utils.TensorDataset(torch.Tensor(np.array(df)), torch.Tensor(np.array(target)))
# train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)


class spendDataset(torch.utils.data.Dataset):
    """This class is the dataset class which is used to load data for training the LSTM
    to forecast timeseries data
    """

    def __init__(self, inputs, outputs):
        """Initialize the class with instance variables
        Args:
            inputs ([list]): [A list of tuples representing input parameters]
            outputs ([list]): [A list of floats for the stock price]
        """
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        """Returns the total number of samples in the dataset
        """
        return len(self.outputs)

    def __getitem__(self, idx):
        """Given an index, it retrieves the input and output corresponding to that index and returns the same
        Args:
            idx ([int]): [An integer representing a position in the samples]
        """
        x = torch.FloatTensor(self.inputs[idx])
        y = torch.FloatTensor([self.outputs[idx]])

        return (x, y)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def getDL(x, y, params):
    """Given the inputs, labels and dataloader parameters, returns a pytorch dataloader
    Args:
        x ([list]): [inputs list]
        y ([list]): [target variable list]
        params ([dict]): [Parameters pertaining to dataloader eg. batch size]
    """
    training_set = spendDataset(x, y)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    return training_generator


def standardizeData(X, SS=None, train=False):
    """Given a list of input features, standardizes them to bring them onto a homogenous scale
    Args:
        X ([dataframe]): [A dataframe of all the input values]
        SS ([object], optional): [A StandardScaler object that holds mean and std of a standardized dataset]. Defaults to None.
        train (bool, optional): [If False, means validation set to be loaded and SS needs to be passed to scale it]. Defaults to False.
    """
    if train:
        SS = preprocessing.StandardScaler()
        new_X = SS.fit_transform(X)
        return (new_X, SS)
    else:
        new_X = SS.transform(X)
        return (new_X, None)


class forecasterModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_lyrs=1, do=.05, device="cpu"):
        """Initialize the network architecture
        Args:
            input_dim ([int]): [Number of time lags to look at for current prediction]
            hidden_dim ([int]): [The dimension of RNN output]
            n_lyrs (int, optional): [Number of stacked RNN layers]. Defaults to 1.
            do (float, optional): [Dropout for regularization]. Defaults to .05.
        """
        super(forecasterModel, self).__init__()

        self.ip_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_lyrs
        self.dropout = do
        self.device = device

        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_lyrs, dropout=do)
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim / 2))
        self.act1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_features=int(hidden_dim / 2))

        self.estimator = nn.Linear(in_features=int(hidden_dim / 2), out_features=1)

    def init_hiddenState(self, bs):
        """Initialize the hidden state of RNN to all zeros
        Args:
            bs ([int]): [Batch size during training]
        """
        return torch.zeros(self.n_layers, bs, self.hidden_dim)

    def forward(self, input):
        """Define the forward propogation logic here
        Args:
            input ([Tensor]): [A 3-dimensional float tensor containing parameters]
        """
        bs = input.shape[1]
        hidden_state = self.init_hiddenState(bs).to(self.device)
        cell_state = hidden_state

        out, _ = self.rnn(input, (hidden_state, cell_state))

        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.act1(self.bn1(self.fc1(out)))
        out = self.estimator(out)

        return out

    def predict(self, input):
        """Makes prediction for the set of inputs provided and returns the same
        Args:
            input ([torch.Tensor]): [A tensor of inputs]
        """
        with torch.no_grad():
            predictions = self.forward(input)

        return predictions


def train():
    n_epochs = 100
    model = forecasterModel(6, 100)
    del df["day1"]
    std_Data = standardizeData(df,None,True)
    loss_func = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=.01)

    # Track the losses across epochs
    train_losses = []
    valid_losses = []

    training_generator = getDL(df,target, {"batch_size": 10})
    for epoch in range(1, n_epochs + 1):
        ls = 0
        valid_ls = 0
        # Train for one epoch
        for xb, yb in training_generator:
            # Perform the forward pass operation
            ips = xb.unsqueeze(0)
            targs = yb
            op = model(ips)

            # Backpropagate the errors through the network
            optim.zero_grad()
            loss = loss_func(op, targs)
            loss.backward()
            optim.step()
            ls += (loss.item() / ips.shape[1])

        # Check the performance on valiation data
        for xb, yb in training_generator:
            ips = xb.unsqueeze(0)
            ops = model.predict(ips)
            vls = loss_func(ops, yb)
            valid_ls += (vls.item() / xb.shape[1])

        rmse = lambda x: round(sqrt(x * 1.000), 3)
        train_losses.append(str(rmse(ls)))
        valid_losses.append(str(rmse(valid_ls)))


train()
# predict()
