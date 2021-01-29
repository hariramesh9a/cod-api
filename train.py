import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from hbaseapi import save_dict

file_path = "/Users/harumughan/Downloads/bench_work/Hackathon_cod/User-1-data.csv"
sdf = pd.read_csv(file_path, header=None, usecols=[1, 2, 3, 4, 5, 6, 7],
                  names=['starting_balance', 'credit', 'amount', 'current_total', 'day1', 'day2', 'day3'])
sdf.drop(sdf.tail(3).index, inplace=True)


def standardizeData(X, SS=None, train=False):
    """Given a list of input features, standardizes them to bring them onto a homogenous scale
    Args:
        X ([dataframe]): [A dataframe of all the input values]
        SS ([object], optional): [A StandardScaler object that holds mean and std of a standardized dataset]. Defaults to None.
        train (bool, optional): [If False, means validation set to be loaded and SS needs to be passed to scale it]. Defaults to False.
    """
    if train:
        SS = MinMaxScaler(feature_range=(-1, 1))
        new_X = SS.fit_transform(X)
        return (new_X, SS)
    else:
        new_X = SS.transform(X)
        return (new_X, None)


dir = 'models/user1/'

mdl = '1'
target = pd.DataFrame(sdf['day' + mdl])


del sdf['day' + mdl]
df, sc = standardizeData(sdf, None, True)
filename = dir + 'scalar.sav'

print(df)
tgt, sp = standardizeData(target, None, True)
pickle.dump(sp, open(filename, 'wb'))

train = data_utils.TensorDataset(torch.Tensor(np.array(df)), torch.Tensor(np.array(tgt)))
train_loader = data_utils.DataLoader(train, batch_size=100, shuffle=True)


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


def train():
    model = LSTM(6, 10)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 150
    for i in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    model.eval()
    print(model.state_dict())
    save_dict("user1~model" + mdl, str(model.state_dict()))
    torch.save(model, "models/user1/mdl" + mdl)


def predict():
    model = torch.load("models/user1/mdl3")
    model.eval()
    test_inputs = df
    train = data_utils.TensorDataset(torch.Tensor(test_inputs))
    train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
    fut_pred = 1
    print(test_inputs)
    seq = torch.FloatTensor(test_inputs)

    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        print(model(seq).item())
        print(sp.inverse_transform(np.array(model(seq).item()).reshape(-1, 1))[0][0])


train()
predict()
