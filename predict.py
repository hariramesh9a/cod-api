import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
import pickle

from datetime import datetime, timedelta


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


def predict(amount=100, user=1):
    dir = "models/user" + str(user)
    test_inputs = np.array([500, -1, float(amount), 500 + float(amount), 0, 0]).reshape(1, -1)
    loaded_model = pickle.load(open(dir + '/scalar.sav', 'rb'))
    seq = torch.FloatTensor(test_inputs)
    output = {"id": 1, "label": [], "data": []}

    for x in range(1, 4):
        output["label"].append((datetime.today() + timedelta(days=x)).strftime('%Y%m%d'))
        model = torch.load(dir + "/mdl" + str(x))
        model.eval()
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            print(model(seq).item())
            output["data"].append(loaded_model.inverse_transform(np.array(model(seq).item()).reshape(-1, 1))[0][0])

    print(output)
    return output
