from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from optuna.trial import TrialState
from optuna.storages import RetryFailedTrialCallback
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import copy
import shutil
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = 'plotly_white'
plot_template = dict(
    layout=go.Layout({
        'font_size': 8,
        'xaxis_title_font_size': 8,
        'yaxis_title_font_size': 8,
        }   
))
import os
import sys
sys.path.append('.') # add parent path to sys.path
from OneShotSamplesGenerator import gen_one_out_samples
from SamplesSpliter import calibration_test_split
from Normalizer import StandardScale,MinMaxScale,MaxAbsScale
from Dataset import SequenceDataset
from EarlyStopping import EarlyStopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMRegressor(nn.Module):
    """ PyTorch Bi-LSTM Regressor
    Parameters
    ----------
    * `input_size`: [int, required]
        The number of input features
    * `hidden_size`:[int, required]
        The size of the hidden layer
    * `num_layers`: [int, required]
        The number of layers
    * `output_size`: [int, required]
        The number of output features
    * `lstm_dropout`: float
        The dropout rate for the LSTM layers
    * `dense_dropout`: float
        The dropout rate for the dense layers
    """
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, output_size:int, lstm_dropout:float, dense_dropout:float):
        super(BiLSTMRegressor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm_dropout = lstm_dropout
        self.dense_dropout = dense_dropout

        if self.num_layers == 1:
            self.lstm_dropout = 0 
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.lstm_dropout,
            batch_first=True,
            bidirectional=True)  # Enable bidirectional processing
        
        # Since we're using bidirectional LSTM, the hidden size is doubled
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        self.dropout = nn.Dropout(self.dense_dropout)

    def forward(self, x):
        # Initialize hidden state and cell state
        # Note: For bidirectional LSTM, we need to double the first dimension
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(DEVICE))
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(DEVICE))

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Concatenate the final forward and backward hidden states
        # hn shape: (num_layers * 2, batch_size, hidden_size)
        # We take the last layer's forward and backward states
        forward_hidden = hn[-2, :, :]  # Last layer forward hidden state
        backward_hidden = hn[-1, :, :]  # Last layer backward hidden state
        
        # Concatenate forward and backward hidden states
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        # Pass through fully connected layer
        out = self.fc(concat_hidden)
        out = self.dropout(out)
        return out

def train_model(
        data_loader:torch.utils.data.DataLoader, 
        model:nn.Module, 
        loss_function:torch.nn.modules.loss._Loss, 
        optimizer:torch.optim.Optimizer
):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")

def test_model(
        data_loader:torch.utils.data.DataLoader, 
        model:nn.Module, 
        loss_function:torch.nn.modules.loss._Loss
):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")

def save_best_trial(best_trial: optuna.trial.Trial, model_path: str):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open("{}.pickle".format(model_path+'best_trial'), "wb") as fout:
        pickle.dump(best_trial, fout)

def load_best_trial(model_file: str):
    with open(model_file, "rb") as fin:
        best_trial = pickle.load(fin)
    return best_trial

def save_model(trial: optuna.trial.Trial, model_path: str):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model = BiLSTMRegressor(
        input_size=trial.user_attrs["input_size"],
        hidden_size=trial.params["hidden_size"],
        num_layers=trial.params["num_layers"],
        output_size=trial.user_attrs["output_size"],
        lstm_dropout=trial.params["lstm_dropout"],
        dense_dropout=trial.params["dense_dropout"],
    ).to(DEVICE)

    model.load_state_dict(trial.user_attrs["best_model_state"])

    with open("{}.pickle".format(model_path+'model'), "wb") as fout:
        pickle.dump(model, fout)

def load_model(model_file: str):
    with open(model_file, "rb") as fin:
        model = pickle.load(fin)
    return model

def predict(data_loader: torch.utils.data.DataLoader, model: nn.Module):
    output = torch.tensor([]).to(DEVICE)
    model.eval()

    with torch.inference_mode():
        for X, _ in data_loader:
            X = X.to(DEVICE)
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output

def plot_predictions(cal_pred, cal_y, test_pred, test_y, cal_index, test_index):
    cal_df = pd.DataFrame()
    cal_df['Observed'] = cal_y.reshape(-1).tolist()
    cal_df['Forecasted'] = cal_pred.reshape(-1).tolist()
    cal_df.index = cal_index
    test_df = pd.DataFrame()
    test_df['Observed'] = test_y.reshape(-1).tolist()
    test_df['Forecasted'] = test_pred.reshape(-1).tolist()
    test_df.index = test_index

    fig = px.line(pd.DataFrame({
        'Calibration(Observed)': cal_df['Observed'],
        'Calibration(Forecasted)': cal_df['Forecasted'],
        'Test(Observed)': test_df['Observed'],
        'Test(Forecasted)': test_df['Forecasted'],
    }), labels={'time': '时间', 'value': '值'}, )
    fig.update_layout(
        template=plot_template,legend=dict(orientation='h',x=0,y=1.1))
    fig.show()

def plot_observed_forecasted_scatters(cal_pred, cal_y, test_pred, test_y, cal_index, test_index):
    cal_df = pd.DataFrame()
    cal_df['Observed'] = cal_y.reshape(-1).tolist()
    cal_df['Forecasted'] = cal_pred.reshape(-1).tolist()
    cal_df.index = cal_index
    test_df = pd.DataFrame()
    test_df['Observed'] = test_y.reshape(-1).tolist()
    test_df['Forecasted'] = test_pred.reshape(-1).tolist()
    test_df.index = test_index

    scatter1 = go.Scatter(
        x=cal_df['Observed'],
        y=cal_df['Forecasted'],
        mode='markers',
        name='Calibration',
        marker=dict(
            size=5,
            color='rgba(0, 0, 255, 1)',
            line=dict(
                width=2,
                color='rgb(0, 0, 0)'
            )
        )
    )

    scatter2 = go.Scatter(
        x=test_df['Observed'],
        y=test_df['Forecasted'],
        mode='markers',
        name='Test',
        marker=dict(
            size=5,
            color='rgba(255, 0, 0, 1)',
            line=dict(
                width=2,
            )
        )
    )

    fig = go.Figure(
        data=[scatter1,scatter2],
        layout=go.Layout(
        title='Observed-Forecasted Scatter',
        xaxis=dict(title='Observed'),
        yaxis=dict(title='Forecasted'),)
    )

    fig.show()

def plot_train_val_loss_cv(train_loss_df,val_loss_df,interval=None):
    if interval is None:
        train_loss_df = train_loss_df.iloc[::interval]
        val_loss_df = val_loss_df.iloc[::interval]

    train_loss_df_std = train_loss_df.std(axis=1)
    val_loss_df_std = val_loss_df.std(axis=1)

    epochs = np.arange(train_loss_df.shape[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss_df.min(axis=1),
        mode='lines',
        # name='Maximum train loss',
        line=dict(color='red',width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss_df.max(axis=1),
        mode='lines',
        # name='Minimum train loss',
        line=dict(color='red',width=0),
        showlegend=False,
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss_df.mean(axis=1),
        mode='lines',
        name='Average train loss',
        line=dict(color='red'),
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss_df.min(axis=1),
        mode='lines',
        # name='Minimum validation loss',
        line=dict(color='blue',width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss_df.max(axis=1),
        mode='lines',
        # name='Maximum validation loss',
        line=dict(color='blue',width=0),
        showlegend=False,
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.2)',
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss_df.mean(axis=1),
        mode='lines',
        name='Average validation loss',
        line=dict(color='blue'),
    ))

    # Set layout properties
    fig.update_layout(title='Training and Validation Losses', xaxis_title='Epochs', yaxis_title='Loss')
    fig.show()

class Objective:
    def __init__(self,
                 train_dataset:torch.utils.data.Dataset,
                 val_dataset:torch.utils.data.Dataset,
                 num_epoch:int=100,
                 batch_size:int=64,
                 shuffle:bool=True,
                 model_path:str=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.checkpoint_dir = self.model_path + '/checkpoint/'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.n_input = self.train_dataset.X.shape[1]
        self.n_out = self.train_dataset.y.reshape(-1, 1).shape[1]

    def __call__(self, trial: optuna.trial.Trial):
        input_size = self.n_input
        output_size = self.n_out

        trial.set_user_attr("input_size", input_size)
        trial.set_user_attr("output_size", output_size)

        num_layers = trial.suggest_int(name = 'num_layers', low=1, high=3, step=1)
        hidden_size = trial.suggest_int(name = 'hidden_size', low=8, high=128,step=8) 
        lstm_dropout = trial.suggest_float(name='lstm_dropout', low=0.0, high=0.5)
        dense_dropout = trial.suggest_float(name='dense_dropout', low=0.0, high=0.5)
        learning_rate = trial.suggest_float(name='learning_rate', low=1e-5, high=1e-1,log=True)

        model = BiLSTMRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            lstm_dropout=lstm_dropout,
            dense_dropout=dense_dropout,
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=self.shuffle)
        val_loader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=self.shuffle)

        best_model_state = model.state_dict()
        best_val_loss = float('inf')

        train_loss_values = []
        val_loss_values = []

        for epoch in range(self.num_epoch):
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1)

                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_loss_values.append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1)
                    output = model(data)
                    loss = loss_fn(output, target)
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                val_loss_values.append(avg_val_loss)

            trial.report(avg_val_loss, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())

            trial.set_user_attr("best_model_state", best_model_state)
            trial.set_user_attr("train_loss_values", train_loss_values)
            trial.set_user_attr("val_loss_values", val_loss_values)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        return avg_val_loss

class Objective_CV:
    """ Optimize the hyperparameters of pytorch Bi-LSTM model using cross-validation
    
    """
    def __init__(self, cal_dataset: torch.utils.data.Dataset, num_epoch: int = 100, batch_size: int = 64, shuffle: bool = True, cv: int = 5, random_state: int = 42, model_path: str = None):
        self.cal_dataset = cal_dataset
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kf = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        self.model_path = model_path
        self.n_input = self.cal_dataset.X.shape[1]
        self.n_out = self.cal_dataset.y.reshape(-1, 1).shape[1]

    def __call__(self, trial: optuna.trial.Trial):
        # Get the input and output sizes
        input_size = self.n_input
        output_size = self.n_out

        trial.set_user_attr("input_features", input_size) # the number of input features
        trial.set_user_attr("output_features", output_size) # the number of output features

        # Set the hyperparameters to tune
        num_layers = trial.suggest_int('num_layers', 1, 3)
        hidden_size = trial.suggest_int('hidden_size', 8, 128)
        lstm_dropout = trial.suggest_float('lstm_dropout', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        dense_dropout = trial.suggest_float('dense_dropout', 0.0, 0.5)

        # Initialize a BiLSTMRegressor model
        model = BiLSTMRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            lstm_dropout=lstm_dropout,
            dense_dropout=dense_dropout,
        ).to(DEVICE)

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Define the loss function
        loss_fn = nn.MSELoss()

        # Restore the best model state dict during the current trial
        best_model_state = model.state_dict()

        # Define the best validation loss
        best_val_loss = float('inf')

        # Define the average validation loss of all folds
        avg_val_loss = 0.0

        # Store the average training and validation loss of all folds
        train_loss_df = pd.DataFrame()
        val_loss_df = pd.DataFrame()

        # Define the cross validation
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.cal_dataset.X)):
            # Get the training and validation data
            x_train, x_val = self.cal_dataset.X[train_idx], self.cal_dataset.X[val_idx]
            y_train, y_val = self.cal_dataset.y[train_idx], self.cal_dataset.y[val_idx]

            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

            # Store the average training and validation loss of each fold
            train_loss_values = []
            val_loss_values = []

            for epoch in range(self.num_epoch):
                # Train the model
                model.train()
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()  # sum up batch loss
                avg_train_loss_ = train_loss / len(train_loader)  # average training loss of each epoch
                train_loss_values.append(avg_train_loss_)  # store the average training loss of each epoch

                # Validate the model
                model.eval()
                val_loss = 0.0
                with torch.inference_mode():
                    for batch_idx, (data, target) in enumerate(val_loader):
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        output = model(data)
                        loss = loss_fn(output, target)
                        val_loss += loss.item()

                    avg_val_loss_ = val_loss / len(val_loader)
                    val_loss_values.append(avg_val_loss_)  # store the average validation loss of each epoch

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            train_loss_df['Fold {}'.format(fold)] = train_loss_values
            val_loss_df['Fold {}'.format(fold)] = val_loss_values

            avg_val_loss += avg_val_loss_

        avg_val_loss /= self.kf.get_n_splits()

        trial.report(avg_val_loss, step=epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        trial.set_user_attr("best_model_state", best_model_state)
        trial.set_user_attr("train_loss_df", train_loss_df)
        trial.set_user_attr("val_loss_df", val_loss_df)

        return avg_val_loss

if __name__ == '__main__':
    pass
