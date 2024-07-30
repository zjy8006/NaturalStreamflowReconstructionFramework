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
from HydroPy.Preprocessing.OneShotSamplesGenerator import gen_one_out_samples
from HydroPy.Preprocessing.SamplesSpliter import calibration_test_split
from HydroPy.Preprocessing.Normalizer import StandardScale,MinMaxScale,MaxAbsScale
from HydroPy.DataTools.Dataset import SequenceDataset
from HydroPy.NeuralNetwork.EarlyStopping import EarlyStopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytorch version 2.3.0
# Optuna version  3.6.1 

class LSTMRegressor(nn.Module):
    """ PyTorch LSTM Regressor (# * passed)
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
    def __init__(self, input_size:int,hidden_size:int, num_layers:int,output_size:int, lstm_dropout:float,dense_dropout:float):
        super(LSTMRegressor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm_dropout = lstm_dropout
        self.dense_dropout = dense_dropout

        if self.num_layers == 1:
            self.lstm_dropout = 0 
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.lstm_dropout,
            batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dense_dropout)

    # Formard method to predict values    
    def forward(self, x): # x is the input data
        # print('x.shape=',x.shape)
        # define the initial hidden state and cell state
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE))

        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE))

        # print('h0.shape=',h0.shape)
        # print('c0.shape=',c0.shape)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        y = hn.view(-1, self.hidden_size)

        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]

        out = self.fc(final_state)
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

def train_model_CV(
        dataset:torch.utils.data.Dataset, 
        model:nn.Module, 
        loss_function:torch.nn.modules.loss._Loss, 
        optimizer:torch.optim.Optimizer,
        batch_size:int=64,
        shuffle:bool=True,
        cv:int=5,
        random_state:int=42
):
    kf = KFold(n_splits=cv,shuffle=shuffle,random_state=random_state)
    for fold,(train_idx,val_idx) in enumerate(kf.split(dataset.X)):
        print(f"Fold {fold}")
        x_train, x_val = dataset.X[train_idx], dataset.X[val_idx]
        y_train, y_val = dataset.y[train_idx], dataset.y[val_idx]

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        # Early stopping variable
        best_loss = float('inf')
        early_stoping_counter = 0

        # Training loop
        EPOCHS = 1000
        epoch = 0
        done = False
        es = EarlyStopping(patience=50,min_delta=0,restore_best_weights=True)
        while not done and epoch < EPOCHS:
            epoch += 1
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.inference_mode():
                val_output = model(x_val)
                val_loss = loss_function(val_output, y_val)

            if es(model,val_loss):
                done = True
        print(f"Epoch {epoch}/{EPOCHS}, Validation Loss: "f"{val_loss.item()}, {es.status}")
    # Final evaluation
    model.eval()
    with torch.inference_mode():
        oos_pred = model(x_val)
    score = torch.sqrt(loss_function(oos_pred, y_val)).item()
    print(f"Fold score (RMSE): {score}")

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

def save_best_trial(best_trial: optuna.trial.Trial,model_path: str):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open("{}.pickle".format(model_path+'best_trial'), "wb") as fout:
        pickle.dump(best_trial, fout)

def load_best_trial(model_file: str):
    # Load a trained model from a file.
    with open(model_file, "rb") as fin:
        best_trial = pickle.load(fin)
    return best_trial

def save_model(trial: optuna.trial.Trial,model_path: str):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    input_size = trial.user_attrs["input_size"] # the number of input features
    output_size = trial.user_attrs["output_size"] # the number of output features

    model = LSTMRegressor(
        input_size=trial.user_attrs["input_size"],
        hidden_size=trial.params["hidden_size"],
        num_layers=trial.params["num_layers"],
        output_size=trial.user_attrs["output_size"],
        lstm_dropout=trial.params["lstm_dropout"],
        dense_dropout=trial.params["dense_dropout"],
    )

    model.load_state_dict(trial.user_attrs["best_model_state"])

    # Save a trained model to a file.
    with open("{}.pickle".format(model_path+'model'), "wb") as fout:
        pickle.dump(model, fout)

def load_model(model_file: str):
    # Load a trained model from a file.
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



def plot_predictions(cal_pred,cal_y,test_pred,test_y,cal_index,test_index):
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

def plot_observed_forecasted_scatters(cal_pred,cal_y,test_pred,test_y,cal_index,test_index):
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
        # Get the input and output sizes
        input_size = self.n_input
        output_size = self.n_out

        trial.set_user_attr("input_size", input_size) # the number of input features
        trial.set_user_attr("output_size", output_size) # the number of output features

        # Set the hyperparameters to tune
        num_layers = trial.suggest_int(name = 'num_layers', low=1, high=3, step=1)
        hidden_size = trial.suggest_int(name = 'hidden_size', low=8, high=128,step=8) 
        lstm_dropout = trial.suggest_float(name='lstm_dropout', low=0.0, high=0.5)
        dense_dropout = trial.suggest_float(name='dense_dropout', low=0.0, high=0.5)
        learning_rate = trial.suggest_float(name='learning_rate', low=1e-5, high=1e-1,log=True)

        # Initialize a DNNRegressor model
        model = LSTMRegressor(
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

        # Define the data loaders
        train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=self.shuffle)
        val_loader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=self.shuffle)

        # Restorethe best model state dict during the current trial
        best_model_state = model.state_dict()

        # print('best_model_state=',best_model_state)

        # Define the best validation loss
        best_val_loss = float('inf')

        # Store the training and validation loss of each epoch
        train_loss_values = []
        val_loss_values = []

        # Train and validate the model        
        for epoch in range(self.num_epoch):
            # Train the model
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                # print('data.shape=',data.shape)
                # data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1)

                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() # sum up batch loss

            avg_train_loss = train_loss / len(train_loader) # average training loss of each epoch
            train_loss_values.append(avg_train_loss) # store the average training loss of each epoch

            # Evaluate the model on the validation set 
            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for batch_idx, (data, target) in enumerate(val_loader):
                    # data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                    data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1)
                    output = model(data)
                    loss = loss_fn(output, target)
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                val_loss_values.append(avg_val_loss) # store the average validation loss of each epoch

            trial.report(avg_val_loss, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())

            # print('best_model_state=',best_model_state)

            trial.set_user_attr("best_model_state", best_model_state)
            trial.set_user_attr("train_loss_values", train_loss_values)
            trial.set_user_attr("val_loss_values", val_loss_values)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        return avg_val_loss


class Objective_CV:
    """ Optimize the hyperparameters of pytorch LSTM model using cross-validation (#!not passed)
    
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

        # Initialize a LSTMRegressor model
        model = LSTMRegressor(
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
    # !Get time series data
    df = pd.read_csv('data/ZhangjiashanRunoffVMD.csv', index_col=['time'], parse_dates=['time'],date_format="%b-%y")

    target_name = 'R'

    test_start_index = df.index[int(df.shape[0] * (1-0.2))]

    df_train = df.loc[df.index < test_start_index].copy()
    df_test = df.loc[df.index >= test_start_index].copy()

    target_mean = df_train[target_name].mean()
    target_std = df_train[target_name].std()

    for c in df_train.columns:
        mean = df_train[c].mean()
        std = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / std
        df_test[c] = (df_test[c] - mean) / std

    # !Train and validate the the Pytorch LSTM model without optuna and cross validation


    # !Tune the hyperparameters of pytorch lstm without cross validation
    # Define a optuna obejctive object to tune hyperparameters
    objective = Objective(
        train_dataframe=df_train,
        val_dataframe=df_test,
        target_name=target_name,
        leadtime=1,
        sequence_length=6,
        num_epoch=1000,
        batch_size=64,
        shuffle=True,
        model_path='./scheme/LSTMRegressor/',
    )
    study = optuna.create_study(
        study_name='example-study',
        direction='minimize',
    )
    study.optimize(objective, n_trials=5)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    best_model_state = trial.user_attrs["best_model_state"]
    # plot_train_val_loss_cv(
    #     trial.user_attrs["train_loss_df"],
    #     trial.user_attrs["val_loss_df"],
    # )

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    # The line of the resumed trial's intermediate values begins with the restarted epoch.
    # plot_intermediate_values_cv(study)
    # optuna.visualization.plot_intermediate_values(study).show()


    save_model(trial,model_path='./scheme/DNNRegressor/')

    model = load_model(model_file='./scheme/DNNRegressor/model.pickle').to(DEVICE)

    cal_dataset = SequenceDataset(
        dataframe=df_train,
        target=target_name,
        features=list(df_train.columns.difference([target_name])),
        lead_time=1,
        sequence_length=6,
    )
    test_dataset = SequenceDataset(
        dataframe=df_test,
        target=target_name,
        features=list(df_test.columns.difference([target_name])),
        lead_time=1,
        sequence_length=6,
    )
    

    cal_y_pred = predict(DataLoader(cal_dataset), model).cpu()
    test_y_pred = predict(DataLoader(test_dataset), model).cpu()

    # Renoemalize the prediction and target
    cal_y_pred = cal_y_pred * target_std + target_mean
    test_y_pred = test_y_pred * target_std + target_mean
    cal_dataset.y = cal_dataset.y.cpu() * target_std + target_mean
    test_dataset.y = test_dataset.y.cpu() * target_std + target_mean

    print(r2_score(cal_dataset.y, cal_y_pred))
    print(r2_score(test_dataset.y, test_y_pred))

    plot_predictions(cal_y_pred,cal_dataset.y,test_y_pred,test_dataset.y,df_train.index,df_test.index)

    plot_observed_forecasted_scatters(cal_y_pred,cal_dataset.y,test_y_pred,test_dataset.y,df_train.index,df_test.index)

        # print('train_dataset.y.shape[1]',train_dataset.y.shape[1])

    # objective = Objective(
    #     train_dataset=train_samples,
    #     val_dataset=val_samples,
    #     target_name=target_name,
    #     num_epoch=1000,
    #     batch_size=64,
    #     shuffle=True,
    # )
    # study = optuna.create_study(
    #     study_name='example-study',
    #     direction='minimize',
    # )
    # study.optimize(objective, n_trials=20)

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trial

    # best_model_state = trial.user_attrs["best_model_state"]

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

    # # The line of the resumed trial's intermediate values begins with the restarted epoch.
    # optuna.visualization.plot_intermediate_values(study).show()

    

    # objective = Objective(train_dataset=train_samples,test_dataset=test_samples,target_name=target_name,num_epoch=1000,batch_size=64,shuffle=True)
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=100)



    


# def objective(trial):
#     # Define hyperparameters to tune
#     n_layers = trial.suggest_int('n_layers', 1, 5)
#     dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    
#     # Define model and optimizer
#     model = DNNRegressor(input_dim, output_dim, n_layers, dropout_rate)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Train model
#     for epoch in range(num_epochs):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             optimizer.zero_grad()
#             output = model(data)
#             loss = nn.MSELoss()(output, target)
#             loss.backward()
#             optimizer.step()
    
#     # Evaluate model on validation set
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0.0
#         for data, target in val_loader:
#             output = model(data)
#             val_loss += nn.MSELoss()(output, target).item()
#         val_loss /= len(val_loader)
    
#     return val_loss

# # Define data and training parameters
# input_dim = ...
# output_dim = ...
# train_loader = ...
# val_loader = ...
# num_epochs = ...

# # Use Optuna to tune hyperparameters
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# # Get best hyperparameters and train final model
# best_params = study.best_params
# model = DNNRegressor(input_dim, output_dim, best_params['n_layers'], best_params['dropout_rate'])
# optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
# for epoch in range(num_epochs):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = nn.MSELoss()(output, target)
#         loss.backward()
#         optimizer.step()
