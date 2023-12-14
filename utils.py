import numpy as np
import pandas as pd
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
from typing import Optional, Any, Union, Callable, Tuple

from timefeatures import time_features

def get_indices(data: pd.DataFrame, window_size: int, step_size: int):
    
    """
    Produce all the start and end index position that is needed to obtain the sub-sequences.
    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a subsequence. These tuples
    should be used to slice the dataset into sub-sequences. These sub-sequences should then be
    passed into a function that sliced them into input and target sequences.
    ----------
    Arguments:
    data (pd.DataFrame): loaded database to generate the subsequences from.
    window_size (int): the desired length of each sub-sequence. Should be (input_sequence_length + 
        tgt_sequence_length). E.g. if you want the model to consider the past 100 time steps in 
        order to predict the future 50 time_steps, window_size = 100 + 50 = 150.
    step_size (int): size of each step as the data sequence is traversed by the moving window.
    
    Return:
    indices: a lits of tuples.
    """
    
    # Define the stop position
    stop_position = len(data) - 1 # because of 0 indexing in Python
    
    # Start the first sub-sequence at index 0
    subseq_first_idx = 0
    subseq_last_idx = window_size
    
    indices = []
    while subseq_last_idx <= stop_position:
        
        indices.append((subseq_first_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        subseq_last_idx += step_size
    
    return indices

def positional_encoder(data, time_encoding, frequency):
    
    time_stamps = data[['time']]
    
    if time_encoding != 'time_frequency':
        time_stamps['month'] = time_stamps.time.apply(lambda row: row.month, 1)
        time_stamps['day'] = time_stamps.time.apply(lambda row: row.day, 1)
        time_stamps['weekday'] = time_stamps.time.apply(lambda row: row.weekday(), 1)
        data_pe = time_stamps.drop(['time'], axis=1).values
    else: # time_encoding = 'fixed' or 'learned'
        data_pe = time_features(pd.to_datetime(time_stamps['time'].values), freq=frequency)
        data_pe = data_pe.transpose(1, 0)
        
    return data_pe

def read_data(data_dir: Union[str, Path] = 'data/utah', timestamp_col_name: str='time') -> pd.DataFrame:
    
    """Read data from csv file and return a pd.DataFrame object.
    ----------
    Arguments:
    data_dir: str or Path object specifying the path to the directory containing the data.
    tgt_col_name: str, the name of the column containing the target variable
    timestamp_col_name: str, the name of the column or named index containing the timestamps
    
    Returns:
    data (pd.DataFrame): data read an loaded as a Pandas DataFrame
    """
    
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)
    
    # Read csv file
    csv_files = list(data_dir.glob("*.csv"))
    
    if len(csv_files) > 1:
        raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
    elif len(csv_files) == 0:
        raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))
    
    data = pd.read_csv(data_path, parse_dates=[timestamp_col_name],  low_memory=False)
    
    # Make sure all "n/e" values have been removed from df. 
    if ne_check(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    # Downcast columns to smallest possible version
    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[timestamp_col_name], inplace=True)

    return data

def ne_check(df:pd.DataFrame):
    
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False

def to_numeric_and_downcast_data(df: pd.DataFrame):
    
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

# Define function to get and format the number of parameters
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    
    print(table)
    print(f"Total trainable parameters: {total_params}")
    
    return total_params

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

# Define metrics
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

# Define Nash-Sutcliffe efficiency
def nash_sutcliffe_efficiency(observed, modeled):
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - modeled)**2)
    denominator = np.sum((observed - mean_observed)**2)
    
    nse = 1 - (numerator / denominator)
    
    return nse
