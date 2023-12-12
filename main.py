# Call the dataset in the same fashion as tstream
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils

if __name__ == '__main__':
    
    # Define seed
    torch.manual_seed(0)
    
    # Hyperparams
    test_size = 0.2
    batch_size = 128
    src_variables = ['X']
    tgt_variables = ['y']
    input_variables = src_variables + tgt_variables
    timestamp_col_name = "time"
    
    features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate'
    
    # Get device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')
    
    # Read the data
    data = utils.read_data(timestamp_col_name=timestamp_col_name)
    data.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Adapt to prediction task
    if features == 'M' or features == 'MS':
        cols_data = data.columns[0:]
        data = data[cols_data]
    elif features == 'S':
        data = data[[tgt_variables[0]]]
    
    # Extract train and test data
    training_data = data[:-(round(len(data)*test_size))]
    testing_data = data[(round(len(data)*(1-test_size))):]
    
    # Scale the data
    scaler = StandardScaler()
    
    # Fit scaler on the training set
    scaler.fit(training_data.values)
    
    training_data.iloc[:, 0:] = scaler.transform(training_data.values)
    testing_data.iloc[:, 0:] = scaler.transform(testing_data.values)
    
    