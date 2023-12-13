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
import dataset as ds

if __name__ == '__main__':
    
    # Define seed
    torch.manual_seed(0)
    
    # Hyperparams
    test_size = 0.2
    val_size = 0.1
    batch_size = 128
    src_variables = ['X']
    tgt_variables = ['y']
    input_variables = src_variables + tgt_variables
    timestamp_col_name = "time"
    
    encoder_sequence_len = 96 # length of input given to encoder
    decoder_sequence_len = 48 # length of input given to decoder
    output_sequence_length = 96 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
    window_size = encoder_sequence_len + output_sequence_length # used to slice data into sub-sequences
    step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
    
    features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate'
    
    # Get device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')
    
    # Read the data
    data = utils.read_data(timestamp_col_name=timestamp_col_name)
    data.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Extract train, test, and validation temporal data for position encoding
    training_data_pe = data[:-(round(len(data)*(test_size+val_size)))].iloc[:, :1]
    testing_data_pe = data.iloc[:, :1][(round(len(data)*(1-test_size-val_size))):(round(len(data)*(1-val_size)))]
    validation_data_pe = data.iloc[:, :1][(round(len(data)*(1-val_size))):]
    
    # Adapt to prediction task
    if features == 'M' or features == 'MS':
        cols_data = data.columns[1:]
        data = data[cols_data]
    elif features == 'S':
        data = data[[tgt_variables[0]]]
    
    # Extract train, test and validaiton data
    training_data = data[:-(round(len(data)*(test_size+val_size)))]
    testing_data = data[(round(len(data)*(1-test_size-val_size))):(round(len(data)*(1-val_size)))]
    validation_data = data[(round(len(data)*(1-val_size))):]
    
    # Scale the data
    scaler = StandardScaler()
    
    # Fit scaler on the training set
    scaler.fit(training_data.values)
    
    training_data = training_data.values
    testing_data = testing_data.values
    validation_data = validation_data.values
    
    # training_data = scaler.transform(training_data.values)
    # testing_data = scaler.transform(testing_data.values)
    # validation_data = scaler.transform(validation_data.values)
    
    # Extract positional encoding data
    training_data_pe = utils.positional_encoder(training_data_pe, time_encoding='time_frequency', frequency='D')
    testing_data_pe = utils.positional_encoder(testing_data_pe, time_encoding='time_frequency', frequency='D')
    validation_data_pe = utils.positional_encoder(validation_data_pe, time_encoding='time_frequency', frequency='D')
    
    # Make instance of the custom dataset class
    training_data = ds.AutoTransformerDataset(data=torch.tensor(training_data), data_pe=torch.tensor(training_data_pe),
                                            encoder_sequence_len=encoder_sequence_len, decoder_sequence_len=decoder_sequence_len,
                                            tgt_sequence_len=output_sequence_length)
    testing_data = ds.AutoTransformerDataset(data=torch.tensor(testing_data), data_pe=torch.tensor(testing_data_pe),
                                            encoder_sequence_len=encoder_sequence_len, decoder_sequence_len=decoder_sequence_len,
                                            tgt_sequence_len=output_sequence_length)
    validation_data = ds.AutoTransformerDataset(data=torch.tensor(validation_data), data_pe=torch.tensor(validation_data_pe),
                                            encoder_sequence_len=encoder_sequence_len, decoder_sequence_len=decoder_sequence_len,
                                            tgt_sequence_len=output_sequence_length)

    # Set up the dataloaders