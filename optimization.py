import time
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
from models.autoformer import Autoformer
from models.informer import Informer
from models.fedformer import FEDformer

"""Hyperparameter tuning with Grid Search.
I cannot do Bayesian optimization because the 
model needs PyTorch == 1.9.0 and Ax requires 
PyTorch >= 2.11.1."""
# https://ax.dev/tutorials/tune_cnn_service.html

# Define train step
def train(dataloader, model, loss_function, optimizer, device, df_training, epoch):
    size = len(dataloader.dataset)
    model.train()
    training_loss = [] # For plotting purposes
    for i, batch in enumerate(dataloader):
        src, tgt, tgt_y, src_pe, tgt_pe = batch
        src, tgt, tgt_y, src_pe, tgt_pe = src.float().to(device), tgt.float().to(device), tgt_y.float().to(device), src_pe.float().to(device), tgt_pe.float().to(device)
        
        # Zero out gradients for every batch
        model.zero_grad()
        
        # Process decoder input
        decoder_input = torch.zeros_like(tgt[:, -output_sequence_len:, :]).float()
        decoder_input = torch.cat([tgt[:, :decoder_sequence_len, :], decoder_input], dim=1).float().to(device)
        
        # Compute prediction error
        pred, attention_weights = model(src, decoder_input, src_pe, tgt_pe)
        
        f_dim = -1 if features == 'MS' else 0
        pred = pred[:, -decoder_sequence_len:, f_dim:].to(device)
        tgt_y = tgt_y[:, -decoder_sequence_len:, f_dim:].to(device)
        
        loss = loss_function(pred, tgt_y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Save results for plotting
        training_loss.append(loss.item())
        epoch_train_loss = np.mean(training_loss)
        df_training.loc[epoch] = [epoch, epoch_train_loss]
        
        if i % 10 == 0:
            print('Current batch', i)
            loss, current = loss.item(), (i + 1) * len(src)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define test step
def test(dataloader, model, loss_function, device, df_testing, epoch):
    num_batches = len(dataloader)
    model.eval()
    testing_loss = [] # For plotting purposes
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, tgt_y, src_pe, tgt_pe = batch
            src, tgt, tgt_y, src_pe, tgt_pe = src.float().to(device), tgt.float().to(device), tgt_y.float().to(device), src_pe.float().to(device), tgt_pe.float().to(device)
            
            # Process decoder input
            decoder_input = torch.zeros_like(tgt[:, -output_sequence_len:, :]).float()
            decoder_input = torch.cat([tgt[:, :decoder_sequence_len, :], decoder_input], dim=1).float().to(device)
            
            # Compute prediction error
            pred, attention_weights = model(src, decoder_input, src_pe, tgt_pe)
            
            f_dim = -1 if features == 'MS' else 0
            pred = pred[:, -decoder_sequence_len:, f_dim:].to(device)
            tgt_y = tgt_y[:, -decoder_sequence_len:, f_dim:].to(device)
        
            loss = loss_function(pred, tgt_y)
            
            # Save results for plotting
            testing_loss.append(loss.item())
            epoch_test_loss = np.mean(testing_loss)
            df_testing.loc[epoch] = [epoch, epoch_test_loss]
    
    loss /= num_batches
    # print(f"Avg test loss: {loss:>8f}")

# Define validation step
def validation(dataloader, model):
    
    # Define lists to store the predictions and ground truth
    y_hats = []
    tgt_ys = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src, tgt, tgt_y, src_pe, tgt_pe = batch
            src, tgt, tgt_y, src_pe, tgt_pe = src.float().to(device), tgt.float().to(device), tgt_y.float().to(device), src_pe.float().to(device), tgt_pe.float().to(device)
            
            # Process decoder input
            decoder_input = torch.zeros_like(tgt[:, -output_sequence_len:, :]).float()
            decoder_input = torch.cat([tgt[:, :decoder_sequence_len, :], decoder_input], dim=1).float().to(device)
            
            # Compute prediction error
            pred, attention_weights = model(src, decoder_input, src_pe, tgt_pe)
            
            f_dim = -1 if features == 'MS' else 0
            pred = pred[:, -decoder_sequence_len:, f_dim:].to(device)
            tgt_y = tgt_y[:, -decoder_sequence_len:, f_dim:].to(device)
            
            y_hat = pred.detach().cpu().numpy()
            tgt_y = tgt_y.detach().cpu().numpy()
            
            y_hats.append(y_hat)
            tgt_ys.append(tgt_y)
    
    y_hats = np.concatenate(y_hats, axis=0)
    tgt_ys = np.concatenate(tgt_ys, axis=0)
    print('Validation shape:', y_hats.shape, tgt_ys.shape)
    y_hats = y_hats.reshape(-1, y_hats.shape[-2], y_hats.shape[-1])
    tgt_ys = tgt_ys.reshape(-1, tgt_ys.shape[-2], tgt_ys.shape[-1])
    print('Validation shape:', y_hats.shape, tgt_ys.shape)
    
    # Get metrics
    mae, mse, rmse, mape, mspe = utils.metric(y_hats, tgt_ys)
    print('MSE: {}\nMAE: {}'.format(mse, mae))
    
    return y_hats.squeeze(), tgt_ys.squeeze()

# Define function to train, test and evaluate the model
def train_test_val(lr, d_model, n_heads, n_encoder_layers, n_decoder_layers, encoder_features_fc_layer, decoder_features_fc_layer):

    # Build model
    if model_selection == 'autoformer':
        model = Autoformer(encoder_sequence_len=encoder_sequence_len, decoder_sequence_len=decoder_sequence_len, output_sequence_len=output_sequence_len,
                    encoder_input_size=encoder_input_size, decoder_input_size=decoder_input_size, decoder_output_size=decoder_output_size, 
                    encoder_features_fc_layer=encoder_features_fc_layer, decoder_features_fc_layer=decoder_features_fc_layer, n_encoder_layers=n_encoder_layers,
                    n_decoder_layers=n_decoder_layers, activation=activation, embed=embed, d_model=d_model, n_heads=n_heads, attention_factor=attention_factor,
                    frequency=frequency, dropout=dropout, output_attention=output_attention, moving_average=moving_average).float()
    elif model_selection == 'informer':
        model = Informer(output_sequence_len=output_sequence_len, encoder_input_size=encoder_input_size, decoder_input_size=decoder_input_size, 
                        decoder_output_size=decoder_output_size, encoder_features_fc_layer=encoder_features_fc_layer, decoder_features_fc_layer=decoder_features_fc_layer, 
                        n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers, activation=activation, embed=embed, d_model=d_model, n_heads=n_heads, 
                        attention_factor=attention_factor, frequency=frequency, dropout=dropout, distill=distill, output_attention=output_attention).float()
    elif model_selection == 'fedformer':
        model = FEDformer(encoder_sequence_len=encoder_sequence_len, decoder_sequence_len=decoder_sequence_len, output_sequence_len=output_sequence_len,
                        encoder_input_size=encoder_input_size, decoder_input_size=decoder_input_size , decoder_output_size=decoder_output_size, 
                        encoder_features_fc_layer=encoder_features_fc_layer, decoder_features_fc_layer=decoder_features_fc_layer, n_encoder_layers=n_encoder_layers, 
                        n_decoder_layers=n_decoder_layers, activation=activation, embed=embed, d_model=d_model, n_heads=n_heads, frequency=frequency, 
                        dropout=dropout, output_attention=output_attention, moving_average=moving_average, version=version, mode_select=model_selection, 
                        modes=modes, L=L, base=base, cross_activation=cross_activation, wavelet=wavelet).float()
    else:
        raise ValueError('Model not implemented')

    # Send model to device
    model.to(device)

    # Define optimizer and loss function
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Update model in the training process and test it
    epochs = 5 # 250
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_testing = pd.DataFrame(columns=('epoch', 'loss_test'))
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data, model, loss_function, optimizer, device, df_training, epoch=t)
        test(testing_data, model, loss_function, device, df_testing, epoch=t)
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))

    # Inference
    y_hats, tgt_ys = validation(validation_data, model)

    nse = utils.nash_sutcliffe_efficiency(tgt_ys, y_hats)

    return nse

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
    model_selection = 'autoformer' # 'autoformer', 'informer', 'fedformer'

    encoder_sequence_len = 365 # length of input given to encoder
    decoder_sequence_len = 1 # length of input given to decoder
    output_sequence_len = 2 # target sequence length (the informer does not work with 1 step ahead)
    encoder_input_size = 1
    decoder_input_size = 1
    decoder_output_size = 1 
    # encoder_features_fc_layer = 32
    # decoder_features_fc_layer = 32
    # n_encoder_layers = 2
    # n_decoder_layers = 1
    activation = 'gelu'
    embed = 'time_frequency'
    # d_model = 16
    # n_heads = 2
    attention_factor = 1
    frequency = 'd'
    dropout = 0.05
    distill = True # For the Informer: whether to use distilling in encoder, using this argument means not using distilling
    output_attention = True # Keep True for now
    moving_average = 25
    window_size = encoder_sequence_len + output_sequence_len
    step_size = 1

    # FEDformer specific hyperparams
    L = 1
    ab = 0
    modes = 32
    mode_select = 'random'
    version = 'Wavelets' # 'Wavelets' or 'Fourier'
    base = 'legendre'
    cross_activation = 'tanh'
    wavelet = 0

    num_workers = 8
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

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chuncks
    training_indices = utils.get_indices(data=training_data, window_size=window_size, step_size=step_size)
    testing_indices = utils.get_indices(data=testing_data, window_size=window_size, step_size=step_size)
    validation_indices = utils.get_indices(data=validation_data, window_size=window_size, step_size=step_size)

    # Scale the data
    scaler = StandardScaler()

    # Fit scaler on the training set
    scaler.fit(training_data.values)

    training_data = scaler.transform(training_data.values)
    testing_data = scaler.transform(testing_data.values)
    validation_data = scaler.transform(validation_data.values)

    # Extract positional encoding data
    training_data_pe = utils.positional_encoder(training_data_pe, time_encoding='time_frequency', frequency='d')
    testing_data_pe = utils.positional_encoder(testing_data_pe, time_encoding='time_frequency', frequency='d')
    validation_data_pe = utils.positional_encoder(validation_data_pe, time_encoding='time_frequency', frequency='d')

    # Make instance of the custom dataset class
    training_data = ds.AutoTransformerDataset(data=torch.tensor(training_data), data_pe=torch.tensor(training_data_pe),
                                            indices=training_indices, encoder_sequence_len=encoder_sequence_len, 
                                            decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    testing_data = ds.AutoTransformerDataset(data=torch.tensor(testing_data), data_pe=torch.tensor(testing_data_pe),
                                            indices=testing_indices, encoder_sequence_len=encoder_sequence_len,
                                            decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    validation_data = ds.AutoTransformerDataset(data=torch.tensor(validation_data), data_pe=torch.tensor(validation_data_pe),
                                            indices=validation_indices, encoder_sequence_len=encoder_sequence_len,
                                            decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)

    # Set up the dataloaders
    training_data = DataLoader(training_data, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    testing_data = DataLoader(testing_data, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    validation_data = DataLoader(validation_data, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)


    # Set up experiment hyperparams
    experiment_hyperparams = {
        'lr': [1e-5, 0.001],
        'd_model': [32, 64, 128, 256, 512],
        'n_heads': [2, 4, 8],
        'n_encoder_layers': [1, 2, 4],
        'n_decoder_layers': [1, 2, 4],
        'encoder_features_fc_layer': [32, 64, 128],
        'decoder_features_fc_layer': [32, 64, 128],
    }

    # Run optimization loop
    counter = 0
    optimization_df = pd.DataFrame(columns=('lr', 'd_model', 'n_heads', 'n_encoder_layers', 'n_decoder_layers', 'encoder_features_fc_layer', 'decoder_features_fc_layer', 'nse'))
    for lr in experiment_hyperparams['lr']:
        for d_model in experiment_hyperparams['d_model']:
            for n_heads in experiment_hyperparams['n_heads']:
                for n_encoder_layers in experiment_hyperparams['n_encoder_layers']:
                    for n_decoder_layers in experiment_hyperparams['n_decoder_layers']:
                        for encoder_features_fc_layer in experiment_hyperparams['encoder_features_fc_layer']:
                            for decoder_features_fc_layer in experiment_hyperparams['decoder_features_fc_layer']:
                                
                                print(f'Iteration {counter}')
                                nse = train_test_val(lr, d_model, n_heads, n_encoder_layers, n_decoder_layers, encoder_features_fc_layer, decoder_features_fc_layer)
                                
                                # Append parameters and nse to dataframe
                                optimization_df.loc[counter] = [lr, d_model, n_heads, n_encoder_layers, n_decoder_layers, 
                                                    encoder_features_fc_layer, decoder_features_fc_layer, nse]
                                
                                # Update counter
                                counter += 1
                                
    # Save dataframe to csv
    optimization_df.to_csv('results/optimization_results.csv', index=False)