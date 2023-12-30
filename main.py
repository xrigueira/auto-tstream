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
from models.autoformer import Autoformer
from models.informer import Informer
from models.fedformer import FEDformer

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
        
        # if i % 20 == 0:
        #     print('Current batch', i)
        #     loss, current = loss.item(), (i + 1) * len(src)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define val step
def val(dataloader, model, loss_function, device, df_validation, epoch):
    
    num_batches = len(dataloader)
    model.eval()
    validation_loss = [] # For plotting purposes
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
            validation_loss.append(loss.item())
            epoch_val_loss = np.mean(validation_loss)
            df_validation.loc[epoch] = [epoch, epoch_val_loss]
    
    loss /= num_batches
    # print(f"Avg test loss: {loss:>8f}")

# Define test step
def test(dataloader, model):
    
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
    print('Test shape:', y_hats.shape, tgt_ys.shape)
    y_hats = y_hats.reshape(-1, y_hats.shape[-2], y_hats.shape[-1])
    tgt_ys = tgt_ys.reshape(-1, tgt_ys.shape[-2], tgt_ys.shape[-1])
    print('Test shape:', y_hats.shape, tgt_ys.shape)
    
    # Get metrics
    mae, mse, rmse, mape, mspe = utils.metric(y_hats, tgt_ys)
    print('MSE: {}\nMAE: {}'.format(mse, mae))
    
    return y_hats.squeeze(), tgt_ys.squeeze()

if __name__ == '__main__':
    
    # Define seed
    torch.manual_seed(0)
    
    # Hyperparams
    batch_size = 128
    validation_size = 0.125
    src_variables = ['X']
    tgt_variables = ['y']
    input_variables = src_variables + tgt_variables
    timestamp_col_name = "time"
    model_selection = 'autoformer' # 'autoformer', 'informer', 'fedformer'
    
    d_model = 16
    n_heads = 2
    n_encoder_layers = 2
    n_decoder_layers = 1
    encoder_sequence_len = 96 # length of input given to encoder
    decoder_sequence_len = 1 # length of input given to decoder
    output_sequence_len = 2 # target sequence length (the informer does not work with 1 step ahead)
    encoder_input_size = 1
    decoder_input_size = 1
    decoder_output_size = 1 
    encoder_features_fc_layer = 32
    decoder_features_fc_layer = 32
    activation = 'gelu'
    embed = 'time_frequency'
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
    
    # Extract train and test data
    training_val_lower_bound = datetime.datetime(1980, 10, 1)
    training_val_upper_bound = datetime.datetime(2010, 9, 30)

    # Extract train/validation and test data
    training_val_data = data[(training_val_lower_bound <= data.time) & (data.time <= training_val_upper_bound)]
    testing_data = data[data.time > training_val_upper_bound]

    # Extract train/calibrate and test temporal data for positional encoding
    training_val_data_pe = training_val_data.iloc[:, :1]
    testing_data_pe = testing_data.iloc[:, :1]
    
    # Adapt to prediction task
    if features == 'M' or features == 'MS':
        cols_data = data.columns[1:]
        training_val_data = training_val_data[cols_data]
        testing_data = testing_data[cols_data]
    elif features == 'S':
        training_val_data = training_val_data[[src_variables[0]]]
        testing_data = testing_data[[src_variables[0]]]

    # Scale the data
    scaler = StandardScaler()
    
    # Fit scaler on the training set
    scaler.fit(training_val_data.values)
    
    training_val_data = scaler.transform(training_val_data.values)
    testing_data = scaler.transform(testing_data.values)
    
    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chuncks
    training_val_indices = utils.get_indices(data=training_val_data, window_size=window_size, step_size=step_size)
    
    training_indices = training_val_indices[:-(round(len(training_val_indices)*validation_size))]
    validation_indices = training_val_indices[(round(len(training_val_indices)*(1-validation_size))):]
    
    testing_indices = utils.get_indices(data=testing_data, window_size=window_size, step_size=step_size)
    
    # Extract positional encoding data
    training_val_data_pe = utils.positional_encoder(training_val_data_pe, time_encoding='time_frequency', frequency='d')
    testing_data_pe = utils.positional_encoder(testing_data_pe, time_encoding='time_frequency', frequency='d')
    
    # Make instance of the custom dataset class
    training_data = ds.AutoTransformerDataset(data=torch.tensor(training_val_data), data_pe=torch.tensor(training_val_data_pe),
                                            indices=training_indices, encoder_sequence_len=encoder_sequence_len, 
                                            decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    validation_data = ds.AutoTransformerDataset(data=torch.tensor(training_val_data), data_pe=torch.tensor(training_val_data_pe),
                                            indices=validation_indices, encoder_sequence_len=encoder_sequence_len,
                                            decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    testing_data = ds.AutoTransformerDataset(data=torch.tensor(testing_data), data_pe=torch.tensor(testing_data_pe),
                                            indices=testing_indices, encoder_sequence_len=encoder_sequence_len,
                                            decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    
    # Set up the dataloaders
    training_val_data = training_data + validation_data # For testing puporses
    training_data = DataLoader(training_data, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    validation_data = DataLoader(validation_data, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    testing_data = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)
    training_val_data = DataLoader(training_val_data, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False) # For testing puporses

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
    
    # Print model and number of parameters
    print('Defined model:\n', model)
    utils.count_parameters(model)
    
    # Define optimizer and loss function
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Update model in the training process and test it
    epochs = 5 # 250
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_validation = pd.DataFrame(columns=('epoch', 'loss_test'))
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data, model, loss_function, optimizer, device, df_training, epoch=t)
        val(validation_data, model, loss_function, device, df_validation, epoch=t)
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))
    
    # # Save the model
    # torch.save(model, "models/model.pth")
    # print("Saved PyTorch entire model to models/model.pth")

    # # Load the model
    # model = torch.load("models/model.pth").to(device)
    # print('Loaded PyTorch model from models/model.pth')
    
    # Inference
    y_hats_train_val, tgt_ys_train_val = test(training_val_data, model)
    y_hats_test, tgt_ys_test = test(testing_data, model)
    
    # Plot loss
    plt.figure(1);plt.clf()
    plt.plot(df_training['epoch'], df_training['loss_train'], '-o', label='loss train')
    plt.plot(df_training['epoch'], df_validation['loss_test'], '-o', label='loss test')
    plt.yscale('log')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend()
    plt.show()
    
    # Plot testing results
    plt.figure(2);plt.clf()
    plt.plot(tgt_ys_train_val, label='observed')
    plt.plot(y_hats_train_val, label='predicted')
    plt.title('Training and validation results')
    plt.xlabel(r'time (days)')
    plt.ylabel(r'y')
    plt.legend()
    plt.show()

    plt.figure(2);plt.clf()
    plt.plot(tgt_ys_test, label='observed')
    plt.plot(y_hats_test, label='predicted')
    plt.title('Testing results')
    plt.xlabel(r'time (days)')
    plt.ylabel(r'y')
    plt.legend()
    plt.show()

    # Metrics
    from sklearn.metrics import mean_squared_error

    nse_train_val = utils.nash_sutcliffe_efficiency(tgt_ys_train_val, y_hats_train_val)
    rmse_train_val = np.sqrt(mean_squared_error(tgt_ys_train_val, y_hats_train_val))
    pbias_train_val = utils.pbias(tgt_ys_train_val, y_hats_train_val)
    kge_train_val = utils.kge(tgt_ys_train_val, y_hats_train_val)
    print('\n-- Train/val results')
    print('NSE = ', nse_train_val)
    print('RMSE = ', rmse_train_val)
    print('PBIAS = ', pbias_train_val)
    print('KGE = ', kge_train_val)
    
    nse_test = utils.nash_sutcliffe_efficiency(tgt_ys_test, y_hats_test)
    rmse_test = np.sqrt(mean_squared_error(tgt_ys_test, y_hats_test))
    pbias_test = utils.pbias(tgt_ys_test, y_hats_test)
    kge_test = utils.kge(tgt_ys_test, y_hats_test)
    print('\n-- Testing results')
    print('NSE = ', nse_test)
    print('RMSE = ', rmse_test)
    print('PBIAS = ', pbias_test)
    print('KGE = ', kge_test)