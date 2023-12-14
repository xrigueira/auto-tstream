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
from models.autoformer import Autoformer

def train(dataloader, model, loss_function, optimizer, patience, device, df_training, epoch):
    size = len(dataloader.dataset)
    model.train()
    training_loss = [] # For plotting purposes
    for i, batch in enumerate(dataloader):
        src, tgt, src_pe, tgt_pe = batch
        src, tgt, src_pe, tgt_pe = src.float().to(device), tgt.float().to(device), src_pe.float().to(device), tgt_pe.float().to(device)
        
        # Zero out gradients for every batch
        model.zero_grad()
        
        # Process decoder input
        decoder_input = torch.zeros_like(tgt[:, -output_sequence_length:, :]).float()
        decoder_input = torch.cat([tgt[:, :decoder_sequence_len, :], decoder_input], dim=1).float().to(device)
        
        # Compute prediction error
        pred, attention_weights = model(src, decoder_input, src_pe, tgt_pe)
        
        f_dim = -1 if features == 'MS' else 0
        pred = pred[:, -decoder_sequence_len:, f_dim:].to(device)
        tgt_y = tgt[:, -decoder_sequence_len:, f_dim:].to(device)
        
        loss = loss_function(pred, tgt_y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Save results for plotting
        training_loss.append(loss.item())
        epoch_train_loss = np.mean(training_loss)
        df_training.loc[epoch] = [epoch, epoch_train_loss]
        
        if i % 5 == 0:
            print('Current batch', i)
            loss, current = loss.item(), (i + 1) * len(src)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_function, patience, device, df_testing, epoch):
    num_batches = len(dataloader)
    model.eval()
    testing_loss = [] # For plotting purposes
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, src_pe, tgt_pe = batch
            src, tgt, src_pe, tgt_pe = src.float().to(device), tgt.float().to(device), src_pe.float().to(device), tgt_pe.float().to(device)
            
            # Process decoder input
            decoder_input = torch.zeros_like(tgt[:, -output_sequence_length:, :]).float()
            decoder_input = torch.cat([tgt[:, :decoder_sequence_len, :], decoder_input], dim=1).float().to(device)
            
            # Compute prediction error
            pred, attention_weights = model(src, decoder_input, src_pe, tgt_pe)
            
            f_dim = -1 if features == 'MS' else 0
            pred = pred[:, -decoder_sequence_len:, f_dim:].to(device)
            tgt_y = tgt[:, -decoder_sequence_len:, f_dim:].to(device)
        
            loss = loss_function(pred, tgt_y)
            
            # Save results for plotting
            testing_loss.append(loss.item())
            epoch_test_loss = np.mean(testing_loss)
            df_testing.loc[epoch] = [epoch, epoch_test_loss]
    
    loss /= num_batches
    # print(f"Avg test loss: {loss:>8f}")

def validation(dataloader, model):
    
    # Define lists to store the predictions and ground truth
    y_hats = []
    tgt_ys = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            rc, tgt, src_pe, tgt_pe = batch
            src, tgt, src_pe, tgt_pe = src.float().to(device), tgt.float().to(device), src_pe.float().to(device), tgt_pe.float().to(device)
            
            # Process decoder input
            decoder_input = torch.zeros_like(tgt[:, -output_sequence_length:, :]).float()
            decoder_input = torch.cat([tgt[:, :decoder_sequence_len, :], decoder_input], dim=1).float().to(device)
            
            # Compute prediction error
            pred, attention_weights = model(src, decoder_input, src_pe, tgt_pe)
            
            f_dim = -1 if features == 'MS' else 0
            pred = pred[:, -decoder_sequence_len:, f_dim:].to(device)
            tgt_y = tgt[:, -decoder_sequence_len:, f_dim:].to(device)
            
            y_hat = pred.detach().cpu().numpy()
            tgt_y = tgt_y.detach().cpu().numpy()
            
            y_hats.append(y_hat)
            tgt_ys.append(tgt_y)
    
    y_hats = np.concatenate(y_hats, axis=0)
    tgt_y = np.concatenate(tgt_ys, axis=0)
    print('Validation shape:', y_hats.shape, tgt_ys.shape)
    y_hats = y_hats.reshape(-1, y_hats.shape[-2], y_hats.shape[-1])
    tgt_y = tgt_y.reshape(-1, tgt_y.shape[-2], tgt_y.shape[-1])
    print('Validation shape:', y_hats.shape, tgt_ys.shape)
    
    # Get metrics
    mae, mse, rmse, mape, mspe = utils.metric(y_hats, tgt_ys)
    print('MAE: {}\nMSE: {}\nRMSE: {}\nMAPE: {}\nMSPE: '.format(mae, mse, rmse, mape, mspe))

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
    encoder_input_size = 1
    decoder_input_size = 1
    decoder_output_size = 1 
    encoder_features_fc_layer = 2048
    decoder_features_fc_layer = 2048
    n_encoder_layers = 2
    n_decoder_layers = 1
    activation = 'gelu'
    embed = 'time_frequency'
    d_model = 512
    n_heads = 8
    attention_factor = 1
    frequency = 'D'
    dropout = 0.05
    output_attention = True # Keep True for now
    moving_average = 25
    
    num_workers = 8
    features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate'
    patience = 3
    
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
    training_data = DataLoader(training_data, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    testing_data = DataLoader(testing_data, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    validation_data = DataLoader(validation_data, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    # Build model
    model = Autoformer(encoder_sequence_len=encoder_sequence_len, decoder_sequence_len=decoder_sequence_len, output_sequence_len=output_sequence_length,
                encoder_input_size=encoder_input_size, decoder_input_size=decoder_input_size, decoder_output_size=decoder_output_size, 
                encoder_features_fc_layer=encoder_features_fc_layer, decoder_features_fc_layer=decoder_features_fc_layer, n_encoder_layers=n_encoder_layers,
                n_decoder_layers=n_decoder_layers, activation=activation, embed=embed, d_model=d_model, n_heads=n_heads, attention_factor=attention_factor,
                frequency=frequency, dropout=dropout, output_attention=output_attention, moving_average=moving_average).float()
    
    # Send model to device
    model.to(device)
    
    # Print model and number of parameters
    print('Defined model:\n', model)
    utils.count_parameters(model)
    
    # Define optimizer and loss function
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Update model in the training process and test it
    epochs = 2 # 250
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_testing = pd.DataFrame(columns=('epoch', 'loss_test'))
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data, model, loss_function, optimizer, patience, device, df_training, epoch=t)
        test(testing_data, model, loss_function, device, df_testing, epoch=t)
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))
    
    # # Save the model
    # torch.save(model, "models/model.pth")
    # print("Saved PyTorch entire model to models/model.pth")

    # # Load the model
    # model = torch.load("models/model.pth").to(device)
    # print('Loaded PyTorch model from models/model.pth')
    
    # Inference
    validation(validation_data, model)
    
    # Plot loss
    plt.figure(1);plt.clf()
    plt.plot(df_training['epoch'], df_training['loss_train'], '-o', label='loss train')
    plt.plot(df_training['epoch'], df_testing['loss_test'], '-o', label='loss test')
    plt.yscale('log')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend()
    plt.show()