# Here would have to be basically the Exp_Main class of the Autoformer

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom, Dataset_Pred
from models import Informer, Autoformer, Transformer, Reformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

class Main():
    
    def __init__(self, is_training, model_id, model, data, root_path, data_path, features, target, freq, 
                checkpoints, seq_len, label_len, pred_len, bucket_size, n_hashes, enc_in, dec_in, c_out,
                d_model, n_heads, e_layers, d_layers, d_ff, moving_avg, factor, distil, dropout, embed,
                activation, output_attention, do_predict, num_workers, itr, train_epochs, batch_size,
                patience, learning_rate, des, loss, lradj, use_amp, use_gpu, gpu, use_multi_gpu, devices) -> None:
        
        # basic config
        self.is_training = is_training              # status
        self.model_id = model_id                    # model id
        self.model = model                          # model name, options: [Autoformer, Informer, Transformer]
        
        # data loader
        self.data = data                            # dataset to be used
        self.root_path = root_path                  # root path of the data file
        self.data_path = data_path                  # data file
        self.features = features                    # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        self.target = target                        # target feature in S or MS mask
        self.freq = freq                            # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
        self.checkpoints = checkpoints              # location of the model checkpoiints
        
        # forecasting task
        self.seq_len = seq_len                      # input sequence length
        self.label_len = label_len                  # start toke length
        self.pred_len = pred_len                    # prediction sequence length
        
        # model definition
        self.bucket_size = bucket_size              # for Reformer
        self.n_hashes = n_hashes                    # for Reformer
        self.enc_in = enc_in                        # encoder input size
        self.dec_in = dec_in                        # decoder input size
        self.c_out = c_out                          # output size
        self.d_model = d_model                      # dimension of the model
        self.n_heads = n_heads                      # number of heads
        self.e_layers = e_layers                    # number of encoder layers
        self.d_layers = d_layers                    # number of decoder layers
        self.d_ff = d_ff                            # dimension of fcn
        self.moving_avg = moving_avg                # window size of moving average
        self.factor = factor                        # attn factor
        self.distil = distil                        # whether to use distilling in encoder, using this argument means not using distilling
        self.dropout = dropout                      # dropout rate
        self.embed = embed                          # time features encoding, options:[timeF, fixed, learned]
        self.activation = activation                # activation
        self.output_attention = output_attention    # whether to output attention in encoder
        self.do_predict = do_predict                # whether to predict unseen future data
        
        # optimization
        self.num_workers = num_workers              # data loader num workers
        self.itr = itr                              # experiment times
        self.train_epochs = train_epochs            # train epochs
        self.batch_size = batch_size                # batch size of train input data
        self.patience = patience                    # early stopping patience
        self.learning_rate = learning_rate          # optimizer learning rate
        self.des = des                              # exp description
        self.loss = loss                            # loss function
        self.lradj = lradj                          # adjust learning rate
        self.use_amp = use_amp                      # use automatic mized precision training
        
        # GPU set up
        self.use_gpu = use_gpu                      # use gpu
        self.gpu = gpu                              # gpu
        self.use_multi_gpu = use_multi_gpu          # use multiple gpus
        self.devices = devices                      # devices
        
        # Experiment set up
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    
    def data_provider(self, flag):
        
        Data = Dataset_Custom
        timeenc = 0 if self.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = self.batch_size
            freq = self.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = self.freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = self.batch_size
            freq = self.freq

        data_set = Data(root_path=self.root_path, data_path=self.data_path, flag=flag, 
            size=[self.seq_len, self.label_len, self.pred_len], 
            features=self.features, target=self.target, timeenc=timeenc, freq=freq)
        
        print(flag, len(data_set))
        
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=self.num_workers, drop_last=drop_last)
        
        return data_set, data_loader

    def _acquire_device(self):
        
        if self.use_gpu:
        
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu) if not self.use_multi_gpu else self.devices
            device = torch.device('cuda:{}'.format(self.gpu))
            print('Use GPU: cuda:{}'.format(self.gpu))
        
        else:
        
            device = torch.device('cpu')
            print('Use CPU')
        
        return device

    def _build_model(self):
        
        model_dict = {'Autoformer': Autoformer, 'Transformer': Transformer, 'Informer': Informer, 'Reformer': Reformer,}
        
        model = model_dict[self.model].Model(self).float()
        
        self.use_gpu = True if torch.cuda.is_available() and self.use_gpu else False
        
        if self.use_gpu and self.use_multi_gpu:
            self.devices = self.devices.replace(' ', '')
            device_ids = self.devices.split(',')
            self.device_ids = [int(id_) for id_ in device_ids]
            self.gpu = self.device_ids[0]
        
        if self.use_multi_gpu and self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.device_ids)
        
        return model
    
    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
        
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.output_attention:
                outputs = outputs[0]
            return outputs

        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
    
    def vali(self, vali_data, vali_loader, criterion):
        
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                
                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        
        self.model.train()
        
        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self.data_provider(flag='train')
        vali_data, vali_loader = self.data_provider(flag='val')
        test_data, test_loader = self.data_provider(flag='test')

        path = os.path.join(self.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self, setting, test=0):
        test_data, test_loader = self.data_provider(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self.data_provider(flag='pred')

        if load:
            path = os.path.join(self.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    
if __name__ == '__main__':
    
    # Fix seeds
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # Create an instance of the Main class
    # Add default possitional arguments from here: https://github.com/thuml/Autoformer/blob/main/run.py
    experiment = Main(is_training=1, model_id='test', model='Autoformer', data='electricity', root_path='./data/electricity/', data_path='electricity.csv', 
                    features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=96, bucket_size=4, 
                    n_hashes=4, enc_in=7, dec_in=7, c_out=7, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=25, factor=1, 
                    distil=True, dropout=0.05, embed='timeF', activation='gelu', output_attention='store_true', do_predict='store_true', 
                    num_workers=8, itr=1, train_epochs=10, batch_size=32, patience=3, learning_rate=0.0001, des='test', loss='mse', lradj='type1', 
                    use_amp='store_true', use_gpu=False, gpu=0, use_multi_gpu=False, devices='0, 1, 2, 3')
    
    data_set, data_loader = experiment.data_provider(flag='train')
    
    # if experiment.is_training:
    #     for ii in range(experiment.itr):
    #         # setting record of experiments
    #         setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
    #             experiment.model_id,
    #             experiment.model,
    #             experiment.data,
    #             experiment.features,
    #             experiment.seq_len,
    #             experiment.label_len,
    #             experiment.pred_len,
    #             experiment.d_model,
    #             experiment.n_heads,
    #             experiment.e_layers,
    #             experiment.d_layers,
    #             experiment.d_ff,
    #             experiment.factor,
    #             experiment.embed,
    #             experiment.distil,
    #             experiment.des, ii)
            
            # experiment.train(setting=setting)
            
            # experiment.test(setting=setting)
            
            # if experiment.do_predict:
            #     experiment.predict(setting, True)