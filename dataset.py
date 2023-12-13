import os
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset

class AutoTransformerDataset(Dataset):
    
    def __init__(self, data: torch.tensor, data_pe: torch.Tensor, encoder_sequence_len: int, decoder_sequence_len: int, tgt_sequence_len: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        super().__init__()

        self.data = data
        self.data_pe = data_pe
        
        self.encoder_sequence_len = encoder_sequence_len
        self.decoder_sequence_len = decoder_sequence_len
        self.tgt_sequence_len = tgt_sequence_len
        
    def __getitem__(self, index):
        
        # Get indices
        src_start_idx = index
        src_end_idx = src_start_idx + self.encoder_sequence_len
        
        tgt_start_idx = src_end_idx - self.decoder_sequence_len
        tgt_end_idx = tgt_start_idx + self.decoder_sequence_len + self.tgt_sequence_len
        print(src_start_idx, src_end_idx, tgt_start_idx, tgt_end_idx)
        
        # Subset data
        src = self.data[src_start_idx:src_end_idx, :-1]
        tgt = self.data[tgt_start_idx:tgt_end_idx, -1:]
        
        src_pe = self.data_pe[src_start_idx:src_end_idx, :]
        tgt_pe = self.data_pe[tgt_start_idx:tgt_end_idx, :]
        
        return src, tgt, src_pe, tgt_pe
        
    def __len__(self):
        return len(self.data) - self.encoder_sequence_len - self.decoder_sequence_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    # https://github.com/xrigueira/tstream/blob/main/dataset.py
    # The positional encoder would have to be a layer of the autoformer and not part of the dataloader to keep the
    # same structure as the tstream transformer.
    # I am going to replace the data_provider by calling the dataset three times. The data also has to be
    # divided into train, test and validation outside the dataset.
    
    # Take is step by step. If I feed the data in the same way the rest doing the test and train loops.
    # The original Autoformer has the Dataset_Custom dataset that does basically everything: get indices,
    # positional encoding, ect. I want to do this in different modules. Follow the structure of tstream
    # Careful with the indices. See how the indices are defined in the Autoformer and compare with tstream.
    
    # They key is putting Dataset_Custom(Dataset) in the same way as tstream. WRITE, REASON AND PLAN before 
    # coding
    