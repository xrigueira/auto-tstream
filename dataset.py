import os
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset

class AutoTransformerDataset(Dataset):
    
    def __init__(self, data: torch.tensor, indices: list, features: str, target: str, scale: bool, time_encoding: int, frequency: str,
                encoder_sequence_len: int, decoder_sequence_len: int, tgt_sequence_len: int) -> None:
        super().__init__()

        self.data = data
        self.indices = indices

        self.target = target
        self.time_encoding = time_encoding
        self.frequency = frequency
        
        self.encoder_sequence_len = encoder_sequence_len
        self.decoder_sequence_len = decoder_sequence_len
        self.tgt_sequence_len = tgt_sequence_len
        
    def __len__(self):
        return len(self.indices)
        

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
    