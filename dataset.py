import os
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset

class AutoTransformerDataset(Dataset):
    
    def __init__(self, data: torch.tensor, data_pe: torch.Tensor, indices: list, encoder_sequence_len: int, decoder_sequence_len: int, tgt_sequence_len: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        super().__init__()

        self.data = data
        self.data_pe = data_pe
        self.indices = indices
        
        self.encoder_sequence_len = encoder_sequence_len
        self.decoder_sequence_len = decoder_sequence_len
        self.tgt_sequence_len = tgt_sequence_len
        
    def __getitem__(self, index):
        
        """
        Returns a tuple with 4 elements:
        1) src (the encoder input)
        2) tgt (the decoder input)
        3) src_pe (the encoder positional encoding input)
        4) tgt_pe (the decoder position encoding input)
        """
        try:
            # Get the first element of the i'th tuple in the list self.indices
            start_idx = self.indices[index][0]

            # Get the second (and last) element of the i'th tuple in the list self.indices
            end_idx = self.indices[index][1]

            sequence = self.data[start_idx:end_idx]
            sequence_pe = self.data_pe[start_idx:end_idx]

            src, tgt, src_pe, tgt_pe = self._get_srcs_tgts(sequence=sequence, sequence_pe=sequence_pe, encoder_sequence_len=self.encoder_sequence_len,
                                                        decoder_sequence_len=self.decoder_sequence_len, tgt_sequence_len=self.tgt_sequence_len)

            return src, tgt, src_pe, tgt_pe
        
        except IndexError:
            # Handle the case where the index is out of range
            print(f"IndexError: Index {index} is out of range.")
            return None
        
    def __len__(self):
        return len(self.indices)
        # return len(self.data) - self.encoder_sequence_len - self.tgt_sequence_len + 1
    
    def _get_srcs_tgts(self, sequence: torch.Tensor, sequence_pe: torch.Tensor, encoder_sequence_len: int, decoder_sequence_len: int,
                    tgt_sequence_len: int) -> None:

        assert len(sequence) == encoder_sequence_len + tgt_sequence_len, "Sequence length does not equal (encoder_sequence_len + tgt_sequence_len)"
        
        src = sequence[:encoder_sequence_len, :-1]
        src_pe = sequence_pe[:encoder_sequence_len, :]
        
        assert len(src) == encoder_sequence_len, "Sequence length does not equal src length: encoder_sequence_length"
        
        tgt = sequence[encoder_sequence_len-decoder_sequence_len:, :-1]
        tgt_pe = sequence_pe[encoder_sequence_len-decoder_sequence_len:, :]
        
        assert len(tgt) == decoder_sequence_len + tgt_sequence_len, "Lenght of tgt does not match target sequence length: (decoder_sequence_len + tgt_sequence_len)"

        tgt_y = sequence[encoder_sequence_len-decoder_sequence_len+1:, -1:]

        assert len(tgt_y) == decoder_sequence_len + tgt_sequence_len - 1, "Lenght of tgt_y does not match target sequence length: (decoder_sequence_len + tgt_sequence_len - 1)"
        
        return src, tgt, src_pe, tgt_pe
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    