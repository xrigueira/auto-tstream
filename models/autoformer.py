import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embed import DataEmbedding, DataEmbedding_wo_pos
from layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.autoformer_encdec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, encoder_sequence_len: int, decoder_sequence_len: int, output_sequence_len: int,
                encoder_input_size: int, decoder_input_size: int, decoder_output_size: int, encoder_features_fc_layer: int,
                decoder_features_fc_layer: int, n_encoder_layers: int, n_decoder_layers: int, activation: str, embed: str, 
                d_model: int, n_heads: int, attention_factor: int, frequency: str, dropout: int, output_attention: bool, 
                moving_average: int):
        super(Autoformer, self).__init__()
        
        # Hyperparameters
        self.encoder_sequence_len = encoder_sequence_len
        self.decoder_sequence_len = decoder_sequence_len
        self.output_sequence_len = output_sequence_len
        self.encoder_input_size = encoder_input_size
        self.decoder_input_size = decoder_input_size
        self.decoder_output_size = decoder_output_size
        self.encoder_features_fc_layer = encoder_features_fc_layer
        self.decoder_features_fd_layer = decoder_features_fc_layer
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.activation = activation
        self.embed = embed
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_factor = attention_factor
        self.frequency = frequency
        self.dropout = dropout
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_average
        self.decomp = series_decomp(kernel_size)
        
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(encoder_input_size, d_model, embed, frequency, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(decoder_input_size, d_model, embed, frequency, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, attention_factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    encoder_features_fc_layer,
                    moving_avg=moving_average,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_encoder_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, attention_factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, attention_factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    decoder_output_size,
                    decoder_features_fc_layer,
                    moving_avg=moving_average,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(n_decoder_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, decoder_output_size, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
