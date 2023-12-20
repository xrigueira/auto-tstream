import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.embed import DataEmbedding, DataEmbedding_wo_pos
from layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.fouriercorrelation import FourierBlock, FourierCrossAttention
from layers.multiwaveletcorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.fedformer_encdec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, encoder_sequence_len: int, decoder_sequence_len: int, output_sequence_len: int,
                encoder_input_size: int, decoder_input_size: int, decoder_output_size: int, encoder_features_fc_layer: int,
                decoder_features_fc_layer: int, n_encoder_layers: int, n_decoder_layers: int, activation: str, embed: str,
                d_model: int, n_heads: int, frequency: str, dropout: int, output_attention: bool, moving_average: int, 
                version: str, mode_select: str, modes: int, L: int, base: str, cross_activation: str, wavelet: int):
        super(FEDformer, self).__init__()
        
        # Hyperparameters
        self.encoder_sequence_len = encoder_sequence_len
        self.decoder_sequence_len = decoder_sequence_len
        self.output_sequence_len = output_sequence_len
        self.decoder_input_size = decoder_input_size
        self.decoder_output_size = decoder_output_size
        self.encoder_features_fc_layer = encoder_features_fc_layer
        self.decoder_features_fc_layer = decoder_features_fc_layer
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_average
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(encoder_input_size, d_model, embed, frequency, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(decoder_input_size, d_model, embed, frequency, dropout)

        if version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
            decoder_cross_att = MultiWaveletCross(in_channels=d_model,
                                                out_channels=d_model,
                                                  seq_len_q=self.encoder_sequence_len // 2 + self.output_sequence_len,
                                                seq_len_kv=self.encoder_sequence_len,
                                                modes=modes,
                                                ich=d_model,
                                                base=base,
                                                activation=cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.encoder_sequence_len,
                                            modes=modes,
                                            mode_select_method=mode_select)
            decoder_self_att = FourierBlock(in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.encoder_sequence_len // 2 + self.output_sequence_len,
                                            modes=modes,
                                            mode_select_method=mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=d_model,
                                                    out_channels=d_model,
                                                    seq_len_q=self.encoder_sequence_len // 2 + self.output_sequence_len,
                                                    seq_len_kv=self.seq_len,
                                                    modes=modes,
                                                    mode_select_method=mode_select)
        # Encoder
        enc_modes = int(min(modes, encoder_sequence_len // 2))
        dec_modes = int(min(modes, (encoder_sequence_len // 2 + output_sequence_len) // 2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
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
                        decoder_self_att,
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
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

    def forward(self, x_enc, x_dec, x_mark_enc, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.output_sequence_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.output_sequence_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.decoder_sequence_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.decoder_sequence_len:, :], (0, 0, 0, self.output_sequence_len))
        
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                trend=trend_init)
        
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.output_sequence_len:, :], attns
        else:
            return dec_out[:, -self.output_sequence_len:, :]  # [B, L, D]
