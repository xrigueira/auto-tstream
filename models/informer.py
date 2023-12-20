import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embed import DataEmbedding
from layers.transformer_encdec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.selfattention_family import FullAttention, ProbAttention, AttentionLayer
from utils import TriangularCausalMask, ProbMask


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, output_sequence_len: int, encoder_input_size: int, decoder_input_size: int, decoder_output_size: int, 
                encoder_features_fc_layer: int,decoder_features_fc_layer: int, n_encoder_layers: int, n_decoder_layers: int, 
                activation: str, embed: str, d_model: int, n_heads: int, attention_factor: int, frequency: str, dropout: int, 
                distill: bool, output_attention: bool):
        super(Informer, self).__init__()

        # Hyperparameters
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
        self.distill = distill
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(encoder_input_size, d_model, embed, frequency, dropout)
        self.dec_embedding = DataEmbedding(decoder_input_size, d_model, embed, frequency, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, attention_factor, attention_dropout=dropout,
                                    output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    encoder_features_fc_layer,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_encoder_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(n_encoder_layers - 1)
            ] if distill else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, attention_factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, attention_factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    decoder_features_fc_layer,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(n_decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, decoder_output_size, bias=True)
        )

    def forward(self, x_enc, x_dec, x_mark_enc, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.output_sequence_len:, :], attns
        else:
            return dec_out[:, -self.output_sequence_len:, :]  # [B, L, D]
