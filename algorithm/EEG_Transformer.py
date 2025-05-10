import math
import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
# from metabci.brainda.algorithms.deep_learning.base import SkorchNet


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x=x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.transpose(0,1))


def compute_conv_outsize(input_size, kernel_size, stride=1, padding=0, dilation=1):
    return int((input_size - dilation * kernel_size + 2 * padding) / stride + 1)


# EEG Feature Extraction Transformer  model
class EFET_block(nn.Module):

    def __init__(self, Chans=64, n_spatial_filters=64*2, n_freq_filters=8, srate=256, tf_dim=1024, n_layers=1 ,dropout=0.5):
        super(EFET_block, self).__init__()
        self.n_freq_kernels = n_freq_filters
        self.dropout = nn.Dropout(dropout)

        # 1.spatial filter
        self.sp_filter_nn = nn.Linear(Chans, n_spatial_filters)

        # 2.frequency filter conv2d
        # input_size (batch_size,in_channels,n_spatial_filter,samples)
        # kernel:(in_channels,n_freq_filter,kernel_size=(1,kernLength)) padding
        # output_size(batch_size,n_freq_filter,n_spatial_filter,samples)

        # kernLength = sample_rate / high_pass_frequency
        freq_kernel = srate // 2 if (srate // 2) % 2 == 1 else (srate // 2+1)

        self.freq_filter_conv = nn.Conv2d(1, n_freq_filters, (1, freq_kernel), stride=(1, 1),
                                          padding=(0, freq_kernel // 2), bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(n_freq_filters)

        # frequency * spatial filter transformer
        # self.filter_transformer = TransformerEncoderLayer(d_model=n_spatial_filter * n_freq_filter, nhead=n_freq_filter,
        #                                                   dim_feedforward=tf_dim, dropout=dropout,
        #                                                   batch_first=True)

        self.filter_transformer_layer = TransformerEncoderLayer(d_model=n_spatial_filters * n_freq_filters, nhead=n_freq_filters,
                                                          dim_feedforward=tf_dim, dropout=dropout,
                                                          batch_first=True)

        self.filter_transformer = TransformerEncoder(self.filter_transformer_layer,num_layers=n_layers)

        self.pos_encoder = PositionalEncoding(d_model=n_spatial_filters * n_freq_filters, dropout=dropout)

    def forward(self, x):
        n_batch, n_channel, n_sample = x.shape

        # x : [batch ,1 , eeg_chan , samples]
        # x = x.view(-1, 1, *x.size()[1:])

        x = x.transpose(1, 2)

        # spatial_x : [batch, 1 ,n_spatial_filter, samples]
        spatial_x = self.sp_filter_nn(x)
        spatial_x = F.tanh(spatial_x)
        spatial_x = self.dropout(spatial_x)
        spatial_x = spatial_x.transpose(1, 2)

        spatial_x = spatial_x.view(-1, 1, *spatial_x.size()[1:])

        # filter_bank_x : [batch, n_freq_filter ,n_spatial_filter, samples]
        filter_bank_x = self.freq_filter_conv(spatial_x)
        filter_bank_x = F.elu(filter_bank_x)
        filter_bank_x = self.dropout(filter_bank_x)
        filter_bank_x = self.batch_norm_1(filter_bank_x)

        # filter_bank_x : [batch, samples, n_filter*n_spatial_filter]
        filter_bank_x = filter_bank_x.view(
            filter_bank_x.shape[0], -1, filter_bank_x.shape[3])
        filter_bank_x = torch.transpose(filter_bank_x, 1, 2)


        p_encoder_x = self.pos_encoder(filter_bank_x)

        # filter_tfm_out : [batch, samples, n_filter*n_spatial_filter]
        filter_tfm_out = self.filter_transformer(p_encoder_x)
        filter_tfm_out = nn.functional.leaky_relu(filter_tfm_out)

        # out : [batch, n_filter*n_spatial_filter, samples]
        out = torch.transpose(filter_tfm_out, 1, 2)
        # out = out.view(out.shape[0],-1,n_channel,out.shape[2])

        return out

class AETF(nn.Module):

    def __init__(self, nClasses: int, nSamples: int, Chans=64, n_freq_filters=8,n_spatial_filters=None, srate=256,
                 dim_feedforward=1024, n_layers=1, dropout1=0.5,dropout2=0.5):
        super().__init__()

        

        # self.EFETc = EFET_block(Chans, n_freq_kernels,
        #                         srate, dim_feedforward, dropout)

        if n_spatial_filters is None:
            n_spatial_filters=Chans*2

        self.EFETc = EFET_block(
            Chans=Chans,
            n_spatial_filters=n_spatial_filters,
            n_freq_filters=n_freq_filters,
            srate=srate, 
            tf_dim=dim_feedforward, 
            n_layers=n_layers,
            dropout=dropout1
        )


        self.dropout = nn.Dropout(dropout2)

        # self.conv = nn.Conv2d(
        #     1, conv_n, (n_freq_kernels * Chans, conv_size), (1, 1))

        # out_size = compute_conv_outsize(nSamples, conv_size, 1)

        self.fc = nn.Linear(n_freq_filters*n_spatial_filters * nSamples, nClasses)

    def forward(self, x):
        tfm_out = self.EFETc(x)
        # tfm_out = tfm_out.view(
        #     tfm_out.shape[0], 1, tfm_out.shape[1], tfm_out.shape[2])

        # conv_out = self.conv(tfm_out)
        # conv_out = self.dropout(conv_out)
        # conv_out = nn.functional.leaky_relu(conv_out)

        flatten_conv_out = tfm_out.flatten(start_dim=1)
        out = self.fc(flatten_conv_out)
        out = self.dropout(out)
        out = nn.functional.softmax(out, dim=1)

        return out




class AE_noTF(nn.Module):

    def __init__(self, nClasses: int, nSamples: int, Chans=64, n_freq_filters=8,n_spatial_filters=None, srate=256,
                 dim_feedforward=1024, n_layers=1, dropout1=0.5,dropout2=0.5):
        super().__init__()



        if n_spatial_filters is None:
            n_spatial_filters=Chans*2

        self.EFETc = EFET_block(
            Chans=Chans,
            n_spatial_filters=n_spatial_filters,
            n_freq_filters=n_freq_filters,
            srate=srate, 
            tf_dim=dim_feedforward, 
            n_layers=n_layers,
            dropout=dropout1
        )


        self.dropout = nn.Dropout(dropout2)

        # self.conv = nn.Conv2d(
        #     1, conv_n, (n_freq_kernels * Chans, conv_size), (1, 1))

        # out_size = compute_conv_outsize(nSamples, conv_size, 1)

        self.fc = nn.Linear(n_freq_filters*n_spatial_filters * nSamples, nClasses)

    def forward(self, x):
        tfm_out = self.EFETc(x)
        # tfm_out = tfm_out.view(
        #     tfm_out.shape[0], 1, tfm_out.shape[1], tfm_out.shape[2])

        # conv_out = self.conv(tfm_out)
        # conv_out = self.dropout(conv_out)
        # conv_out = nn.functional.leaky_relu(conv_out)

        flatten_conv_out = tfm_out.flatten(start_dim=1)
        out = self.fc(flatten_conv_out)
        out = self.dropout(out)
        out = nn.functional.softmax(out, dim=1)

        return out

