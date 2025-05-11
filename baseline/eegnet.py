# -*- coding: utf-8 -*-

"""
EEGNet.
Modified from https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot uniform/xavier initialization, and setting biases to zero. 
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("Norm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

# @SkorchNet
class EEGNet(nn.Module):
    """
    Modified from https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

    near exactly the same one.
    
    Assuming the input is a 1-second EEG signal sampled at 128Hz.
    EEGNet Settings:
    Parameter     vlawhern 
    kernel_time   64 
    n_filter      8
    D             2
    
    Add max norm constraint on convolutional layers and classification layer.

    Remove softmax layer with cross entropy loss in pytorch.
    """
    def __init__(self, n_channels, n_samples, n_classes,
            time_kernel=(8, (1, 64), (1, 1)),
            D=2,
            pool_kernel1=((1, 4), (1, 4)),
            separa_kernel=(16, (1, 16), (1, 1)),
            pool_kernel2=((1, 8), (1, 8)),
            dropout_rate=0.5, fc_norm_rate=0.25, depthwise_norm_rate=1,
            bn_affine=True):
        super().__init__()
        
        # time convolution
        self.step1 = nn.Sequential(OrderedDict([
            ('same_padding',
            nn.ConstantPad2d(compute_same_pad2d((n_channels, n_samples), time_kernel[1], stride=time_kernel[2]), 0)),
            ('time_conv', 
            nn.Conv2d(1, time_kernel[0], time_kernel[1],
                stride=time_kernel[2], padding=0, bias=False)),
            ('bn', 
            nn.BatchNorm2d(time_kernel[0], affine=bn_affine)),
            ('drop', nn.Dropout(dropout_rate))
        ]))

        # depthwise convolution
        self.step2 = nn.Sequential(OrderedDict([
            ('depthwise_conv', 
            MaxNormConstraintConv2d(time_kernel[0], time_kernel[0]*D, (n_channels, 1),
                groups=time_kernel[0], bias=False, max_norm_value=depthwise_norm_rate)),
            ('bn', 
            nn.BatchNorm2d(time_kernel[0]*D, affine=bn_affine)),
            ('elu', nn.ELU()),
            ('ave_pool', nn.AvgPool2d(pool_kernel1[0], stride=pool_kernel1[1])),
            ('drop', nn.Dropout(dropout_rate))
        ]))

        with torch.no_grad():
            fake_input = torch.zeros((1, 1, n_channels, n_samples))
            fake_output = self.step2(self.step1(fake_input))
            middle_size = fake_output.shape[2:]

        # separable convolution
        self.step3 = nn.Sequential(OrderedDict([
            ('same_padding',
            nn.ConstantPad2d(compute_same_pad2d(middle_size, separa_kernel[1], stride=separa_kernel[2]), 0)),
            ('separable_conv', 
            SeparableConv2d(time_kernel[0]*D, separa_kernel[0], separa_kernel[1],
                stride=separa_kernel[2], padding=0, bias=False)),
            ('bn', nn.BatchNorm2d(separa_kernel[0], affine=bn_affine)),
            ('elu', nn.ELU()),
            ('ave_pool', nn.AvgPool2d(pool_kernel2[0], pool_kernel2[1])),
            ('drop', nn.Dropout(dropout_rate)),
            ('flatten', nn.Flatten())
        ]))

        with torch.no_grad():
            fake_output = self.step3(fake_output)
            self.middle_size = fake_output.shape[1]

        self.fc_layer = MaxNormConstraintLinear(self.middle_size, n_classes, max_norm_value=fc_norm_rate)

        self.model = nn.Sequential(self.step1, self.step2, self.step3, self.fc_layer)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        _glorot_weight_zero_bias(self)

    def forward(self, X):
        X = X.unsqueeze(1) # 4D
        out = self.model(X)
        return out





class MaxNormConstraintConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm_value=1, norm_axis=2, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True))
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= (desired/norms)
        return w


class MaxNormConstraintLinear(nn.Linear):
    def __init__(self, *args, max_norm_value=1, norm_axis=0, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)
    
    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True))
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= (desired/norms)
        return w  

class SeparableConv2d(nn.Module):
    """An equally SeparableConv2d in Keras.
    A depthwise conv followed by a pointwise conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros', D=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels*D, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False, padding_mode=padding_mode)
        self.pointwise_conv = nn.Conv2d(in_channels*D, out_channels, 1, stride=1, padding=0, bias=bias)
        self.model = nn.Sequential(self.depthwise_conv, self.pointwise_conv)

    def forward(self, X):
        return self.model(X)
    
class MaxNormConstraint:

    def __init__(self,max_value=1,axis=None):
        self.max_value=max_value
        self.axis=axis

    def __call__(self, module):
        if hasattr(module,'weight'):
            weights=module.weight.data
            norms=torch.sqrt(torch.sum(torch.pow(weights,2),dim=self.axis,keepdim=True))
            desired=torch.clamp(norms,min=0,max=self.max_value)
            weights*=desired/(np.finfo(np.float).eps+norms)
            module.weight.data=weights
            return module

def compute_conv_outsize(input_size, kernel_size, stride=1, padding=0, dilation=1):
    return int((input_size - dilation * kernel_size + 2 * padding) / stride + 1)
def compute_out_size(input_size: int, kernel_size: int,
        stride: int = 1, padding: int = 0, dilation: int = 1):
    return int((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

def compute_same_pad1d(input_size, kernel_size, stride=1, dilation=1):
    all_padding = ((stride-1)*input_size - stride + dilation*(kernel_size-1) + 1)
    return (all_padding//2, all_padding-all_padding//2)

def compute_same_pad2d(input_size, kernel_size, stride=(1, 1), dilation=(1, 1)):
    ud = compute_same_pad1d(input_size[0], kernel_size[0], stride=stride[0], dilation=dilation[0])
    lr = compute_same_pad1d(input_size[1], kernel_size[1], stride=stride[1], dilation=dilation[1])
    return [*lr, *ud]
