import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import depthwise_seperable_conv as dsc
from double_conv import DoubleConv

class DoubleDepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2):
        super(DoubleDepthwiseSepConv, self).__init__()
        self.conv = nn.Sequential(
            dsc.DepthwiseSeperableConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            dsc.DepthwiseSeperableConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        )

    def forward(self, x):
        return self.conv(x)
