import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# kernel_size = 3

from .depthwise_seperable_conv import DepthwiseSeperableConv


class DoubleSepConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 ):
        super(DoubleSepConv, self).__init__()

        self.conv = nn.Sequential(
            DepthwiseSeperableConv(
                in_channels=in_channels, out_channels=out_channels),

            DepthwiseSeperableConv(
                in_channels=out_channels,
                out_channels=out_channels
            )
        )

    def forward(self, x):
        return self.conv(x)
