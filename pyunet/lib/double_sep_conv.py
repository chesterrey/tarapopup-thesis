import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# kernel_size = 3

<<<<<<< HEAD
from .depthwise_seperable_conv import DepthwiseSeperableConv

=======
>>>>>>> 62ad25c5c04475fddcfa06d0cb7a1ac6b1e2298a

class DoubleSepConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 ):
        super(DoubleSepConv, self).__init__()

        self.conv = nn.Sequential(
<<<<<<< HEAD
            DepthwiseSeperableConv(
                in_channels=in_channels, out_channels=out_channels),

            DepthwiseSeperableConv(
                in_channels=out_channels,
                out_channels=out_channels
            )
=======
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      padding=padding, groups=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      padding=padding, groups=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
>>>>>>> 62ad25c5c04475fddcfa06d0cb7a1ac6b1e2298a
        )

    def forward(self, x):
        return self.conv(x)
