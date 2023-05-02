import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# kernel_size = 3


class DoubleSepConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 ):
        super(DoubleSepConv, self).__init__()

        self.conv = nn.Sequential(
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
        )

    def forward(self, x):
        return self.conv(x)
