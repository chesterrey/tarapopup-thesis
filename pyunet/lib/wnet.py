import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os

from .double_conv import DoubleConv
from .double_sep_conv import DoubleSepConv


class WNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, k=3
    ):
        super(WNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ups = nn.ModuleList()
        self.left_downs = nn.ModuleList()
        self.right_downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_downs.append(DoubleConv(in_channels, 64))
        self.left_downs.append(DoubleSepConv(64, 128))
        self.left_downs.append(DoubleSepConv(128, 256))
        self.left_downs.append(DoubleSepConv(256, 512))

        self.bottleneck = DoubleSepConv(512, 512 * 2)

        self.ups.append(
            nn.ConvTranspose2d(
                512 * 2,
                512,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleSepConv(512 * 2, 512))

        self.ups.append(
            nn.ConvTranspose2d(
                256 * 2,
                256,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleSepConv(256 * 2, 256))

        self.ups.append(
            nn.ConvTranspose2d(
                128 * 2,
                128,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleSepConv(128 * 2, 128))

        self.ups.append(
            nn.ConvTranspose2d(
                64 * 2,
                64,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleConv(64 * 2, 64))

        self.final_conv_left = nn.Conv2d(64, k, kernel_size=1)
        self.softmax = nn.Softmax2d()

        self.right_downs.append(DoubleConv(k, 64))
        self.right_downs.append(DoubleConv(64, 128))
        self.right_downs.append(DoubleConv(128, 256))
        self.right_downs.append(DoubleConv(256, 512))

        self.final_conv_right = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):

        # Leftside of W-Net
        skip_connections = []

        for down in self.left_downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse skip_connections
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # print(x.shape[-1])
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv_left(x)
        x = self.softmax(x)

        # Rightside of W-Net
        skip_connections = []

        for down in self.right_downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse skip_connections
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # print(x.shape[-1])
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv_right(x)
