import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .double_conv import DoubleConv
from .double_sep_conv import DoubleSepConv
import torch.utils.checkpoint as cp

class UEnc(nn.Module):
    def __init__(
        self, in_channels, out_channels,
    ):
        super(UEnc, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downs.append(DoubleConv(in_channels, 64))
        self.downs.append(DoubleSepConv(64, 128))
        self.downs.append(DoubleSepConv(128, 256))
        self.downs.append(DoubleSepConv(256, 512))

        self.bottleneck = DoubleSepConv(512, 512 * 2)

        self.ups.append(
            nn.ConvTranspose2d(
                512 * 2,
                512,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleConv(512 * 2, 512))

        self.ups.append(
            nn.ConvTranspose2d(
                256 * 2,
                256,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleConv(256 * 2, 256))

        self.ups.append(
            nn.ConvTranspose2d(
                128 * 2,
                128,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleConv(128 * 2, 128))

        self.ups.append(
            nn.ConvTranspose2d(
                64 * 2,
                64,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleConv(64 * 2, 64))

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Leftside of W-Net
        skip_connections = []

        for down in self.downs:
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

        x = self.final_conv(x)
        # x = self.softmax2d(x)

        return x
    
class UDec(nn.Module):
    def __init__(
        self, in_channels, out_channels
    ):
        super(UDec, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downs.append(DoubleConv(in_channels, 64))
        self.downs.append(DoubleSepConv(64, 128))
        self.downs.append(DoubleSepConv(128, 256))
        self.downs.append(DoubleSepConv(256, 512))

        self.bottleneck = DoubleSepConv(512, 512 * 2)

        self.ups.append(
            nn.ConvTranspose2d(
                512 * 2,
                512,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleConv(512 * 2, 512))

        self.ups.append(
            nn.ConvTranspose2d(
                256 * 2,
                256,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleConv(256 * 2, 256))

        self.ups.append(
            nn.ConvTranspose2d(
                128 * 2,
                128,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleConv(128 * 2, 128))

        self.ups.append(
            nn.ConvTranspose2d(
                64 * 2,
                64,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DoubleConv(64 * 2, 64))

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Rightside of W-Net
        skip_connections = []

        for down in self.downs:
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

        x = self.final_conv(x)

        return x


class WNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, k=3
    ):
        super(WNet, self).__init__()

        self.enc = UEnc(in_channels, k)
        self.dec = UDec(k, out_channels)
        self.softmax2d = nn.Softmax2d()

    def forward(self, x, mode='dec'):

        encoded = self.enc(x)
        encoded = self.softmax2d(encoded)
        if mode == 'enc':
            return encoded

        decoded = self.dec(encoded)
        
        if mode == 'dec':
            return decoded
        
        return encoded, decoded