import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from lib.double_conv import DoubleConv

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(Conv2dReLU, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x
    

class AlexNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(AlexNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (AlexNet features)
        self.downs = nn.ModuleList()

        # Bottom of U-Net

        # Decoder
        self.ups = nn.ModuleList()

        self.downs.append(
            Conv2dReLU(
                in_channels,
                64,
                kernel_size=11,
                padding=5
            )
        )
        self.downs.append(
            Conv2dReLU(
                64,
                192,
                kernel_size=5,
                padding=2
            )
        )
        self.downs.append(
            nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        )

        self.bottleneck = DoubleConv(256, 256 * 2)

        self.ups.append(
            nn.ConvTranspose2d(
                256 * 2,
                256,
                kernel_size=2,
                stride=2
            )
        )
        self.ups.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        )
        self.ups.append(
            nn.ConvTranspose2d(
                192,
                192,
                kernel_size=2,
                stride=2
            )
        )
        self.ups.append(
            Conv2dReLU(
                192 * 2,
                64,
                kernel_size=5,
                padding=2
            )
        )
        self.ups.append(
            nn.ConvTranspose2d(
                64,
                64,
                kernel_size=2,
                stride=2
            )
        )
        self.ups.append(
            Conv2dReLU(
                64 * 2,
                64,
                kernel_size=11,
                padding=5
            )
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom
        x = self.bottleneck(x)

        # Decoder with skip connections
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
