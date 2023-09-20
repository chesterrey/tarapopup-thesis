import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from lib.double_conv import DoubleConv

class UNetWithResNet50(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetWithResNet50, self).__init__()

        # Load pre-trained ResNet-50 model as the encoder
        self.encoder = models.resnet50(pretrained=True)
        self.encoder_layers = list(self.encoder.children())

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Initialize the decoder part
        self.init_decoder()

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def init_decoder(self):
        # Define the decoder layers
        self.decoder = nn.ModuleList([
            DoubleConv(512, 256 * 2),
            DoubleConv(256 * 2, 128 * 2),
            DoubleConv(128 * 2, 64 * 2),
            DoubleConv(64 * 2, 64),
        ])

        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256 * 2, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128 * 2, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64 * 2, 64, kernel_size=2, stride=2),
        ])

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = down.pool(x)

        x = self.encoder(x)

        # Reverse skip_connections
        skip_connections = skip_connections[::-1]

        for idx, (up, double_conv) in enumerate(zip(self.ups, self.decoder)):
            x = up(x)
            skip_connection = skip_connections[idx]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = double_conv(concat_skip)

        return self.final_conv(x)
