import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from vgg_cnn import TripleConv2dRelu

class ResNet50(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet50, self).__init__()

        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downs.append(TripleConv2dRelu(in_channels, 512))
        self.downs.append(TripleConv2dRelu(256, 512))
        self.downs.append(TripleConv2dRelu(128, 256))
        self.downs.append(TripleConv2dRelu(64, 128))

        self.bottleneck = TripleConv2dRelu(64, 64 * 2)

        self.ups.append(
            nn.ConvTranspose2d(
                64 * 2,
                64,
                kernel_size=2,
                stride=2
            )
        )
        self.ups.append(TripleConv2dRelu(64 * 2, 64))
        self.ups.append(
            nn.ConvTranspose2d(
                128 * 2,
                128,
                kernel_size=2,
                stride=2
            )
        )
        self.ups.append(TripleConv2dRelu(128 * 2, 128))
        self.ups.append(
            nn.ConvTranspose2d(
                256 * 2,
                256,
                kernel_size=2,
                stride=2
            )
        )
        self.ups.append(TripleConv2dRelu(256 * 2, 256))
        self.ups.append(
            nn.ConvTranspose2d(
                512 * 2,
                512,
                kernel_size=2,
                stride=2
            )
        )
        self.ups.append(TripleConv2dRelu(512 * 2, 512))

        self.final_conv = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])

            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
    
    