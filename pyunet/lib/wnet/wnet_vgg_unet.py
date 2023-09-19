import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vgg_cnn import VGG16
from unet import UNet


class VGGUNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, k=1
    ):
        super(VGGUNet, self).__init__()

        self.enc = VGG16(in_channels, k)
        self.dec = UNet(k, out_channels)
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

