import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models

class DoubleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.vgg19(pretrained=True).features


if __name__ == '__main__':
    model = DoubleUnet()
    print(model)




        