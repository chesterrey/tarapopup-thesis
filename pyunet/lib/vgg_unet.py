import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.hub import load_state_dict_from_url
from lib.vgg_cnn import VGG16
import torchvision.models as models

class VGGUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGUNet, self).__init__()

        self.vgg16 = VGG16(in_channels, out_channels)

        pretrained_state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/vgg16-397923af.pth',
            progress=True
        )

        vgg16_keys = list(self.vgg16.state_dict().keys())
        vgg16_keys = [key for key in vgg16_keys if 'downs' in key]


        pretrained_keys = list(pretrained_state_dict.keys())
        pretrained_keys = [key for key in pretrained_keys if 'features' in key]
        pretrained_keys = pretrained_keys[:-6]
        

        for i in range(len(vgg16_keys)):
            pretrained_state_dict[vgg16_keys[i]] = pretrained_state_dict[pretrained_keys[i]]
                
        features = 0,2,5,7,10,12,14,17,19,21,24,26,28
        classifier = 0,3,6    

        for i in features:
            del pretrained_state_dict[f'features.{i}.weight']
            del pretrained_state_dict[f'features.{i}.bias']
        
        for i in classifier:
            del pretrained_state_dict[f'classifier.{i}.weight']
            del pretrained_state_dict[f'classifier.{i}.bias']
        
        self.vgg16.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, x):
        x = self.vgg16(x)
        return x

# VGGUNet(
#   (vgg16): VGG(
#     (features): Sequential(
#       (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): ReLU(inplace=True)
#       (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (3): ReLU(inplace=True)
#       (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (6): ReLU(inplace=True)
#       (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (8): ReLU(inplace=True)
#       (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (11): ReLU(inplace=True)
#       (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (13): ReLU(inplace=True)
#       (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (15): ReLU(inplace=True)
#       (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (18): ReLU(inplace=True)
#       (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (20): ReLU(inplace=True)
#       (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (22): ReLU(inplace=True)
#       (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (25): ReLU(inplace=True)
#       (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (27): ReLU(inplace=True)
#       (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (29): ReLU(inplace=True)
#       (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     )
#     (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#     (classifier): Sequential(
#       (0): Linear(in_features=25088, out_features=4096, bias=True)
#       (1): ReLU(inplace=True)
#       (2): Dropout(p=0.5, inplace=False)
#       (3): Linear(in_features=4096, out_features=4096, bias=True)
#       (4): ReLU(inplace=True)
#       (5): Dropout(p=0.5, inplace=False)
#       (6): Linear(in_features=4096, out_features=1000, bias=True)
#     )
#   )
# )