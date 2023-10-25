import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from alexnet_unet import AlexNet
from torch.hub import load_state_dict_from_url

class AlexNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AlexNetUNet, self).__init__()

        self.alexnet = AlexNet(in_channels, out_channels)

        pretrained_state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
            progress=True
        )

        alexnet_keys = list(self.alexnet.state_dict().keys())
        alexnet_keys = [key for key in alexnet_keys if 'downs' in key]

        pretrained_keys = list(pretrained_state_dict.keys())
        pretrained_keys = [key for key in pretrained_keys if 'features' in key]
        pretrained_keys = pretrained_keys[:10]

        for i in range(len(alexnet_keys)):
            pretrained_state_dict[alexnet_keys[i]] = pretrained_state_dict[pretrained_keys[i]]

        features = 0,3,6,8,10
        classifier = 1,4,6

        for i in features:
            del pretrained_state_dict[f'features.{i}.weight']
            del pretrained_state_dict[f'features.{i}.bias']
        
        for i in classifier:
            del pretrained_state_dict[f'classifier.{i}.weight']
            del pretrained_state_dict[f'classifier.{i}.bias']

        self.alexnet.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, x):
        
        return self.alexnet(x)
    

# (alexnet_): AlexNet(
# (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace=True)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace=True)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
# (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
# (classifier): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace=True)
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
# )
# )
# )