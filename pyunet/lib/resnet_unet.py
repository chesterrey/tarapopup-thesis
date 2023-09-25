import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.hub import load_state_dict_from_url
from lib.resnet_cnn import ResNet50
import torchvision.models as models

class ResNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetUNet, self).__init__()

        self.resnet = ResNet50(in_channels, out_channels)

        pretrained_state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet50-0676ba61.pth',
            progress=True
        )
        #print("Length of Pretrained State Dict:"+str(pretrained_state_dict))

        resnet_keys = list(self.resnet.state_dict().keys())
        #for i in range(len(resnet_keys)):
        #    print(resnet_keys[i])
        resnet_keys = [key for key in resnet_keys if 'downs' in key]


        pretrained_keys = list(pretrained_state_dict.keys())
        for i in range(len(pretrained_keys)):
            print(pretrained_keys[i])
        pretrained_keys = [key for key in pretrained_keys if not 'fc' in key]
        pretrained_keys = pretrained_keys[:-6]
        
        print("Length of ResNet Keys:"+str(len(resnet_keys))+ "\nLength of Pretrained Keys:"+str(len(pretrained_keys)))
        if len(resnet_keys) != len(pretrained_keys):
            raise ValueError("Mismatched lengths of resnet_keys and pretrained_keys")

        for resnet_key, pretrained_key in zip(resnet_keys, pretrained_keys):
            pretrained_state_dict[resnet_key] = pretrained_state_dict[pretrained_key]

                
        features = 0,2,5,7,10,12,14,17,19,21,24,26,28
        classifier = 0,3,6    

        for i in features:
            del pretrained_state_dict[f'features.{i}.weight']
            del pretrained_state_dict[f'features.{i}.bias']
        
        for i in classifier:
            del pretrained_state_dict[f'classifier.{i}.weight']
            del pretrained_state_dict[f'classifier.{i}.bias']
        
        self.resnet.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
