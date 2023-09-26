import torch
import torch.nn as nn
import torchvision.models as models
    
class ResNet50(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet50, self).__init__()

        # Create the ResNet-50 backbone
        resnet = models.resnet50(num_classes = in_channels, pretrained=True)
        # Remove the classification head (fc layer)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

        # Create a bridge layer
        self.bridge = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)   
        )

        reversed_layer3 = nn.Sequential(*list(reversed(resnet.layer3)))
        reversed_layer2 = nn.Sequential(*list(reversed(resnet.layer2)))
        reversed_layer1 = nn.Sequential(*list(reversed(resnet.layer1)))
        
        # Create the decoder
        self.decoder = nn.Sequential(
            reversed_layer3,
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            reversed_layer2,
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            reversed_layer1,
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Bridge
        x2 = self.bridge(x1)

        # Decoder
        x3 = self.decoder(x2)

        # Final output
        output = self.final_conv(x3)

        return output