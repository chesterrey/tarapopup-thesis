import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models
from double_unet import DoubleUnet

import cv2
import numpy as np

if __name__ == '__main__':
    model = DoubleUnet(3, 3)

    # load image 'D:/School Stuff/CSCI 199.2/pyunet/notebooks/images/ebhi-seg-polyp/images/gt2012173-1-400-001.png'
    image = cv2.imread('D:/School Stuff/CSCI 199.2/pyunet/notebooks/images/ebhi-seg-polyp/images/gt2012173-1-400-001.png')

    img = image.transpose((2, 0, 1))

    x = torch.Tensor(np.array([img]))

    result = model.forward(x)
    result = torch.argmax(result, 1).detach().cpu().numpy().astype(np.float32)
    result = result.transpose((1, 2, 0)) / 2

    cv2.imshow("result", result)
    cv2.imshow("Original", image)

    cv2.waitKey(0)


        