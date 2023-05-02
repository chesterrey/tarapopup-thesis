from wnet import WNet
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def test_wnet():

    model = WNet(in_channels=3, out_channels=3)

    img = cv2.imread('test.jpg')
    print(img.shape)
    input_img = img.transpose((2, 0, 1))
    x = torch.Tensor(np.array([input_img]))

    result = model.forward(x)
    # convert to numpy array
    result = result.detach().numpy()[0]
    # transpose to (H, W, C)
    result = result.transpose((1, 2, 0))
    print(result.shape)

    cv2.imshow('image2', img)
    cv2.imshow('image', result)
    cv2.waitKey(0)

test_wnet()