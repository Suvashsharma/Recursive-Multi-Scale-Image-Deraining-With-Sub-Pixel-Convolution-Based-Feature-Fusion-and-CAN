#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
class can_only(nn.Module):
    def __init__(self, recurrent_iter = 6, use_GPU=True):
        super(can_only,self).__init__()
        self.recurrent_iter = recurrent_iter
        self.use_GPU = use_GPU
        ##Context aggregation network
        self.can1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, 1),
            nn.ReLU()
            )
        self.can2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 2, 2), ##2 is dilation rate
            nn.ReLU()
            )
        self.can3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 4, 4),
            nn.ReLU()
            )
        self.can4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 8, 8),
            nn.ReLU()
            )
        self.can5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 16, 16),
            nn.ReLU()
            )
        self.can6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 16, 16), 
            nn.ReLU()
            )
        
        self.can7 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU()
            )
        self.can8 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU()
            )
        ##feature fusion as in GraNet
        self.fusion = nn.Sequential(
            nn.Conv2d(32*8, 32, 1, 1, 0),
            nn.Conv2d(32, 32, 3, 1, 1),
            )
        ##PixelShuffling with upscale=2, output_#_channels=inp#/(upscale)^2
        self.PixelShuffle = nn.Sequential(*[
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 3, 3, 1, 1),
            ])

    def forward(self, input):
        x=input
        #CAN connections
        for i in range(self.recurrent_iter):
            x1 = self.can1(x)
            x2 = self.can2(x1)
            x3 = self.can3(x2)
            x4 = self.can4(x3)
            x5 = self.can5(x4)
            x6 = self.can6(x5)
            x7 = self.can7(x6)
            x8 = self.can8(x7)
            #feature fusion and PixelShuffling
            x = self.fusion(torch.cat((x1,x2,x3,x4,x5,x6,x7,x8), dim=1))
            x = self.PixelShuffle(x)
            x = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
            x = x+input

        return x
