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

class inception(nn.Module):
    def __init__(self, use_GPU=True, in_channels = 6):
        super(inception,self).__init__()
        self.use_GPU = use_GPU
        self.in_channels = in_channels
###1x1,3x3,5x5==>8,12,12 feature maps
###3x3,5x5,7x7==>10,11,11 feature maps
        self.scale1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, 1, 1, 0)
            )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(self.in_channels, 12, 1, 1, 0),
            nn.Conv2d(12, 12, 3, 1, 1)
            )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(self.in_channels, 12, 1, 1, 0),
            nn.Conv2d(12, 12, 5, 1, 2)
            )
        
##        self.scale4 = nn.Sequential(
##            nn.Conv2d(self.in_channels, 11, 1, 1, 0),
##            nn.Conv2d(11, 11, 7, 1, 3)
##            )

    def forward(self, input):
        o1 = self.scale1(input)
        o2 = self.scale2(input)
        o3 = self.scale3(input)
##        o4 = self.scale4(input)

        output = F.relu(torch.cat([o1,o2,o3], dim=1))

        return output


class inception_v2(nn.Module):
    def __init__(self, use_GPU=True, in_channels = 6):
        super(inception_v2,self).__init__()
        self.use_GPU = use_GPU
        self.in_channels = in_channels
###1x1,3x3,5x5==>8,12,12 feature maps
###3x3,5x5,7x7==>10,11,11 feature maps
        self.scale1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, 1, 1, 0)
            )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(self.in_channels, 12, 1, 1, 0),
            nn.Conv2d(12, 12, 3, 1, 1)
            )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(self.in_channels, 12, 1, 1, 0),
            nn.Conv2d(12, 12, (1,5), 1, 1),
            nn.Conv2d(12, 12, (5,1), 1, 1)
            )
        
##        self.scale4 = nn.Sequential(
##            nn.Conv2d(self.in_channels, 11, 1, 1, 0),
##            nn.Conv2d(11, 11, 7, 1, 3)
##            )

    def forward(self, input):
        o1 = self.scale1(input)
        o2 = self.scale2(input)
        o3 = self.scale3(input)
##        o4 = self.scale4(input)

        output = F.relu(torch.cat([o1,o2,o3], dim=1))

        return output

class jordar_multiscale(nn.Module):
    def __init__(self, use_GPU=True, in_channels = 6):
        super(jordar_multiscale, self).__init__()
        self.use_GPU = use_GPU
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(self.in_channels, 32, 3, 1, 1)
        self.scale1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, dilation = 1),
            nn.Conv2d(32, 32, 3, 1, 1, dilation=1)
            )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 2, dilation = 2),
            nn.Conv2d(32, 32, 3, 1, 2, dilation = 2)
            )

        self.scale3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 3, dilation = 3),
            nn.Conv2d(32, 32, 3, 1, 3, dilation = 3)
            )
    def forward(self,input):
        conv_out = self.conv1(input)
        o1 = self.scale1(conv_out)
        o2 = self.scale1(conv_out)
        o3 = self.scale1(conv_out)

        output = o1+o2+o3

        return output
        
            

        
