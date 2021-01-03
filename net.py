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
from Inception_module import *

class fusion_can_multiscale(nn.Module):
    def __init__(self, recurrent_iter=5, use_GPU=True):
        super(fusion_can_multiscale, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = inception() #replacing conv0 from previous model by one inception module
        self.conv0_can = inception(in_channels = 32)
        #self.conv0_can = inception_v2(in_channels = 32)
##        self.conv0 = nn.Sequential(
##            nn.Conv2d(6, 32, 3, 1, 1),
##            nn.ReLU()
##            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )

##        self.res_conv6 = nn.Sequential(
##            nn.Conv2d(32, 32, 1, 1),
##            nn.ReLU(),
##            nn.Conv2d(32, 32, 3, 1, 1),
##            nn.ReLU()
##            )

        
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )

        self.fuse_1 = nn.Sequential(*[
            nn.Conv2d(32*2, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_2 = nn.Sequential(*[
            nn.Conv2d(32*3, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_3 = nn.Sequential(*[
            nn.Conv2d(32*4, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_4 = nn.Sequential(*[
            nn.Conv2d(32*5, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_5 = nn.Sequential(*[
            nn.Conv2d(32*6, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])

##        self.fuse_6 = nn.Sequential(*[
##            nn.Conv2d(32*7, 32, 1, 1, 0),
##            nn.ReLU(),
##            nn.Conv2d(32, 32*4, 3, 1, 1),
##            nn.PixelShuffle(2),
##            nn.Conv2d(32, 32, 3, 1, 1)
##            ])


        ##Context aggregation network
        self.can1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
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

        
##      feature fusion 
        self.fusion_can = nn.Sequential(
            nn.Conv2d(32*8, 32, 1, 1, 0),
            nn.Conv2d(32, 32, 3, 1, 1),
            )
        
        self.PixelShuffle_can = nn.Sequential(*[
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 3, 3, 1, 1),
            ])

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()


        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)
            x = h
            x_lstm_future = x
            
            x = F.relu(self.res_conv1(x) + x)
            c1 = torch.cat([x,x_lstm_future], dim=1)
            x = F.relu(self.fuse_1(c1))
            x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c1x = x

            
            x = F.relu(self.res_conv2(x) + x)
            c2 = torch.cat([x, x_lstm_future, c1x],dim=1)
            x = F.relu(self.fuse_2(c2))
            x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False)
            c2x = x

            x = F.relu(self.res_conv3(x) + x)
            c3 = torch.cat([c2x,c1x, x, x_lstm_future],dim=1)
            x = F.relu(self.fuse_3(c3))
            x = F.interpolate(x, size=c3.shape[-2:], mode='bilinear', align_corners=False)
            c3x = x

            x = F.relu(self.res_conv4(x) + x)
            c4 = torch.cat([c3x, c2x, c1x, x, x_lstm_future],dim=1)
            x = F.relu(self.fuse_4(c4))
            x = F.interpolate(x, size=c4.shape[-2:], mode='bilinear', align_corners=False)
            c4x = x

            x = F.relu(self.res_conv5(x) + x)
            c5 = torch.cat([c4x,c3x,c2x,c1x, x, x_lstm_future],dim=1)
            x = F.relu(self.fuse_5(c5))
            x = F.interpolate(x, size=c5.shape[-2:], mode='bilinear', align_corners=False)
            c5x = x

##            x = F.relu(self.res_conv6(x) + x)
##            c6 = torch.cat([c5x,c4x,c3x,c2x,c1x, x, x_lstm_future],dim=1)
##            x = F.relu(self.fuse_6(c6))
##            x = F.interpolate(x, size=c6.shape[-2:], mode='bilinear', align_corners=False)
##            c6x = x
       
    

            #CAN connections
            resx = x
            x = self.conv0_can(x)
            x1 = self.can1(x)
            x2 = self.can2(x1)
            x3 = self.can3(x2)
            x4 = self.can4(x3)
            x5 = self.can5(x4)
            x6 = self.can6(x5)
            x7 = self.can7(x6)
            x8 = self.can8(x7)
            #feature fusion and PixelShuffling
            x = self.fusion_can(torch.cat((x1,x2,x3,x4,x5,x6,x7,x8), dim=1))
            x = x + resx
            x = self.PixelShuffle_can(x)
            x = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
            
            x = x+input # long_residual added newly after running dense_can_in_the_loop_PS

        return x

###############################################################################

##Lightweight networks utilizing Group Convolutions
    
##Group convolutions only in residual blocks in v1
##Group convolutions in both residual block and fusion block in v2
class fusion_can_multiscale_lightweight_v1(nn.Module):
    def __init__(self, recurrent_iter=5, use_GPU=True):
        super(fusion_can_multiscale_lightweight_v1, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = inception() #replacing conv0 from previous model by one inception module
        self.conv0_can = inception(in_channels = 32)
        #self.conv0_can = inception_v2(in_channels = 32)
##        self.conv0 = nn.Sequential(
##            nn.Conv2d(6, 32, 3, 1, 1),
##            nn.ReLU()
##            )
        ##Efficient Residual Block
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
##        self.conv = nn.Sequential(
##            nn.Conv2d(32, 32, 3, 1, 1),
##            
##            )
        self.fuse_1 = nn.Sequential(*[
            nn.Conv2d(32*2, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_2 = nn.Sequential(*[
            nn.Conv2d(32*3, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_3 = nn.Sequential(*[
            nn.Conv2d(32*4, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_4 = nn.Sequential(*[
            nn.Conv2d(32*5, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_5 = nn.Sequential(*[
            nn.Conv2d(32*6, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])


        ##Context aggregation network
        self.can1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
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
##        self.conv_last = nn.Sequential(
##            nn.Conv2d(32, 3, 3, 1, 1)
##            )
        
##        feature fusion as in GraNet
        self.fusion_can = nn.Sequential(
            nn.Conv2d(32*8, 32, 1, 1, 0),
            nn.Conv2d(32, 32, 3, 1, 1),
            )
        ##PixelShuffling with upscale=2, output_#_channels=inp#/(upscale)^2
        self.PixelShuffle_can = nn.Sequential(*[
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 3, 3, 1, 1),
            ])

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()


        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)
            x = h
            x_lstm_future = x
            
            x = F.relu(self.res_conv1(x) + x)
            c1 = torch.cat([x,x_lstm_future], dim=1)
            x = F.relu(self.fuse_1(c1))
            x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c1x = x

            
##            x = F.relu(self.res_conv2(x) + x)
##            c2 = torch.cat([x, x_lstm_future, c1x],dim=1)
##            x = F.relu(self.fuse_2(c2))
##            x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False)
##            c2x = x

##            x = F.relu(self.res_conv3(x) + x)
##            c3 = torch.cat([c2x,c1x, x, x_lstm_future],dim=1)
##            x = F.relu(self.fuse_3(c3))
##            x = F.interpolate(x, size=c3.shape[-2:], mode='bilinear', align_corners=False)
##            c3x = x

##            x = F.relu(self.res_conv4(x) + x)
##            c4 = torch.cat([c3x, c2x, c1x, x, x_lstm_future],dim=1)
##            x = F.relu(self.fuse_4(c4))
##            x = F.interpolate(x, size=c4.shape[-2:], mode='bilinear', align_corners=False)
##            c4x = x
##
##            x = F.relu(self.res_conv5(x) + x)
##            c5 = torch.cat([c4x,c3x,c2x,c1x, x, x_lstm_future],dim=1)
##            x = F.relu(self.fuse_5(c5))
##            x = F.interpolate(x, size=c5.shape[-2:], mode='bilinear', align_corners=False)
##            c5x = x
    

            #CAN connections
            resx = x
            x = self.conv0_can(x)
            x1 = self.can1(x)
            x2 = self.can2(x1)
            x3 = self.can3(x2)
            x4 = self.can4(x3)
            x5 = self.can5(x4)
            x6 = self.can6(x5)
            x7 = self.can7(x6)
            x8 = self.can8(x7)
            #feature fusion and PixelShuffling
            x = self.fusion_can(torch.cat((x1,x2,x3,x4,x5,x6,x7,x8), dim=1))
            x = x + resx
            x = self.PixelShuffle_can(x)
            x = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
            
##            x = self.conv_last(x) #if no fusion and PixelShuffling used, uncomment this line
            x = x+input # long_residual added newly after running dense_can_in_the_loop_PS

        return x

##Group convolution in resnet and fusion block both
class fusion_can_multiscale_lightweight_v2(nn.Module):
    def __init__(self, recurrent_iter=5, use_GPU=True):
        super(fusion_can_multiscale_lightweight_v2, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = inception() #replacing conv0 from previous model by one inception module
        self.conv0_can = inception(in_channels = 32)
        #self.conv0_can = inception_v2(in_channels = 32)
##        self.conv0 = nn.Sequential(
##            nn.Conv2d(6, 32, 3, 1, 1),
##            nn.ReLU()
##            )
        ##EResBlock
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0)
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
##        self.conv = nn.Sequential(
##            nn.Conv2d(32, 32, 3, 1, 1),
##            
##            )
        self.fuse_1 = nn.Sequential(*[
            nn.Conv2d(32*2, 32, 1, 1, 0, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1, groups=2),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_2 = nn.Sequential(*[
            nn.Conv2d(32*3, 32, 1, 1, 0, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1, groups=2),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_3 = nn.Sequential(*[
            nn.Conv2d(32*4, 32, 1, 1, 0, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1, groups=2),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_4 = nn.Sequential(*[
            nn.Conv2d(32*5, 32, 1, 1, 0, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1, groups=2),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_5 = nn.Sequential(*[
            nn.Conv2d(32*6, 32, 1, 1, 0, groups=2),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1, groups=2),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])


        ##Context aggregation network
        self.can1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
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
##        self.conv_last = nn.Sequential(
##            nn.Conv2d(32, 3, 3, 1, 1)
##            )
        
##        feature fusion as in GraNet
        self.fusion_can = nn.Sequential(
            nn.Conv2d(32*8, 32, 1, 1, 0),
            nn.Conv2d(32, 32, 3, 1, 1),
            )
        ##PixelShuffling with upscale=2, output_#_channels=inp#/(upscale)^2
        self.PixelShuffle_can = nn.Sequential(*[
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 3, 3, 1, 1),
            ])

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()


        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)
            x = h
            x_lstm_future = x
            
            x = F.relu(self.res_conv1(x) + x)
            c1 = torch.cat([x,x_lstm_future], dim=1)
            x = F.relu(self.fuse_1(c1))
            x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c1x = x

            
##            x = F.relu(self.res_conv2(x) + x)
##            c2 = torch.cat([x, x_lstm_future, c1x],dim=1)
##            x = F.relu(self.fuse_2(c2))
##            x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False)
##            c2x = x

##            x = F.relu(self.res_conv3(x) + x)
##            c3 = torch.cat([c2x,c1x, x, x_lstm_future],dim=1)
##            x = F.relu(self.fuse_3(c3))
##            x = F.interpolate(x, size=c3.shape[-2:], mode='bilinear', align_corners=False)
##            c3x = x

##            x = F.relu(self.res_conv4(x) + x)
##            c4 = torch.cat([c3x, c2x, c1x, x, x_lstm_future],dim=1)
##            x = F.relu(self.fuse_4(c4))
##            x = F.interpolate(x, size=c4.shape[-2:], mode='bilinear', align_corners=False)
##            c4x = x
##
##            x = F.relu(self.res_conv5(x) + x)
##            c5 = torch.cat([c4x,c3x,c2x,c1x, x, x_lstm_future],dim=1)
##            x = F.relu(self.fuse_5(c5))
##            x = F.interpolate(x, size=c5.shape[-2:], mode='bilinear', align_corners=False)
##            c5x = x
    

            #CAN connections
            resx = x
            x = self.conv0_can(x)
            x1 = self.can1(x)
            x2 = self.can2(x1)
            x3 = self.can3(x2)
            x4 = self.can4(x3)
            x5 = self.can5(x4)
            x6 = self.can6(x5)
            x7 = self.can7(x6)
            x8 = self.can8(x7)
            #feature fusion and PixelShuffling
            x = self.fusion_can(torch.cat((x1,x2,x3,x4,x5,x6,x7,x8), dim=1))
            x = x + resx
            x = self.PixelShuffle_can(x)
            x = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
            
##            x = self.conv_last(x) #if no fusion and PixelShuffling used, uncomment this line
            x = x+input # long_residual added newly after running dense_can_in_the_loop_PS

        return x


class fusion_can_multiscale_frontend(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(fusion_can_multiscale_frontend, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = inception() #replacing conv0 from previous model by one inception module
##        self.conv0_can = inception(in_channels = 32)
        #self.conv0_can = inception_v2(in_channels = 32)
##        self.conv0 = nn.Sequential(
##            nn.Conv2d(6, 32, 3, 1, 1),
##            nn.ReLU()
##            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
##        self.conv = nn.Sequential(
##            nn.Conv2d(32, 32, 3, 1, 1),
##            
##            )
        self.fuse_1 = nn.Sequential(*[
            nn.Conv2d(32*2, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_2 = nn.Sequential(*[
            nn.Conv2d(32*3, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_3 = nn.Sequential(*[
            nn.Conv2d(32*4, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_4 = nn.Sequential(*[
            nn.Conv2d(32*5, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        
        self.fuse_5 = nn.Sequential(*[
            nn.Conv2d(32*6, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, 3, 1, 1)
            ])
        #to test only front-end performance we remove refinement module and use one
        #Convolution layer to produce 3 output feature maps from 32 input feature maps
        
        self.conv_last = nn.Conv2d(32,3,3,1,1)


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()


        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)
            x = h
            x_lstm_future = x
            
            x = F.relu(self.res_conv1(x) + x)
            c1 = torch.cat([x,x_lstm_future], dim=1)
            x = F.relu(self.fuse_1(c1))
            x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            c1x = x

            
            x = F.relu(self.res_conv2(x) + x)
            c2 = torch.cat([x, x_lstm_future, c1x],dim=1)
            x = F.relu(self.fuse_2(c2))
            x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False)
            c2x = x

            x = F.relu(self.res_conv3(x) + x)
            c3 = torch.cat([c2x,c1x, x, x_lstm_future],dim=1)
            x = F.relu(self.fuse_3(c3))
            x = F.interpolate(x, size=c3.shape[-2:], mode='bilinear', align_corners=False)
            c3x = x

            x = F.relu(self.res_conv4(x) + x)
            c4 = torch.cat([c3x, c2x, c1x, x, x_lstm_future],dim=1)
            x = F.relu(self.fuse_4(c4))
            x = F.interpolate(x, size=c4.shape[-2:], mode='bilinear', align_corners=False)
            c4x = x

            x = F.relu(self.res_conv5(x) + x)
            c5 = torch.cat([c4x,c3x,c2x,c1x, x, x_lstm_future],dim=1)
            x = F.relu(self.fuse_5(c5))
            x = F.interpolate(x, size=c5.shape[-2:], mode='bilinear', align_corners=False)
            c5x = x
    

            x = self.conv_last(x)
            x = x+input # long_residual added newly after running dense_can_in_the_loop_PS

        return x

    ###############################################################################

