# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:54:11 2022

@author: masteryi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np 
import torch.nn.init as init
import torch.nn as nn

from models import register
from argparse import Namespace


def xavier(param):
    # init.xavier_uniform(param)
    nn.init.xavier_uniform_(param)

class SingleLayer(nn.Module):
    def __init__(self, inChannels,growthRate):
        super(SingleLayer, self).__init__()
        self.conv =nn.Conv2d(inChannels,growthRate,kernel_size=3,padding=1, bias=True)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    
class SingleBlock(nn.Module):
    def __init__(self, inChannels,growthRate,nDenselayer):
        super(SingleBlock, self).__init__()
        self.block= self._make_dense(inChannels,growthRate, nDenselayer)
        
    def _make_dense(self,inChannels,growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels,growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)
                
    def forward(self, x):
        out=self.block(x)
        return out

class SRDenseNet(nn.Module):
    def __init__(self,inChannels,growthRate,nDenselayer,nBlock,no_upsampling):
        super(SRDenseNet,self).__init__()
        
        self.no_upsampling = no_upsampling
        self.out_dim = 64
        
        self.conv1 = nn.Conv2d(3,growthRate,kernel_size=3, padding=1,bias=True)
        
        inChannels = growthRate
        
        self.denseblock = self._make_block(inChannels,growthRate, nDenselayer,nBlock)
        inChannels +=growthRate* nDenselayer*nBlock
        
        self.Bottleneck = nn.Conv2d(in_channels=inChannels, out_channels=128, kernel_size=1,padding=0, bias=True)
        
        self.Bottleneck_to64 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1,padding=0, bias=True)
        
        self.convt1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        
        self.convt2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        
        self.conv2 =nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3,padding=1, bias=True)
    
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _make_block(self, inChannels,growthRate, nDenselayer,nBlock):
        blocks =[]
        for i in range(int(nBlock)):
            blocks.append(SingleBlock(inChannels,growthRate,nDenselayer))
            inChannels += growthRate* nDenselayer
        return nn.Sequential(* blocks)
    
    def forward(self,x):
        out = F.relu(self.conv1(x))
        # print(out.shape) # 8
        out = self.denseblock(out)
        # print(out.shape) # 520
        out = self.Bottleneck(out)
        # print(out.shape) # 128
        out = self.Bottleneck_to64(out)
        
        if self.no_upsampling:
            return out
        
        out = self.convt1(out)
        out = self.convt2(out)
        HR = self.conv2(out)
        
        return HR


@register('sr-dense')
def make_dense(inChannels = 8, growthRate = 8, nDenselayer = 8, nBlock = 8, no_upsampling = False):
    args = Namespace()
    args.inChannels = 8
    args.growthRate = 8
    args.nDenselayer = 8
    args.nBlock = 8
    args.no_upsampling = True
    return SRDenseNet(args.inChannels,args.growthRate,args.nDenselayer,args.nBlock,args.no_upsampling)



if __name__ == '__main__':
    x = torch.randn(4,3,48,48)
    model = make_dense()
    y = model(x)
    print(y.shape)
