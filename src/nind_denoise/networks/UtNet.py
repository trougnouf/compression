# -*- coding: utf-8 -*-
from torch import nn
import torch
'''
# U-Net with transposed convolutions (consistent shape). Also w/dense concat connections (UtdNet)
# ((((INPUTSIZE+2*BUILTINPADDING-4)/2-4)/2-4)/2-4)/2-2
# bottom: (((cs÷2−4)÷2−4)÷2−4)÷2−2
Activation function: probably best to go with quantizable methods listed in 
https://pytorch.org/docs/stable/torch.nn.quantized.html
s.a. ELU, Hardswish
torch.nn.intrisic.ConvReLU2d might also be useful here
'''
class UtNet(nn.Module):
    def __init__(self, funit = 64, activation = 'PReLU'):
        super(UtNet, self).__init__()
        activation_params = {}
        if activation == 'PReLU':
            activation_fun = nn.PReLU
        elif activation == 'ELU':
            activation_fun = nn.ELU
            activation_params['inplace'] = True
        elif activation == 'Hardswish':
            activation_fun = nn.Hardswish
            activation_params['inplace'] = True
        else:
            exit(f'UtNet: unknown activation function: {activation}')
        self.pad = nn.ReflectionPad2d(2)
        self.convs1 = nn.Sequential(
                nn.Conv2d(3, funit, 3),
                activation_fun(**activation_params),
                nn.Conv2d(funit, funit, 3),
                activation_fun(**activation_params)
                )
        self.maxpool = nn.MaxPool2d(2)
        self.convs2 = nn.Sequential(
                nn.Conv2d(funit, 2*funit, 3),
                activation_fun(**activation_params),
                nn.Conv2d(2*funit, 2*funit, 3),
                activation_fun(**activation_params)
                )
        self.convs3 = nn.Sequential(
                nn.Conv2d(2*funit, 4*funit, 3),
                activation_fun(**activation_params),
                nn.Conv2d(4*funit, 4*funit, 3),
                activation_fun(**activation_params)
                )
        self.convs4 = nn.Sequential(
                nn.Conv2d(4*funit, 8*funit, 3),
                activation_fun(**activation_params),
                nn.Conv2d(8*funit, 8*funit, 3),
                activation_fun(**activation_params)
                )
        self.bottom = nn.Sequential(
                nn.Conv2d(8*funit, 16*funit, 3),
                activation_fun(**activation_params),
                nn.ConvTranspose2d(16*funit, 16*funit, 3),
                activation_fun(**activation_params)
                )
        self.up1 = nn.ConvTranspose2d(16*funit, 8*funit, 2, stride=2)
        self.tconvs1 = nn.Sequential(
                nn.ConvTranspose2d(16*funit, 8*funit, 3),
                activation_fun(**activation_params),
                nn.ConvTranspose2d(8*funit, 8*funit, 3),
                activation_fun(**activation_params)
                )
        self.up2 = nn.ConvTranspose2d(8*funit, 4*funit, 2, stride=2)
        self.tconvs2 = nn.Sequential(
                nn.ConvTranspose2d(8*funit, 4*funit, 3),
                activation_fun(**activation_params),
                nn.ConvTranspose2d(4*funit, 4*funit, 3),
                activation_fun(**activation_params)
                )
        self.up3 = nn.ConvTranspose2d(4*funit, 2*funit, 2, stride=2)
        self.tconvs3 = nn.Sequential(
                nn.ConvTranspose2d(4*funit, 2*funit, 3),
                activation_fun(**activation_params),
                nn.ConvTranspose2d(2*funit, 2*funit, 3),
                activation_fun(**activation_params)
                )
        self.up4 = nn.ConvTranspose2d(2*funit, funit, 2, stride=2)
        self.tconvs4 = nn.Sequential(
                nn.ConvTranspose2d(2*funit, funit, 3),
                activation_fun(**activation_params),
                nn.ConvTranspose2d(funit, funit, 3),
                activation_fun(**activation_params),
                nn.Conv2d(funit, 3, 1)
                )
        self.unpad = nn.ZeroPad2d(-2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # TODO try xavier_normal_ ?
        
    def forward(self, l):
        l = self.pad(l)
        l1 = self.convs1(l)
        l2 = self.convs2(self.maxpool(l1))
        l3 = self.convs3(self.maxpool(l2))
        l4 = self.convs4(self.maxpool(l3))
        l = torch.cat([self.up1(self.bottom(self.maxpool(l4))), l4], dim=1)
        l = torch.cat([self.up2(self.tconvs1(l)), l3], dim=1)
        l = torch.cat([self.up3(self.tconvs2(l)), l2], dim=1)
        l = torch.cat([self.up4(self.tconvs3(l)), l1], dim=1)
        l = self.tconvs4(l)
        l = self.unpad(l)
        return l


class UpsNet(nn.Module):
    def __init__(self, funit=64, out_activation='PReLU'):
        pass  # TODO pixelshuffle

def testNets():
    at = torch.rand(10,3,136,136)
    net = UtNet()
    res = net(at)
    assert at.shape == res.shape