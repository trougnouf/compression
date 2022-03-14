import torch
from torch import nn
import math
import sys
sys.path.append('..')
from compression.models import abstract_model
from compression.models import GDN
from dc.models.manynets_dc import ManyPriors_DC

# DEPRECATED in favor of ../compression/models/testolina_compressor.py

# ###################

class ManyPriors2xL_DC(ManyPriors_DC):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=Analysis2xL_net, decoder_cls=Synthesis2xL_net, **kwargs)

class Analysis2xL_net(nn.Module):
    '''
    Analysis net (liu's imprementation)
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, activation_function='GDN'):
        super().__init__()
        if activation_function == 'GDN':
            activation_function = GDN.GDN
            self.gdn1 = activation_function(out_channel_N)
            self.gdn2 = activation_function(out_channel_N)
            self.gdn3 = activation_function(out_channel_N)
            self.gdn1b = activation_function(out_channel_N)
            self.gdn2b = activation_function(out_channel_N)
            self.gdn3b = activation_function(out_channel_N)
            # TODO try w/ gdn4b
        else:
            raise NotImplementedError(activation_function)

        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)

        self.conv1b = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=1, padding=2)
        torch.nn.init.xavier_normal_(self.conv1b.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1b.bias.data, 0.01)

        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)

        self.conv2b = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=1, padding=2)
        torch.nn.init.xavier_normal_(self.conv2b.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2b.bias.data, 0.01)

        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

        self.conv3b = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=1, padding=2)
        torch.nn.init.xavier_normal_(self.conv3b.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3b.bias.data, 0.01)

        self.conv4 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

        self.conv4b = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=1, padding=2)
        torch.nn.init.xavier_normal_(self.conv4b.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4b.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn1b(self.conv1b(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn2b(self.conv2b(x))
        x = self.gdn3(self.conv3(x))
        x = self.gdn3b(self.conv3b(x))
        return self.conv4b(self.conv4(x)) # TODO try one more GDN


class Synthesis2xL_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, out_channel_fin=3, activation_function='GDN'):
        super().__init__()
        if activation_function == 'GDN':
            self.igdn1 = GDN.GDN(out_channel_N, inverse=True)
            self.igdn2 = GDN.GDN(out_channel_N, inverse=True)
            self.igdn3 = GDN.GDN(out_channel_N, inverse=True)
            self.igdn1b = GDN.GDN(out_channel_N, inverse=True)
            self.igdn2b = GDN.GDN(out_channel_N, inverse=True)
            self.igdn3b = GDN.GDN(out_channel_N, inverse=True)
        else:
            raise NotImplementedError(activation_function)

        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)

        self.deconv1b = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=1, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1b.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1b.bias.data, 0.01)

        #self.igdn1 = GDN.GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)

        self.deconv2b = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=1, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2b.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2b.bias.data, 0.01)

        #self.igdn2 = GDN.GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

        self.deconv3b = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=1, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3b.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3b.bias.data, 0.01)

        #self.igdn3 = GDN.GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

        self.deconv4b = nn.ConvTranspose2d(out_channel_N, out_channel_fin, 5, stride=1, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4b.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4b.bias.data, 0.01)


    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn1b(self.deconv1b(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn2b(self.deconv2b(x))
        x = self.igdn3(self.deconv3(x))
        x = self.igdn3b(self.deconv3b(x))
        x = self.deconv4(x)
        x = self.deconv4b(x)
        return x

class Analysis2017_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis2017_net, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN.GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN.GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN.GDN(out_channel_M)


    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return x

class Synthesis2017_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, out_channel_fin=3):
        super(Synthesis2017_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN.GDN(out_channel_M, inverse=True)

        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN.GDN(out_channel_N, inverse=True)

        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_fin, 9, stride=4, padding=3, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN.GDN(out_channel_N, inverse=True)


    def forward(self, x):
        x = self.deconv1(self.igdn1(x))
        x = self.deconv2(self.igdn2(x))
        x = self.deconv3(self.igdn3(x))
        return x
