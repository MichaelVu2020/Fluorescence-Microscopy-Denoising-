#!/usr/bin/env python

from __future__ import print_function, division
import torch.utils.data
from torch.nn import Parameter
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# URDNetx2_RENOIR+SIDD
class FeedbackURDNet(nn.Module):

    def __init__(self, in_chs=1, out_chs=1, mid_feats=64, num_fb=3, nDlayer=8, growthRate=16):
        super(FeedbackURDNet, self).__init__()
        self.inx = nn.Sequential(
            nn.Conv2d(in_chs, mid_feats, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_feats, mid_feats, kernel_size=4, stride=2, padding=1),)

        self.num_fb = num_fb
        modules = []
        for i in range(self.num_fb):
            modules.append(FeedbackMainNet(in_nc=mid_feats, out_nc=mid_feats, nDlayer=nDlayer, growthRate=growthRate))
        self.main = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(mid_feats * (self.num_fb), mid_feats, kernel_size=1, stride=1, padding=0)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(mid_feats, mid_feats, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(mid_feats, out_chs, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x, seg):
        fea0 = self.inx(x)
        fea = fea0
        cond = self.inx(seg)
        outputlist = []
        for main in self.main:
            fea = main((fea, cond))
            outputlist.append(fea)
        concat = torch.cat(outputlist, 1)
        fea = fea0 + self.conv_1x1(concat)
        out = self.out(fea) + x
        return out


# Sub classes
class FeedbackMainNet(nn.Module):
    def __init__(self, in_nc=64, out_nc=64, nDlayer=4, growthRate=16, ):
        super(FeedbackMainNet, self).__init__()
        self.inc = nn.Sequential(
            # single_conv(in_nc, feats),
            FeedbackRDB3(in_nc, nDlayer, growthRate),
        )
        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(in_nc, in_nc * 2),
            RDB(in_nc * 2, nDlayer, growthRate),
        )
        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(in_nc * 2, in_nc * 4),
            RDB(in_nc * 4, nDlayer, growthRate),
        )
        self.up1 = up(in_nc * 4)
        self.conv3 = nn.Sequential(
            RDB(in_nc * 2, nDlayer, growthRate),
        )
        self.up2 = up(in_nc * 2)
        self.conv4 = nn.Sequential(
            RDB(in_nc, nDlayer, growthRate),
        )
        self.outc = outconv(in_nc, out_nc)



    def forward(self, x):
        # x[0]: fea; x[1]: cond
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# subnet for feedback net
class SFTLayer3(nn.Module):
    def __init__(self, nChannels, nChannels_):
        super(SFTLayer3, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(nChannels, nChannels, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nChannels, nChannels_, 1)
        self.SFT_shift_conv0 = nn.Conv2d(nChannels, nChannels, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nChannels, nChannels_, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        # print(x[0].shape,x[1].shape,scale.shape,shift.shape)
        return x[0] * (scale + 1) + shift

class make_dense_SFT3(nn.Module):
    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_dense_SFT3, self).__init__()
        self.sft = SFTLayer3(nChannels, nChannels_)
        self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.nChannels = nChannels

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft(x)
        fea = F.relu(self.conv(fea))
        fea = torch.cat((x[0], fea), 1)
        return (fea, x[1])  # return a tuple containing features and conditions

class FeedbackRDB3(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, nChannels, nDenselayer, growthRate):

        super(FeedbackRDB3, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense_SFT3(nChannels, nChannels_, growthRate))
            nChannels_ += growthRate
        modules.append(SFTLayer3(nChannels, nChannels_))
        self.dense_layers = nn.Sequential(*modules)
        ###################kingrdb ver2##############################################
        # self.conv_1x1 = nn.Conv2d(nChannels_ + growthRate, nChannels, kernel_size=1, padding=0, bias=False)
        ###################else######################################################
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.dense_layers(x)
        fea = self.conv_1x1(fea)
        # local residual 구조
        fea = fea + x[0]
        return fea



# extract feat
def weights_init_kaiming(lyr):

    if isinstance(lyr, nn.Conv2d):
        nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')

    # elif classname.find('Linear') != -1:
    elif isinstance(lyr, nn.Linear):
        nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
    # elif classname.find('BatchNorm') != -1:
    elif isinstance(lyr, nn.BatchNorm2d):
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
            clamp_(-0.025, 0.025)
        nn.init.constant_(lyr.bias.data, 0.0)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


