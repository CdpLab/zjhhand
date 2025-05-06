import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from dataset.dataset_utils import IMG_SIZE
from utils.utils import projection_batch
from models.manolayer import ManoLayer
from models.model_zoo import get_hrnet, conv1x1, conv3x3, deconv3x3, weights_init, GCN_vert_convert, build_fc_layer, Bottleneck

from utils.config import load_cfg

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNetSimple_decoder(nn.Module):
    def __init__(self, expansion=4,
                 fDim=[256, 256, 256, 256], direction=['flat', 'up', 'up', 'up'],
                 out_dim=3):
        super(ResNetSimple_decoder, self).__init__()
        self.models = nn.ModuleList()
        fDim = [512 * expansion] + fDim
        for i in range(len(direction)):
            kernel_size = 1 if direction[i] == 'flat' else 3
            self.models.append(self.make_layer(fDim[i], fDim[i + 1], direction[i], kernel_size=kernel_size))

        self.final_layer = nn.Conv2d(
            in_channels=fDim[-1],
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def make_layer(self, in_dim, out_dim,
                   direction, kernel_size=3, relu=True, bn=True):
        assert direction in ['flat', 'up']
        assert kernel_size in [1, 3]
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0

        layers = []
        if direction == 'up':
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        fmaps = []
        for i in range(len(self.models)):
            x = self.models[i](x)
            fmaps.append(x)
        x = self.final_layer(x)
        return x, fmaps


class ResNetSimple(nn.Module):
    def __init__(self, model_type='resnet50',
                 pretrained=False,
                 fmapDim=[256, 256, 256, 256],
                 handNum=2,
                 heatmapDim=21):
        super(ResNetSimple, self).__init__()
        assert model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
            self.expansion = 1
        elif model_type == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
            self.expansion = 1
        elif model_type == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
            self.expansion = 4
        elif model_type == 'resnet101':
            self.resnet = resnet101(pretrained=pretrained)
            self.expansion = 4
        elif model_type == 'resnet152':
            self.resnet = resnet152(pretrained=pretrained)
            self.expansion = 4

        self.hms_decoder = ResNetSimple_decoder(expansion=self.expansion,
                                                fDim=fmapDim,
                                                direction=['flat', 'up', 'up', 'up'],
                                                out_dim=heatmapDim * handNum)
        for m in self.hms_decoder.modules():
            weights_init(m)

        self.dp_decoder = ResNetSimple_decoder(expansion=self.expansion,
                                               fDim=fmapDim,
                                               direction=['flat', 'up', 'up', 'up'],
                                               out_dim=handNum + 3 * handNum)
        self.handNum = handNum

        for m in self.dp_decoder.modules():
            weights_init(m)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x4 = self.resnet.layer1(x)
        x3 = self.resnet.layer2(x4)
        x2 = self.resnet.layer3(x3)
        x1 = self.resnet.layer4(x2)

        img_fmaps = [x1, x2, x3, x4]

        hms, hms_fmaps = self.hms_decoder(x1)
        out, dp_fmaps = self.dp_decoder(x1)
        mask = out[:, :self.handNum]
        dp = out[:, self.handNum:]

        return hms, mask, dp, \
            img_fmaps, hms_fmaps, dp_fmaps


class resnet_mid(nn.Module):
    def __init__(self,
                 model_type='resnet50',
                 in_fmapDim=[256, 256, 256, 256],
                 out_fmapDim=[256, 256, 256, 256]):
        super(resnet_mid, self).__init__()
        assert model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if model_type == 'resnet18' or model_type == 'resnet34':
            self.expansion = 1
        elif model_type == 'resnet50' or model_type == 'resnet101' or model_type == 'resnet152':
            self.expansion = 4

        self.img_fmaps_dim = [512 * self.expansion, 256 * self.expansion,
                              128 * self.expansion, 64 * self.expansion]
        self.dp_fmaps_dim = in_fmapDim
        self.hms_fmaps_dim = in_fmapDim

        self.convs = nn.ModuleList()
        for i in range(len(out_fmapDim)):
            inDim = self.dp_fmaps_dim[i] + self.hms_fmaps_dim[i]
            if i > 0:
                inDim = inDim + self.img_fmaps_dim[i]
            self.convs.append(conv1x1(inDim, out_fmapDim[i]))

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

        self.global_feature_dim = 512 * self.expansion
        self.fmaps_dim = out_fmapDim

    def get_info(self):
        return {'global_feature_dim': self.global_feature_dim,
                'fmaps_dim': self.fmaps_dim}

    def forward(self, img_fmaps, hms_fmaps, dp_fmaps):
        #global_feature = self.output_layer(img_fmaps[0])
        global_feature = img_fmaps[0]
        fmaps = []
        for i in range(len(self.convs)):
            x = torch.cat((hms_fmaps[i], dp_fmaps[i]), dim=1)
            if i > 0:
                x = torch.cat((x, img_fmaps[i]), dim=1)
            fmaps.append(self.convs[i](x))
        return global_feature, fmaps

def load_encoder(cfg):
    if cfg.MODEL.ENCODER_TYPE.find('resnet') != -1:
        encoder = ResNetSimple(model_type=cfg.MODEL.ENCODER_TYPE,
                               pretrained=True,
                               fmapDim=[128, 128, 128, 128],
                               handNum=2,
                               heatmapDim=21)
        mid_model = resnet_mid(model_type=cfg.MODEL.ENCODER_TYPE,
                               in_fmapDim=[128, 128, 128, 128],
                               out_fmapDim=cfg.MODEL.DECONV_DIMS)
    return encoder, mid_model
