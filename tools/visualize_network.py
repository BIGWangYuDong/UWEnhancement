import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')
import torchvision.models as models
from UW.core.Models.weight_init import normal_init


# print(densenet.features)
# visualize the network
import torch
from UW.core.Models.builder import NETWORK, build_backbone
from UW.core.Models.base_model import BaseNet
from UW.core.Models.backbone.densenet import DenseBlock
from UW.core.Models.backbone.resnet import Bottleneck
from UW.core.Models.weight_init import normal_init



# @NETWORK.register_module()
class DehazeNet(BaseNet):
    def __init__(self,
                 backbone,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True, ):
        super(DehazeNet, self).__init__(backbone, pretrained, init_weight_type, get_parameter)
        self.backbone = build_backbone(backbone)
        if init_weight_type is not None:
            self.init_weight_type = init_weight_type
        self.get_parameter = get_parameter
        self._init_layers()
        self.init_weight(pretrained=pretrained)
        self.get_parameters()
        print(self.net_parameters)
        print()

    def _init_layers(self):
        inplanes1 = 2048
        inplanes2 = 1536
        inplanes3 = 768
        inplanes4 = 512
        inplanes5 = 256
        outplanes = 32
        block = Bottleneck
        self.relu = nn.ReLU(inplace=True)
        self.EMdownsample1 = nn.AvgPool2d(2)
        self.EMdownsample2 = nn.AvgPool2d(2)
        self.EMdownsample3 = nn.AvgPool2d(2)
        self.EMdownsample4 = nn.AvgPool2d(2)
        self.EMdownsample5 = nn.AvgPool2d(2)

        self.CASAdownsample1 = nn.AvgPool2d(2)
        self.CASAdownsample2 = nn.AvgPool2d(2)
        self.CASAdownsample3 = nn.AvgPool2d(2)
        self.CASAdownsample4 = nn.AvgPool2d(2)
        self.CASAdownsample5 = nn.AvgPool2d(2)

        self.EMblock1 = EMBlock(index=0, in_fea=3, mid_fea=32, out_fea=32)
        self.EMblock2 = EMBlock(index=2, in_fea=128 + 32 + 32, mid_fea=64, out_fea=64)
        self.EMblock3 = EMBlock(index=4, in_fea=256 + 64 + 64, mid_fea=128, out_fea=128)
        self.EMblock4 = EMBlock(index=8, in_fea=256 + 128 + 128, mid_fea=256, out_fea=256)
        self.EMblock5 = EMBlock(index=16, in_fea=512 + 256 + 256, mid_fea=256, out_fea=512)

        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()
        self.SA5 = SpatialAttention()

        self.CA1 = ChannelAttention(32)
        self.CA2 = ChannelAttention(64)
        self.CA3 = ChannelAttention(128)
        self.CA4 = ChannelAttention(256)
        self.CA5 = ChannelAttention(512)

        self.inplanes = inplanes1
        self.Resblock1 = self._make_reslayer(block, planes=int(inplanes1 / 4), blocks=3)
        self.inplanes = inplanes2
        self.Resblock2 = self._make_reslayer(block, planes=int(inplanes2 / 4), blocks=3)
        self.inplanes = inplanes3
        self.Resblock3 = self._make_reslayer(block, planes=int(inplanes3 / 4), blocks=3)
        self.inplanes = inplanes4
        self.Resblock4 = self._make_reslayer(block, planes=int(inplanes4 / 4), blocks=3)
        self.inplanes = inplanes5
        self.Resblock5 = self._make_reslayer(block, planes=int(inplanes5 / 4), blocks=3)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Res_conv1 = nn.Conv2d(inplanes1, 512, 1, 1, 0)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Res_conv2 = nn.Conv2d(inplanes2, 256, 1, 1, 0)

        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Res_conv3 = nn.Conv2d(inplanes3, 128, 1, 1, 0)

        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Res_conv4 = nn.Conv2d(inplanes4, 64, 1, 1, 0)

        self.upsample5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Res_conv5 = nn.Conv2d(inplanes5, 32, 1, 1, 0)

        self.EMblockout = EMBlock(index=0, in_fea=35, mid_fea=20, out_fea=16)
        self.EMblockout_conv1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.EMblockout_bn1 = nn.BatchNorm2d(16)
        self.EMblockout_conv2 = nn.Conv2d(16, 3, 3, 1, 1)

    def _make_reslayer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def init_weight(self, pretrained=None):
        # super(DehazeNet, self).init_weight(pretrained)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        self.backbone.init_weights(pretrained)

    def forward(self, x):
        dense_out1, dense_out2, dense_out3, dense_out4, dense_out5 = self.backbone(x)
        em1 = self.EMblock1(x)
        casa1 = self.CA1(em1) * em1
        casa1 = self.SA1(casa1) * casa1
        em1 = self.EMdownsample1(em1)
        casa1 = self.CASAdownsample1(casa1)

        em2 = self.EMblock2(torch.cat([dense_out1, em1, casa1], dim=1))
        casa2 = self.CA2(em2) * em2
        casa2 = self.SA2(casa2) * casa2
        em2 = self.EMdownsample2(em2)
        casa2 = self.CASAdownsample2(casa2)

        em3 = self.EMblock3(torch.cat([dense_out2, em2, casa2], dim=1))
        casa3 = self.CA3(em3) * em3
        casa3 = self.SA3(casa3) * casa3
        em3 = self.EMdownsample3(em3)
        casa3 = self.CASAdownsample3(casa3)

        em4 = self.EMblock4(torch.cat([dense_out3, em3, casa3], dim=1))
        casa4 = self.CA4(em4) * em4
        casa4 = self.SA4(casa4) * casa4
        em4 = self.EMdownsample4(em4)
        casa4 = self.CASAdownsample4(casa4)

        em5 = self.EMblock5(torch.cat([dense_out4, em4, casa4], dim=1))
        casa5 = self.CA5(em5) * em5
        casa5 = self.SA5(casa5) * casa5
        em5 = self.EMdownsample5(em5)
        casa5 = self.CASAdownsample5(casa5)

        resout1 = self.Resblock1(torch.cat([dense_out5, em5, casa5], dim=1))
        resout1 = self.Res_conv1(self.upsample1(resout1))

        resout2 = self.Resblock2(torch.cat([dense_out4, em4, casa4, resout1], dim=1))
        resout2 = self.Res_conv2(self.upsample2(resout2))

        resout3 = self.Resblock3(torch.cat([dense_out3, em3, casa3, resout2], dim=1))
        resout3 = self.Res_conv3(self.upsample3(resout3))

        resout4 = self.Resblock4(torch.cat([dense_out2, em2, casa2, resout3], dim=1))
        resout4 = self.Res_conv4(self.upsample4(resout4))

        resout5 = self.Resblock5(torch.cat([dense_out1, em1, casa1, resout4], dim=1))
        resout5 = self.Res_conv5(self.upsample5(resout5))

        out = self.EMblockout(torch.cat([resout5, x], dim=1))
        out = self.EMblockout_conv1(out)
        out = self.EMblockout_bn1(out)
        out = self.relu(out)
        out = self.EMblockout_conv2(out)

        # return F.sigmoid(out)
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class EMBlock(nn.Module):
    def __init__(self, index, in_fea, mid_fea, out_fea):
        super(EMBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_fea, mid_fea, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_fea, mid_fea, 3, 1, 1)
        if index == 16:
            self.avgpool32 = nn.AvgPool2d(16)
        else:
            self.avgpool32 = nn.AvgPool2d(32)
        self.avgpool16 = nn.AvgPool2d(16)
        self.avgpool8 = nn.AvgPool2d(8)
        self.avgpool4 = nn.AvgPool2d(4)

        if index == 16:
            self.upsample32 = nn.UpsamplingNearest2d(scale_factor=16)
        else:
            self.upsample32 = nn.UpsamplingNearest2d(scale_factor=32)
        self.upsample16 = nn.UpsamplingNearest2d(scale_factor=16)
        self.upsample8 = nn.UpsamplingNearest2d(scale_factor=8)
        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=4)

        self.conv3_4 = nn.Conv2d(mid_fea, 1, kernel_size=1, stride=1, padding=0)
        self.conv3_8 = nn.Conv2d(mid_fea, 1, kernel_size=1, stride=1, padding=0)
        self.conv3_16 = nn.Conv2d(mid_fea, 1, kernel_size=1, stride=1, padding=0)
        self.conv3_32 = nn.Conv2d(mid_fea, 1, kernel_size=1, stride=1, padding=0)

        self.conv4 = nn.Conv2d(mid_fea+4, out_fea, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_4 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_8 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_16 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_32 = nn.LeakyReLU(0.2, inplace=True)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out_32 = self.avgpool32(out)
        out_16 = self.avgpool16(out)
        out_8 = self.avgpool8(out)
        out_4 = self.avgpool4(out)

        out_32 =self.upsample32(self.relu3_32(self.conv3_32(out_32)))
        out_16 =self.upsample16(self.relu3_16(self.conv3_16(out_16)))
        out_8 =self.upsample8(self.relu3_8(self.conv3_8(out_8)))
        out_4 = self.upsample4(self.relu3_4(self.conv3_4(out_4)))
        out = torch.cat((out_32, out_16, out_8, out_4, out), dim=1)
        out = self.relu4(self.conv4(out))
        return out


import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import torch
import torch.nn as nn
from Dehaze.configs import Config
from Dehaze.core.Models import build_network


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str, default='/home/dong/python-project/Dehaze/configs/UIEC2Net.py',
                        help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models,')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(args.config)

model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

model = model.cuda()
dummy_input = torch.rand(1, 3, 256, 256).cuda()          # hypothesis the input is b*n*w*h
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input))
