import torch
import torch.nn as nn
import torch.nn.functional as F
from core.Models.builder import NETWORK, build_backbone
from core.Models.base_model import BaseNet
from core.Models.backbone import DenseNew
from core.Models.backbone.resnet import Bottleneck
from core.Models.weight_init import normal_init, xavier_init


@NETWORK.register_module()
class DCPDehazeSimple(BaseNet):
    def __init__(self,
                 backbone,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True):
        super(DCPDehazeSimple, self).__init__(backbone, pretrained, init_weight_type, get_parameter)
        self.backbone = build_backbone(backbone)
        if init_weight_type is not None:
            self.init_weight_type = init_weight_type
        self.get_parameter = get_parameter
        self._init_layers()
        self.init_weight(pretrained=pretrained)
        self.get_parameters()

    def _init_layers(self):
        self.Resblock1 = UpBlock(1024, 512, 256)
        self.Resblock2 = UpBlock(512+256, 256, 128)
        self.Resblock3 = UpBlock(256+128, 128, 256)
        self.Resblock4 = UpBlock(256, 64, 128)
        self.Resblock5 = UpBlock(128, 32, 32)

        self.upsample = F.upsample_nearest
        self.upsample2 = F.upsample_nearest
        self.upsample3 = F.upsample_nearest
        self.upsample4 = F.upsample_nearest
        self.upsample5 = F.upsample_nearest

        self.instance1 = nn.InstanceNorm2d(512, affine=False)
        self.instance2 = nn.InstanceNorm2d(256, affine=False)

        self.DCP = DarkChannelPrior(windowsize=5)
        self.SFT_conv1 = nn.Conv2d(4, 32, 3, 1, 1)
        self.SFT_bn1 = nn.BatchNorm2d(32)
        self.SFTblock = nn.Sequential(IMDModule(32),
                                      IMDModule(32),
                                      IMDModule(32),
                                      IMDModule(32),
                                      IMDModule(32))

        self.SFT_scale_conv1 = nn.Conv2d(32, 32, 1, 1, 0)
        self.SFT_scale_bn1 = nn.BatchNorm2d(32)
        self.SFT_scale_conv2 = nn.Conv2d(32, 32, 1, 1, 0)

        self.SFT_shift_conv1 = nn.Conv2d(32, 32, 1, 1, 0)
        self.SFT_shift_bn1 = nn.BatchNorm2d(32)
        self.SFT_shift_conv2 = nn.Conv2d(32, 32, 1, 1, 0)
        self.SFT_relu = nn.LeakyReLU(0.1, inplace=True)

        # for p in self.parameters():
        #     p.requires_grad = False

        self.SA_dehaze = SpatialAttention()
        self.CA_dehaze = ChannelAttention(32)

        # DR
        self.DRHead = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32)
        )
        self.DRblock = DRBlock(32, 32, 64)
        self.DR1 = IMDModule(64)
        self.DR2 = IMDModule(64)
        self.DR3 = IMDModule(64)
        self.DRTail = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        )
        self.SA_DR = SpatialAttention()
        self.CA_DR = ChannelAttention(32)

        self.concat_org = nn.Conv2d(64, 64, 1, 1, 0)
        self.concat_CASA = nn.Conv2d(64, 64, 1, 1, 0)

        self.SA_tail = SpatialAttention()
        self.CA_tail = ChannelAttention(64)

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        # self.EMblockout = EMBlock(index=0, in_fea=35, mid_fea=20, out_fea=16)
        self.EMblockout_conv1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.EMblockout_bn1 = nn.BatchNorm2d(32)
        self.EMblockout_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.EMblockout_bn2 = nn.BatchNorm2d(32)
        self.EMblockout_conv3 = nn.Conv2d(32, 3, 3, 1, 1)

    def init_weight(self, pretrained=None):
        # super(DehazeNet, self).init_weight(pretrained)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        self.backbone.init_weights(pretrained)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        dense_out1, dense_out2, dense_out3, dense_out4, dense_out5 = self.backbone(x)

        shape_out4 = dense_out4.data.size()
        shape_out4 = shape_out4[2:4]

        shape_out3 = dense_out3.data.size()
        shape_out3 = shape_out3[2:4]

        shape_out2 = dense_out2.data.size()
        shape_out2 = shape_out2[2:4]

        shape_out1 = dense_out1.data.size()
        shape_out1 = shape_out1[2:4]

        shape_out = x.data.size()
        shape_out = shape_out[2:4]

        up_4 = self.Resblock1(dense_out5)
        up_4 = self.upsample(up_4, size=shape_out4)
        dense_out4 = self.instance1(dense_out4)

        up_3 = self.Resblock2(torch.cat([up_4, dense_out4], 1))
        up_3 = self.upsample(up_3, size=shape_out3)
        dense_out3 = self.instance2(dense_out3)

        up_2 = self.Resblock3(torch.cat([up_3, dense_out3], 1))
        up_2 = self.upsample(up_2, size=shape_out2)

        up_1 = self.Resblock4(up_2)
        up_1 = self.upsample(up_1, size=shape_out1)

        resout5 = self.Resblock5(up_1)
        resout5 = self.upsample(resout5, size=shape_out)
        # resout5 = F.tanh(resout5)

        t = self.DCP(x).detach()
        t = self.SFT_relu(self.SFT_bn1(self.SFT_conv1(torch.cat([t,x],dim=1))))
        t = self.SFTblock(t)
        scale = self.SFT_scale_conv1(t)
        scale = self.SFT_scale_bn1(scale)
        scale = self.SFT_relu(scale)
        scale = self.SFT_scale_conv2(scale)
        # scale = F.tanh(scale)

        shift = self.SFT_shift_conv1(t)
        shift = self.SFT_shift_bn1(shift)
        shift = self.SFT_relu(shift)
        shift = self.SFT_shift_conv2(shift)
        # shift = F.tanh(shift)

        dehaze_out = resout5 * scale + shift

        casa_dehaze_out = self.CA_dehaze(dehaze_out) * dehaze_out
        casa_dehaze_out = self.SA_dehaze(casa_dehaze_out) * casa_dehaze_out

        DR_out = self.DRHead(x)
        DR_out = self.DRblock(DR_out)
        DR_out = self.DR3(self.DR2(self.DR1(DR_out)))
        DR_out = self.DRTail(DR_out)

        casa_DR_out = self.CA_DR(DR_out) * DR_out
        casa_DR_out = self.SA_DR(casa_DR_out) * casa_DR_out

        out = self.concat_org(torch.cat([dehaze_out, DR_out], dim=1))
        casa_out = self.concat_CASA(torch.cat([casa_dehaze_out, casa_DR_out], dim=1))
        casa_out = self.CA_tail(casa_out) * casa_out
        casa_out = self.SA_tail(casa_out) * casa_out

        out = casa_out + out

        # out = self.EMblockout(torch.cat([resout5, x], dim=1))
        # out = self.EMblockout_conv1(torch.cat([resout5, x], dim=1))
        out = self.EMblockout_conv1(out)
        out = self.EMblockout_bn1(out)
        out = self.relu(out)
        out = self.EMblockout_conv2(out)
        out = self.EMblockout_bn2(out)
        out = self.relu(out)
        out = self.EMblockout_conv3(out)

        # out = self.relu(out)
        return F.tanh(out)
        # return out

class UpBlock(nn.Module):
    def __init__(self, inchannel, midchannel, outchannel):
        super(UpBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, midchannel, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannel, midchannel, 3, 1, 1)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(inchannel+midchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel+midchannel, outchannel, 1, 1, 0)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(torch.cat([x, out], dim=1))
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
    def __init__(self, in_fea, mid_fea, out_fea):
        super(EMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_fea, mid_fea, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(mid_fea)
        self.conv2 = nn.Conv2d(mid_fea, mid_fea, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(mid_fea)
        self.avgpool32 = nn.AvgPool2d(16)
        self.avgpool16 = nn.AvgPool2d(8)
        self.avgpool8 = nn.AvgPool2d(4)
        self.avgpool4 = nn.AvgPool2d(2)

        self.upsample = F.upsample_nearest

        self.conv3_4 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.bn3_4 = nn.BatchNorm2d(int(mid_fea / 4))
        self.conv3_8 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.bn3_8 = nn.BatchNorm2d(int(mid_fea / 4))
        self.conv3_16 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.bn3_16 = nn.BatchNorm2d(int(mid_fea / 4))
        self.conv3_32 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.bn3_32 = nn.BatchNorm2d(int(mid_fea / 4))

        self.conv4 = nn.Conv2d(mid_fea + mid_fea, out_fea, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_4 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_8 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_16 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_32 = nn.LeakyReLU(0.2, inplace=True)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        shape_out = out.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        out_32 = self.avgpool32(out)
        out_16 = self.avgpool16(out)
        out_8 = self.avgpool8(out)
        out_4 = self.avgpool4(out)

        out_32 = self.upsample(self.relu3_32(self.bn3_32(self.conv3_32(out_32))), size=shape_out)
        out_16 = self.upsample(self.relu3_16(self.bn3_16(self.conv3_16(out_16))), size=shape_out)
        out_8 = self.upsample(self.relu3_8(self.bn3_32(self.conv3_8(out_8))), size=shape_out)
        out_4 = self.upsample(self.relu3_4(self.bn3_16(self.conv3_4(out_4))), size=shape_out)
        out = torch.cat((out_32, out_16, out_8, out_4, out), dim=1)
        out = self.relu4(self.conv4(out))
        return out

class DarkChannelPrior(nn.Module):
    def __init__(self, windowsize=5):
        super(DarkChannelPrior, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=windowsize, stride=1, padding=windowsize//2)

    def forward(self, x):
        x.requires_grad = False
        x = (x + 1) / 2 * 255
        min_rgb, _ = torch.min(x, dim=1)
        min_rgb = min_rgb.unsqueeze(dim=1)
        min_rgb = 1 - min_rgb
        dark_channel = self.maxpooling(min_rgb)
        dark_channel = 1 - dark_channel
        darkchannelmax = torch.max(dark_channel)
        darkchannel3 = torch.cat([dark_channel, dark_channel, dark_channel], dim=1)
        mask = darkchannel3 >= (0.95 * darkchannelmax)

        A = torch.zeros_like(darkchannel3)
        if torch.cuda.is_available():
            A.cuda()
        A[mask] = x[mask]
        A = A.max()
        # t = 1 - 0.95 * (dark_channel / A)
        t = dark_channel / A
        # Dark Channel Prior
        # mask = t >= 0.1
        # T = torch.zeros_like(t)
        # T[mask] = t[mask]
        # image = (x - A) / T + A
        # image = ((image / 255) * 2) - 1
        return t


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=4):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels // distillation_rate)  # 32/3 = 8
        self.remaining_channels_3 = int(in_channels - self.distilled_channels)  # 32-8 = 24
        self.remaining_channels_2 = int(self.remaining_channels_3 - self.distilled_channels)  # 24-8=16
        self.remaining_channels_1 = int(self.remaining_channels_2 - self.distilled_channels)  # 16-8=8

        self.conv0 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(self.remaining_channels_3, self.remaining_channels_3, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.remaining_channels_3)

        self.conv2 = nn.Conv2d(self.remaining_channels_2, self.remaining_channels_2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.remaining_channels_2)

        self.conv3 = nn.Conv2d(self.remaining_channels_1, self.remaining_channels_1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(self.remaining_channels_1)

        self.conv4 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.relu(self.bn0(self.conv0(x)))

        distilled_c1, remaining_c1 = torch.split(out, (self.distilled_channels, self.remaining_channels_3), dim=1)

        out_c2 = self.relu(self.bn1(self.conv1(remaining_c1)))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels_2), dim=1)

        out_c3 = self.relu(self.bn2(self.conv2(remaining_c2)))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels_1), dim=1)

        out_c4 = self.relu(self.bn3(self.conv3(remaining_c3)))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out = self.relu(self.conv4(out)) + x

        return out

class DRBlock(nn.Module):
    def __init__(self, in_fea, mid_fea, out_fea):
        super(DRBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_fea, mid_fea, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(mid_fea)
        self.conv2 = nn.Conv2d(mid_fea, mid_fea, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(mid_fea)
        self.avgpool32 = nn.AvgPool2d(32)
        self.avgpool16 = nn.AvgPool2d(16)
        self.avgpool8 = nn.AvgPool2d(8)
        self.avgpool4 = nn.AvgPool2d(4)
        self.avgpool2 = nn.AvgPool2d(2)

        self.upsample = F.upsample_nearest

        self.conv3_4 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.conv3_8 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.conv3_16 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.conv3_32 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)

        self.bn3_4 = nn.BatchNorm2d(int(mid_fea / 4))
        self.bn3_8 = nn.BatchNorm2d(int(mid_fea / 4))
        self.bn3_16 = nn.BatchNorm2d(int(mid_fea / 4))
        self.bn3_32 = nn.BatchNorm2d(int(mid_fea / 4))

        self.conv4 = nn.Conv2d(mid_fea + mid_fea, out_fea, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_4 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_8 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_16 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_32 = nn.LeakyReLU(0.2, inplace=True)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out_2 = self.avgpool2(out)
        shape_out = out_2.data.size()
        shape_out = shape_out[2:4]
        out_32 = self.avgpool32(out)
        out_16 = self.avgpool16(out)
        out_8 = self.avgpool8(out)
        out_4 = self.avgpool4(out)

        out_32 = self.upsample(self.relu3_32(self.bn3_32(self.conv3_32(out_32))), size=shape_out)
        out_16 = self.upsample(self.relu3_16(self.bn3_16(self.conv3_16(out_16))), size=shape_out)
        out_8 = self.upsample(self.relu3_8(self.bn3_8(self.conv3_8(out_8))), size=shape_out)
        out_4 = self.upsample(self.relu3_4(self.bn3_4(self.conv3_4(out_4))), size=shape_out)
        out = torch.cat((out_32, out_16, out_8, out_4, out_2), dim=1)
        out = self.relu4(self.conv4(out))
        return out