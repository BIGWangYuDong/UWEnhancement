import torch
import torch.nn as nn
import torch.nn.functional as F
from core.Models.builder import NETWORK, build_backbone
from core.Models.base_model import BaseNet
from core.Models.weight_init import normal_init, xavier_init

@NETWORK.register_module()
class WaterNet(BaseNet):
    def __init__(self,
                 backbone=None,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True):
        super(WaterNet, self).__init__(backbone, pretrained, init_weight_type, get_parameter)
        if backbone is not None:
            self.backbone = build_backbone(backbone)
        else:
            self.backbone = None
        if init_weight_type is not None:
            self.init_weight_type = init_weight_type
        self.get_parameter = get_parameter
        self._init_layers()
        self.init_weight(pretrained=pretrained)
        self.get_parameters()

    def _init_layers(self):
        self.conv2wb_1 = nn.Conv2d(12, 128, 7, 1, 3)
        self.conv2wb_1_relu = nn.ReLU(inplace=True)

        self.conv2wb_2 = nn.Conv2d(128, 128, 5, 1, 2)
        self.conv2wb_2_relu = nn.ReLU(inplace=True)

        self.conv2wb_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2wb_3_relu = nn.ReLU(inplace=True)

        self.conv2wb_4 = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv2wb_4_relu = nn.ReLU(inplace=True)

        self.conv2wb_5 = nn.Conv2d(64, 64, 7, 1, 3)
        self.conv2wb_5_relu = nn.ReLU(inplace=True)

        self.conv2wb_6 = nn.Conv2d(64, 64, 5, 1, 2)
        self.conv2wb_6_relu = nn.ReLU(inplace=True)

        self.conv2wb_7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2wb_7_relu = nn.ReLU(inplace=True)

        self.conv2wb_77 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv2wb_77_sigmoid = nn.Sigmoid()

        # wb
        self.conv2wb_9 = nn.Conv2d(6, 32, 7, 1, 3)
        self.conv2wb_9_relu = nn.ReLU(inplace=True)

        self.conv2wb_10 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv2wb_10_relu = nn.ReLU(inplace=True)

        self.conv2wb_11 = nn.Conv2d(32, 3, 3, 1, 1)
        self.conv2wb_11_relu = nn.ReLU(inplace=True)

        # ce
        self.conv2wb_99 = nn.Conv2d(6, 32, 7, 1, 3)
        self.conv2wb_99_relu = nn.ReLU(inplace=True)

        self.conv2wb_100 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv2wb_100_relu = nn.ReLU(inplace=True)

        self.conv2wb_111 = nn.Conv2d(32, 3, 3, 1, 1)
        self.conv2wb_111_relu = nn.ReLU(inplace=True)

        # gc
        self.conv2wb_999 = nn.Conv2d(6, 32, 7, 1, 3)
        self.conv2wb_999_relu = nn.ReLU(inplace=True)

        self.conv2wb_1000 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv2wb_1000_relu = nn.ReLU(inplace=True)

        self.conv2wb_1111 = nn.Conv2d(32, 3, 3, 1, 1)
        self.conv2wb_1111_relu = nn.ReLU(inplace=True)

    def init_weight(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        if self.backbone is not None:
            self.backbone.init_weights(pretrained)

    def forward(self, x, wb, ce, gc):
        conb0 = torch.cat([x, wb, ce, gc], dim=1)
        conv_wb1 = self.conv2wb_1_relu(self.conv2wb_1(conb0))
        conv_wb2 = self.conv2wb_2_relu(self.conv2wb_2(conv_wb1))
        conv_wb3 = self.conv2wb_3_relu(self.conv2wb_3(conv_wb2))
        conv_wb4 = self.conv2wb_4_relu(self.conv2wb_4(conv_wb3))
        conv_wb5 = self.conv2wb_5_relu(self.conv2wb_5(conv_wb4))
        conv_wb6 = self.conv2wb_6_relu(self.conv2wb_6(conv_wb5))
        conv_wb7 = self.conv2wb_7_relu(self.conv2wb_7(conv_wb6))
        conv_wb77 = self.conv2wb_77_sigmoid(self.conv2wb_77(conv_wb7))

        # wb
        conb00 = torch.cat([x, wb], dim=1)
        conv_wb9 = self.conv2wb_9_relu(self.conv2wb_9(conb00))
        conv_wb10 = self.conv2wb_10_relu(self.conv2wb_10(conv_wb9))
        wb1 = self.conv2wb_11_relu(self.conv2wb_11(conv_wb10))

        # ce
        conb11 = torch.cat([x, ce], dim=1)
        conv_wb99 = self.conv2wb_99_relu(self.conv2wb_99(conb11))
        conv_wb100 = self.conv2wb_100_relu(self.conv2wb_100(conv_wb99))
        ce1 = self.conv2wb_111_relu(self.conv2wb_111(conv_wb100))

        # gc
        conb111 = torch.cat([x, gc], dim=1)
        conv_wb999 = self.conv2wb_999_relu(self.conv2wb_999(conb111))
        conv_wb1000 = self.conv2wb_1000_relu(self.conv2wb_1000(conv_wb999))
        gc1 = self.conv2wb_1111_relu(self.conv2wb_1111(conv_wb1000))

        weight_wb, weight_ce, weight_gc = conv_wb77[:, 0:1, :, :], conv_wb77[:, 1:2, :, :], conv_wb77[:, 2:3, :, :]
        out = (weight_wb * wb1) + (weight_ce * ce1) + (weight_gc * gc1)

        return out
