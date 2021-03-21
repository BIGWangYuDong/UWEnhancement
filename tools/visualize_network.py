import torch
import torch.nn as nn
import torch.nn.functional as F
from UW.core.Models.builder import NETWORK, build_backbone
from UW.core.Models.base_model import BaseNet
from UW.core.Models.weight_init import normal_init, xavier_init
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from UW.core.Models import build_network
from tensorboardX import SummaryWriter


@NETWORK.register_module()
class Net(BaseNet):
    def __init__(self,
                 backbone=None,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True):
        super(Net, self).__init__(backbone, pretrained, init_weight_type, get_parameter)
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
        '''
        init layers
        e.g. self.conv1 = nn.Conv2d(3,16,3,1,1)
        '''
        self.conv1 = nn.Conv2d(3,16,3,1,1)

    def init_weight(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        if self.backbone is not None:
            self.backbone.init_weights(pretrained)

    def forward(self, x):
        out = self.conv1(x)
        return out

if __name__ == '__main__':
    writer = SummaryWriter('log')
    cfg_model = dict(type='Net',
                 get_parameter=True)                        # add parameters if need

    model = build_network(cfg_model)

    model = model.cuda()
    dummy_input = torch.rand(1, 3, 256, 256).cuda()          # hypothesis the input is b*n*w*h
    with SummaryWriter(comment='LeNet') as w:
        w.add_graph(model, (dummy_input))
