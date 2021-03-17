import torch
import torch.nn as nn
import torch.nn.functional as F
from UW.core.Models.builder import NETWORK
from UW.core.Models.weight_init import normal_init
from UW.utils import get_root_logger, print_log

class BaseNet(nn.Module):
    def __init__(self,
                 backbone,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True,
                 ):
        super(BaseNet, self).__init__()
        pass
        # print()

    def _init_layers(self):
        self.x1 = nn.Conv2d(3,16,3,1,1)
        self.x2 = nn.Conv2d(3,16,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)

    def init_weight(self, pretrained=None):
        if self.backbone and pretrained is not None:
            self.backbone.init_weights(pretrained=pretrained)
            logger = get_root_logger()
            print_log(f'load model from: {pretrained}', logger=logger)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m)

    def get_parameters(self):
        if self.get_parameter:
            total_num = sum(p.numel() for p in self.parameters())
            trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
            self.net_parameters = {'Total': total_num, 'Trainable': trainable_num}
        else:
            self.net_parameters = None

    def forward(self, x):
        pass
