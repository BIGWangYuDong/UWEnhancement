import torch
import torch.nn as nn
import torch.nn.functional as F
from core.Models.builder import NETWORK, build_backbone
from core.Models.base_model import BaseNet
from core.Models.weight_init import normal_init, xavier_init


@NETWORK.register_module()
class UWGANUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1_conv_1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.layer1_conv_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.layer1_pooling = nn.MaxPool2d(2)

        self.layer2_conv_1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.layer2_conv_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer2_pooling = nn.MaxPool2d(2)

        self.layer3_conv_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.layer3_conv_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.layer3_pooling = nn.MaxPool2d(2)

        self.layer4_conv_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.layer4_conv_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer4_up = nn.ConvTranspose2d(256, 128, 2, 2)

        self.layer5_conv_1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.layer5_conv_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.layer5_up = nn.ConvTranspose2d(128, 64, 2, 2)

        self.layer6_conv_1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.layer6_conv_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer6_up = nn.ConvTranspose2d(64, 32, 2, 2)

        self.layer7_conv_1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.layer7_conv_2 = nn.Conv2d(32, 32, 3, 1, 1)

        self.out = nn.Conv2d(32, 3, 1, 1, 0)

    def forward(self, inputs):
        layer1_out = F.relu(self.layer1_conv_1(inputs))
        layer1_out = F.relu(self.layer1_conv_2(layer1_out))

        layer2_out = self.layer1_pooling(layer1_out)
        layer2_out = F.relu(self.layer2_conv_1(layer2_out))
        layer2_out = F.relu(self.layer2_conv_2(layer2_out))

        layer3_out = self.layer2_pooling(layer2_out)
        layer3_out = F.relu(self.layer3_conv_1(layer3_out))
        layer3_out = F.relu(self.layer3_conv_2(layer3_out))

        layer4_out = self.layer3_pooling(layer3_out)
        layer4_out = F.relu(self.layer4_conv_1(layer4_out))
        layer4_out = F.relu(self.layer4_conv_2(layer4_out))

        layer5_out = self.layer4_up(layer4_out)
        layer5_out = torch.cat([layer5_out, layer3_out], dim=1)
        layer5_out = F.relu(self.layer5_conv_1(layer5_out))
        layer5_out = F.relu(self.layer5_conv_2(layer5_out))

        layer6_out = self.layer5_up(layer5_out)
        layer6_out = torch.cat([layer6_out, layer2_out], dim=1)
        layer6_out = F.relu(self.layer6_conv_1(layer6_out))
        layer6_out = F.relu(self.layer6_conv_2(layer6_out))

        layer7_out = self.layer6_up(layer6_out)
        layer7_out = torch.cat([layer7_out, layer1_out], dim=1)
        layer7_out = F.relu(self.layer7_conv_1(layer7_out))
        layer7_out = F.relu(self.layer7_conv_2(layer7_out))

        out = torch.tanh(self.out(layer7_out))

        return out
