import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
from UW.core.Models import BACKBONES
from UW.core.Models.weight_init import normal_init

@BACKBONES.register_module()
class DenseBlock(nn.Module):
    def __init__(self, pretrained):
        super(DenseBlock, self).__init__()
        '''
        check the network
        '''
        # densenet = models.densenet121(pretrained=pretrained)  # 121, 161, 169, 201
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                padding=1, bias=True)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True))
        ]))

        self.denseblock1_pooling = nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(256)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
            ]))
        # self.trans1 = nn.Conv2d()

        self.denseblock2_pooling = nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(512)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))


        self.denseblock3_pooling = nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(1024)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.denseblock4_pooling = nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(1024)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))
        self.denseblock5_pooling = nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(1024)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False)),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m)
        densenet = models.densenet121(pretrained=pretrained)
        self.denseblock1 = densenet.features.denseblock1  # 6
        self.denseblock2 = densenet.features.denseblock2  # 12
        self.denseblock3 = densenet.features.denseblock3  # 24
        self.denseblock4 = densenet.features.denseblock3  # 24
        self.denseblock5 = densenet.features.denseblock4  # 16


    def forward(self, x):
        '''
        forward
        '''
        out1 = self.features(x)
        out1 = self.denseblock1(out1)
        out1 = self.denseblock1_pooling(out1)
        out2 = self.denseblock2(out1)
        out2 = self.denseblock2_pooling(out2)
        out3 = self.denseblock3(out2)
        out3 = self.denseblock3_pooling(out3)
        out4 = self.denseblock4(out3)
        out4 = self.denseblock4_pooling(out4)
        out5 = self.denseblock5(out4)
        out5 = self.denseblock5_pooling(out5)

        return out1, out2, out3, out4, out5