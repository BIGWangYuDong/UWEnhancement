import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
from core.Models import BACKBONES
from core.Models.weight_init import normal_init


@BACKBONES.register_module()
class DenseNew(nn.Module):
    def __init__(self, pretrained):
        super(DenseNew, self).__init__()
        '''
        check the network
        '''
        # densenet = models.densenet121(pretrained=pretrained)  # 121, 161, 169, 201
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained):
        densenet = models.densenet121(pretrained=pretrained)
        self.block1 = nn.Sequential(OrderedDict([
            ('conv1', densenet.features[0]),
            ('norm1', densenet.features[1]),
            ('relu1', densenet.features[2])]))
        self.block2 = nn.Sequential(OrderedDict([
            ('pool2', densenet.features[3]),
            ('block2', densenet.features[4]),
            ('norm2', densenet.features[5].norm),
            ('relu2', densenet.features[5].relu),
            ('conv2', densenet.features[5].conv)]))
        self.block3 = nn.Sequential(OrderedDict([
            ('pool3', densenet.features[5].pool),
            ('block3', densenet.features[6]),
            ('norm3', densenet.features[7].norm),
            ('relu3', densenet.features[7].relu),
            ('conv3', densenet.features[7].conv)]))
        self.block4 = nn.Sequential(OrderedDict([
            ('pool4', densenet.features[7].pool),
            ('block4', densenet.features[8]),
            ('norm4', densenet.features[9].norm),
            ('relu4', densenet.features[9].relu),
            ('conv4', densenet.features[9].conv)]))
        self.block5 = nn.Sequential(OrderedDict([
            ('pool5', densenet.features[9].pool),
            ('block5', densenet.features[10]),
            ('norm5', densenet.features[11])]))

    def forward(self, x):
        '''
        forward
        '''
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)

        return out1, out2, out3, out4, out5