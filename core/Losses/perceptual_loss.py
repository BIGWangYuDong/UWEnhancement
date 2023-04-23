import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from torch.autograd import Variable
from core.Losses.builder import LOSSES
from core.Losses.utils import weighted_loss


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, vgg_choose='conv4_3', vgg_maxpooling='False'):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        if vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        if vgg_choose != "no_maxpool":
            if vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2)
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        if vgg_choose == "conv4_3":
            return conv4_3
        elif vgg_choose == "relu4_2":
            return relu4_2
        elif vgg_choose == "relu4_1":
            return relu4_1
        elif vgg_choose == "relu4_3":
            return relu4_3
        elif vgg_choose == "conv5_3":
            return conv5_3
        elif vgg_choose == "relu5_1":
            return relu5_1
        elif vgg_choose == "relu5_2":
            return relu5_2
        elif vgg_choose == "relu5_3" or "maxpool":
            return relu5_3

def vgg_preprocess(batch, vgg_mean=False):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    if vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(Variable(mean))  # subtract mean
    return batch


def load_vgg16(model_dir, cfg):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir,
                                                                                                        'vgg16.t7'))
        vgglua = torch.load(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    # vgg.cuda()
    if torch.cuda.is_available():
        vgg.cuda(device=cfg.gpu_ids[0])
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    # vgg = torch.nn.DataParallel(vgg, gpu_ids)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

@LOSSES.register_module()
class PerceptualLoss(nn.Module):
    '''
    parser.add_argument('--vgg', type=float, default=1, help='use perceptrual loss')
    parser.add_argument('--vgg_mean', action='store_true', help='substract mean in vgg loss')
    parser.add_argument('--vgg_choose', type=str, default='conv4_3', help='choose layer for vgg')
    # parser.add_argument('--vgg_choose', type=str, default='relu5_3', help='choose layer for vgg')

    parser.add_argument('--no_vgg_instance', action='store_true', help='vgg instance normalization')
    parser.add_argument('--vgg_maxpooling', action='store_true', help='normalize attention map')
    parser.add_argument('--IN_vgg', action='store_true', help='patch vgg individual')
    # criterionPerceptual = PerceptualLoss(opt).to(self.device)
    # vgg = load_vgg16("./vgg", gpu_ids)
    # vgg.eval()
    # for param in vgg.parameters():
    #     param.requires_grad = False
    # loss_perc = criterionPerceptual(vgg, pred, target)
    '''
    def __init__(self,
                 no_vgg_instance=False,
                 vgg_mean=False,
                 vgg_choose='conv4_3',
                 vgg_maxpooling=False,
                 loss_weight=1.0):
        super(PerceptualLoss, self).__init__()
        self.no_vgg_instance = no_vgg_instance
        self.vgg_mean = vgg_mean
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.loss_weight = loss_weight
        self.vgg_choose =vgg_choose
        self.vgg_maxpooling =vgg_maxpooling

    def forward(self, img, target, cfg):
        self.vgg = load_vgg16("./core/Losses/vgg", cfg)
        img_vgg = vgg_preprocess(img, self.vgg_mean)
        target_vgg = vgg_preprocess(target, self.vgg_mean)
        img_fea = self.vgg(img_vgg, self.vgg_choose, self.vgg_maxpooling)
        target_fea = self.vgg(target_vgg, self.vgg_choose, self.vgg_maxpooling)
        if self.no_vgg_instance:
            return torch.mean((img_fea - target_fea) ** 2) * self.loss_weight
        else:
            return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2) * self.loss_weight

