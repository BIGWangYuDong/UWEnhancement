import torch
import torch.nn as nn
import torch.nn.functional as F
from core.Models.builder import NETWORK, build_backbone
from core.Models.base_model import BaseNet
from core.Models.blocks import HSV2RGB, RGB2HSV
from core.Models.weight_init import normal_init, xavier_init

@NETWORK.register_module()
class UIEC2Net(BaseNet):
    def __init__(self,
                 backbone=None,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True):
        super(UIEC2Net, self).__init__(backbone, pretrained, init_weight_type, get_parameter)
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
        self.rgb2hsv = RGB2HSV()
        self.hsv2rgb = HSV2RGB()
        # rgb
        self.norm_batch = nn.InstanceNorm2d  # choose one

        self.rgb_norm_batch1 = self.norm_batch(64)
        self.rgb_con1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch2 = self.norm_batch(64)
        self.rgb_con2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch3 = self.norm_batch(64)
        self.rgb_con3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch4 = self.norm_batch(64)
        self.rgb_con4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch5 = self.norm_batch(64)
        self.rgb_con5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch6 = self.norm_batch(64)
        self.rgb_con6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_con7 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.rgb_fuction_down = nn.LeakyReLU(inplace=True)
        self.rgb_fuction_up = nn.ReLU(inplace=True)

        # hsv
        self.relu = nn.LeakyReLU(inplace=True)
        self.rrelu = nn.ReLU(inplace=True)
        self.M = 11
        # New /1/./2/./3/ use number_f = 32
        number_f = 64
        self.e_conv1 = nn.Conv2d(6, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_convfc = nn.Linear(number_f, 44)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.avagepool = nn.AdaptiveAvgPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # confidence
        self.norm_batch = nn.InstanceNorm2d  # choose one

        self.norm_batch1 = self.norm_batch(64)
        self.con1 = nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch2 = self.norm_batch(64)
        self.con2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch3 = self.norm_batch(64)
        self.con3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch4 = self.norm_batch(64)
        self.con4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch5 = self.norm_batch(64)
        self.con5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch6 = self.norm_batch(64)
        self.con6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.con7 = nn.Conv2d(64, 6, kernel_size=1, stride=1, padding=0)

        self.fuction_down = nn.LeakyReLU(inplace=True)
        self.fuction_up = nn.ReLU(inplace=True)


    def init_weight(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        if self.backbone is not None:
            self.backbone.init_weights(pretrained)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        h = self.rgb_fuction_down(self.rgb_norm_batch1(self.rgb_con1(x)))
        h = self.rgb_fuction_down(self.rgb_norm_batch2(self.rgb_con2(h)))
        h = self.rgb_fuction_down(self.rgb_norm_batch3(self.rgb_con3(h)))
        h = self.rgb_fuction_down(self.rgb_norm_batch4(self.rgb_con4(h)))
        h = self.rgb_fuction_up(self.rgb_norm_batch5(self.rgb_con5(h)))
        h = self.rgb_fuction_up(self.rgb_norm_batch6(self.rgb_con6(h)))  # try to use
        rgb_out = F.sigmoid(self.rgb_con7(h))
        rgb_out = rgb_out[:, 0:3, :, :]
        hsv_fromrgbout = self.rgb2hsv(rgb_out)
        hsv_frominput = self.rgb2hsv(x)

        hsv_input = torch.cat([hsv_fromrgbout, hsv_fromrgbout], dim=1)
        batch_size = hsv_input.size()[0]
        x1 = self.relu(self.e_conv1(hsv_input))
        x1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        x2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        x3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))
        x_r = self.relu(self.e_conv7(x4))
        x_r = self.avagepool(x_r).view(batch_size, -1)
        x_r = self.e_convfc(x_r)
        H, S, V, H2S = torch.split(x_r, self.M, dim=1)
        H_in, S_in, V_in = hsv_input[:, 0:1, :, :], hsv_input[:, 1:2, :, :], hsv_input[:, 2:3, :, :],
        H_out = piece_function_org(H_in, H, self.M)
        S_out1 = piece_function_org(S_in, S, self.M)
        V_out = piece_function_org(V_in, V, self.M)

        S_out2 = piece_function_org(H_in, H2S, self.M)
        S_out = (S_out1 + S_out2) / 2

        zero_lab = torch.zeros(S_out.shape).cuda()
        s_t = torch.where(S_out < 0, zero_lab, S_out)
        one_lab = torch.ones(S_out.shape).cuda()
        S_out = torch.where(s_t > 1, one_lab, s_t)

        zero_lab = torch.zeros(V_out.shape).cuda()
        s_t = torch.where(V_out < 0, zero_lab, V_out)
        one_lab = torch.ones(V_out.shape).cuda()
        V_out = torch.where(s_t > 1, one_lab, s_t)

        hsv_out = torch.cat([H_out, S_out, V_out], dim=1)
        curve = torch.cat([H.view(batch_size, 1, -1),
                           S.view(batch_size, 1, -1),
                           V.view(batch_size, 1, -1),
                           H2S.view(batch_size, 1, -1)], dim=1)

        hsv_out_rgb = self.hsv2rgb(hsv_out)

        confindencenet_input = torch.cat([x,
                                          rgb_out,
                                          hsv_out_rgb], dim=1)

        h = self.fuction_down(self.norm_batch1(self.con1(confindencenet_input)))
        h = self.fuction_down(self.norm_batch2(self.con2(h)))
        h = self.fuction_down(self.norm_batch3(self.con3(h)))
        h = self.fuction_down(self.norm_batch4(self.con4(h)))
        h = self.fuction_down(self.norm_batch5(self.con5(h)))
        h = self.fuction_down(self.norm_batch6(self.con6(h)))  # try to use
        confindence_out = F.sigmoid(self.con7(h))

        # 需要改名
        confindence_rgb = confindence_out[:, 0:3, :, :]
        confindence_hsv = confindence_out[:, 3:6, :, :]
        output_useconf = 0.5 * confindence_rgb * rgb_out + \
                              0.5 * confindence_hsv * hsv_out_rgb

        return output_useconf
               #, rgb_out, hsv_out_rgb



def piece_function_org(x_m, para_m, M):
    b, c, w, h = x_m.shape
    r_m = para_m[:, 0].view(b, c, 1, 1).expand(b, c, w, h)
    for i in range(M-1):
        para = (para_m[:, i + 1] - para_m[:, i]).view(b, c, 1, 1).expand(b, c, w, h)
        r_m = r_m + para * \
              sgn_m(M * x_m - i * torch.ones(x_m.shape).cuda())
    return r_m

def sgn_m(x):
    # x = torch.Tensor(x)
    zero_lab = torch.zeros(x.shape).cuda()
    # print("one_lab",one_lab)
    s_t = torch.where(x < 0, zero_lab, x)
    one_lab = torch.ones(x.shape).cuda()
    s = torch.where(s_t > 1, one_lab, s_t)
    return s