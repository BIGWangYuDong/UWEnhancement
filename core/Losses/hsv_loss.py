import torch
import torch.nn as nn
import math
from UW.core.Losses.builder import LOSSES


@LOSSES.register_module()
class HSVLoss(nn.Module):
    def __init__(self):
        super(HSVLoss, self).__init__()
        self.criterionloss = nn.L1Loss()
        self.pi = math.pi

    def forward(self, pred, gt):
        # attention if hp between 0,1 , hp *= 360
        # gt.type(torch.cuda.FloatTensor)
        # hp, sp, vp = pred[:, 0, :, :], pred[:, 1, :, :], pred[:, 2, :, :]
        # hg, sg, vg = gt[:, 0, :, :], gt[:, 1, :, :], gt[:, 2, :, :]

        hi, si, vi = pred[:, 0:1, :, :], pred[:, 1:2, :, :], pred[:, 2:, :, :]
        hj, sj, vj = gt[:, 0:1, :, :], gt[:, 1:2, :, :], gt[:, 2:, :, :]

        hipi = hi * self.pi * 2
        hjpi = hj * self.pi * 2

        coshp = torch.cos(hipi)
        sinhg = torch.sin(hjpi)
        sv_p = torch.mul(si, vi)
        sv_g = torch.mul(sj, vj)
        temp_pred = torch.mul(coshp, sv_p)
        temp_gt = torch.mul(sinhg, sv_g)
        loss = torch.mean(torch.abs(torch.add(temp_pred, -1, temp_gt)))
        
        return loss
