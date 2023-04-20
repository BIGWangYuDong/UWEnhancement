import torch
import torch.nn as nn
from core.Losses.builder import LOSSES


@LOSSES.register_module()
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, img1, img2):
        zeros = torch.zeros(img1.size()).cuda(img1.device)
        loss = nn.L1Loss(size_average=True)(torch.fft(torch.stack((img1, zeros), -1), 2),
                                            torch.fft(torch.stack((img2, zeros), -1), 2))
        loss = self.loss_weight * loss
        return loss