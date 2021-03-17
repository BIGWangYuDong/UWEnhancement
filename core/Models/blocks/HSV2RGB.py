import torch
import torch.nn as nn


def function_delta(input):
    out = torch.clamp(input, min=0, max=60)
    return out


class HSV2RGB(nn.Module):
    def __init__(self):
        super(HSV2RGB, self).__init__()

    def forward(self, hsv):
        batch, c, w, height = hsv.size()
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        htemp = (h * 360) % 360
        h = htemp / 360
        # h = h / 360
        vs = torch.div(torch.mul(v, s), 60)  # (v * s) / 60
        R1_delta = function_delta(torch.add(torch.mul(h, 360), -60))  # delta(360h - 60)
        R2_delta = function_delta(torch.add(torch.mul(h, 360), -240))

        G1_delta = function_delta(torch.add(torch.mul(h, 360), 0))
        G2_delta = function_delta(torch.add(torch.mul(h, 360), -180))

        B1_delta = function_delta(torch.add(torch.mul(h, 360), -120))
        B2_delta = function_delta(torch.add(torch.mul(h, 360), -300))

        one_minus_s = torch.mul(torch.add(s, -1), -1)
        R_1 = torch.add(v, -1, torch.mul(vs, R1_delta))
        R_2 = torch.mul(vs, R2_delta)
        R = torch.add(R_1, R_2)

        G_1 = torch.add(torch.mul(v, one_minus_s), torch.mul(vs, G1_delta))
        G_2 = torch.mul(vs, G2_delta)
        G = torch.add(G_1, -1, G_2)

        B_1 = torch.add(torch.mul(v, one_minus_s), torch.mul(vs, B1_delta))
        B_2 = torch.mul(vs, B2_delta)
        B = torch.add(B_1, -1, B_2)

        del h, s, v, vs, R1_delta, R2_delta, G1_delta, G2_delta, B1_delta, B2_delta, one_minus_s, R_1, R_2, G_1, G_2, B_1, B_2

        R = torch.reshape(R, (batch, 1, w, height))
        G = torch.reshape(G, (batch, 1, w, height))
        B = torch.reshape(B, (batch, 1, w, height))
        RGB_img = torch.cat([R, G, B], 1)

        return RGB_img