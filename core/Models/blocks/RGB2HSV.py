import torch
import torch.nn as nn


class RGB2HSV(nn.Module):
    def __init__(self):
        super(RGB2HSV, self).__init__()

    def forward(self, rgb):
        batch, c, w, h = rgb.size()
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        V, max_index = torch.max(rgb, dim=1)
        min_rgb = torch.min(rgb, dim=1)[0]
        v_plus_min = V - min_rgb
        S = v_plus_min / (V + 0.0001)
        H = torch.zeros_like(rgb[:, 0, :, :])
        # if rgb.type() == 'torch.cuda.FloatTensor':
        #     H = torch.zeros(batch, w, h).type(torch.cuda.FloatTensor)
        # else:
        #     H = torch.zeros(batch, w, h).type(torch.FloatTensor)
        mark = max_index == 0
        H[mark] = 60 * (g[mark] - b[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 1
        H[mark] = 120 + 60 * (b[mark] - r[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 2
        H[mark] = 240 + 60 * (r[mark] - g[mark]) / (v_plus_min[mark] + 0.0001)

        mark = H < 0
        H[mark] += 360
        H = H % 360
        H = H / 360
        HSV_img = torch.cat([H.view(batch, 1, w, h), S.view(batch, 1, w, h), V.view(batch, 1, w, h)], 1)
        return HSV_img
