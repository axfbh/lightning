import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ArcMarginLinear(nn.Module):
    def __init__(self, feat_dim, num_classes, margin, s, device):
        super(ArcMarginLinear, self).__init__()

        self.margin = margin
        self.s = s

        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, num_classes).to(device))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, inputs: torch.Tensor, targets):
        x_normalize = F.normalize(inputs, 2, 1)
        W_normalize = F.normalize(self.weight, 2, 0)

        cos_theta = x_normalize.mm(W_normalize)

        # cos(θ) = sqrt(1 - sin2(θ))
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2))

        # cos(Θ+m) = cosθcosm - sinθsinm
        cos_theta_m = cos_theta * math.cos(self.margin) - sin_theta * math.sin(self.margin)

        one_hot = torch.zeros_like(cos_theta_m)

        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        output = self.s * (one_hot * cos_theta_m) + ((1 - one_hot) * cos_theta)

        return output

#
# if __name__ == '__main__':
#     x = torch.tensor([[-8, 6, 0, -15],
#                       [-6, 5, 7, 0],
#                       [-6, 4, 7, 2]], dtype=torch.float32)
#     labels = torch.tensor([0, 1, 1])
#     num_classes = 2
#     dim = 4
#
#     loss = ArcMarginLinear(dim, num_classes, 1.56, 3, 'cpu')(x, labels)
#     print(loss)
