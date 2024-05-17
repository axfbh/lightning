import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def margin_inferred(C, K):
    if K == 2:
        return 1 - math.cos((2 * math.pi) / C)
    elif C <= (K + 1):
        return C / (C - 1)
    return 1 - math.cos(math.pi / 4)


def s_inferred(C, Pw):
    return ((C - 1) / C) * math.log(((C - 1) * Pw) / (1 - Pw))


class CosineMarginLinear(nn.Module):
    def __init__(self, feat_dim, num_classes, device):
        super(CosineMarginLinear, self).__init__()

        self.s = s_inferred(num_classes, 0.95)
        self.margin = margin_inferred(num_classes, feat_dim)
        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, num_classes).to(device))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        x_normalize = F.normalize(inputs, 2, 1)
        W_normalize = F.normalize(self.weight, 2, 0)

        cos_theta = x_normalize.mm(W_normalize)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1.0)

        output = self.s * (cos_theta - one_hot * self.margin)

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
#     loss = CosineMarginLinear(dim, num_classes, 'cpu')(x, labels)
#     print(loss)
