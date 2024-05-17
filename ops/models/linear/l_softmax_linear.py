import torch
import torch.nn as nn
import math
from scipy.special import binom


class LSoftmaxLinear(nn.Module):
    def __init__(self, feat_dim, num_classes, margin, device):
        super().__init__()
        self.margin = margin  # m
        self.lambdas_start = 100
        self.lambdas_min = 0
        self.scale = 0.99

        self.device = device  # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, num_classes).to(device))
        nn.init.kaiming_normal_(self.weight.data)

        # 计算组合数
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        """
        cos(mθ) 的加速计算方式
        :param cos_theta:
        :return:
        """
        sin2_theta = 1 - cos_theta ** 2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = ((acos * self.margin) / math.pi).floor().detach()
        return k

    def forward(self, inputs, targets):
        lambdas = max(self.lambdas_start, self.lambdas_min)

        logit = inputs.mm(self.weight)

        # 选择第几张图片
        indexes = range(logit.size(0))

        # indexes:第几张图片
        # targets:图片对应的类别特征
        logit_yi = logit[indexes, targets]

        # cos(theta) = w * x / ||w||*||x||
        Wyi = self.weight[:, targets]
        Wyi_norm = Wyi.norm(p=2, dim=0)
        x_norm = inputs.norm(p=2, dim=1)
        cos_theta_yi = logit_yi / (Wyi_norm * x_norm + 1e-10)

        # equation 7
        cos_m_theta_yi = self.calculate_cos_m_theta(cos_theta_yi)

        # find k in equation 6
        k = self.find_k(cos_theta_yi)

        # f_y_i
        psi_theta = (-1) ** k * cos_m_theta_yi - 2 * k

        fyi = (Wyi_norm * x_norm * psi_theta + lambdas * logit[indexes, targets]) / (1 + lambdas)

        logit[indexes, targets] = fyi
        self.lambdas_start *= self.scale
        return logit
#
#
# if __name__ == '__main__':
#     x = torch.tensor([[-8, 6, 0, -15],
#                       [-6, 5, 7, 0],
#                       [-6, 4, 7, 2]], dtype=torch.float32)
#     labels = torch.tensor([0, 1, 1])
#     num_classes = 2
#     dim = 4
#
#     loss = LSoftmaxLinear(dim, num_classes, 4, 'cpu')(x, labels)
#
#     print(loss)
