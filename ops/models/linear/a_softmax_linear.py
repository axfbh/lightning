import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ASoftmaxLinear(nn.Module):
    def __init__(self, feat_dim, num_classes, margin, device):
        super(ASoftmaxLinear, self).__init__()

        self.margin = margin

        self.it = 0

        self.lambdas_min = 5
        self.lambdas_max = 1500

        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, num_classes).to(device))
        nn.init.xavier_uniform_(self.weight.data)

        # 利用 l-softmax 的 cos(mθ) 计算方式，简化模式
        self.mfunc = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,

        ]

    def find_k(self, cos: torch.Tensor):
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        theta = cos.acos()
        k = ((self.margin * theta) / math.pi).floor().detach()
        return k

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.it += 1

        lambdas = max(self.lambdas_min, self.lambdas_max / (1 + 0.1 * self.it))

        # 对w进行归一化
        W_normalize = F.normalize(self.weight, 2, 0)

        logits = inputs.mm(W_normalize)
        indexes = range(logits.size(0))
        logits_yi = logits[indexes, targets]

        Wyi = W_normalize[:, targets]
        Wyi_norm = Wyi.norm(2, 0)
        x_norm = inputs.norm(2, 1)
        cos_theta_yi = logits_yi / (Wyi_norm * x_norm + 1e-10)

        cos_m_theta_yi = self.mfunc[self.margin](cos_theta_yi)

        k = self.find_k(cos_theta_yi)

        psi_theta = (-1) ** k * cos_m_theta_yi - 2 * k

        fyi = (x_norm * psi_theta + lambdas * x_norm * cos_theta_yi) / (1 + lambdas)

        logits[indexes, targets] = fyi

        return logits


# if __name__ == '__main__':
#     x = torch.tensor([[-8, 6, 0, -15],
#                       [-6, 5, 7, 0],
#                       [-6, 4, 7, 2]], dtype=torch.float32)
#     labels = torch.tensor([0, 1, 1])
#     num_classes = 2
#     dim = 4
#
#     loss = ASoftmaxLinear(dim, num_classes, 4, 'cpu')(x, labels)
#     print(loss)
