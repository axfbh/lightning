import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class CBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,
                 buffer_num=0,
                 rho=1.0,
                 burnin=0,
                 two_stage=True,
                 FROZEN=False,
                 out_p=False):
        """

        :param num_features: 特征图通道数
        :param eps: 归一化时加在分母上，防止除零
        :param momentum: 平滑值
        :param affine: 是否使用 linear transform
        :param track_running_stats: 是否使用 平滑
        :param buffer_num: 保留多少个 iteration
        :param rho: 泰勒公式，归一化时加在分母上，防止除零
        :param burnin: 第几个迭代开始 统计 iteration
        :param two_stage: 采用什么方式计算 buffer_num
        :param FROZEN: 上个卷积层是否是冻结层
        :param out_p: True : (self.running_var.view(-1, 1) + self.eps) ** .5
                      False : (self.running_var.view(-1, 1) ** .5 + self.eps)
        """
        super(CBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.buffer_num = buffer_num
        self.max_buffer_num = buffer_num
        self.rho = rho
        self.burnin = burnin
        self.two_stage = two_stage
        self.FROZEN = FROZEN
        self.out_p = out_p

        self.iter_count = 0
        self.pre_mu = []
        self.pre_meanx2 = []  # mean(x^2)
        self.pre_dmudw = []
        self.pre_dmeanx2dw = []
        self.pre_weight = []
        self.ones = torch.ones(self.num_features).cuda()

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def _update_buffer_num(self):
        if self.two_stage:
            if self.iter_count > self.burnin:
                self.buffer_num = self.max_buffer_num
            else:
                self.buffer_num = 0
        else:
            self.buffer_num = int(self.max_buffer_num * min(self.iter_count / self.burnin, 1.0))

    def forward(self, input, weight):
        # ------------- 查看 特征图形状 是否满足 [B,C,H,W] -------------
        self._check_input_dim(input)

        # ------------- 更改 特征图形状 [B,C,H,W] -> [C,B,H,W] -------------
        y = input.transpose(0, 1)
        return_shape = y.shape

        # ------------- 更改 特征图形状 [C,B,H,W] -> [B,C*H*W] -------------
        y = y.contiguous().view(input.size(1), -1)

        # ------------- 累加 最近 burnin 迭代 -------------
        if self.training and self.burnin > 0:
            self.iter_count += 1
            self._update_buffer_num()

        # ------------------- 计算 估计值 μ 和 σ -------------------
        # 1：满足 burin 迭代
        # 2：满足 训练模式
        # 3：满足 不是冻结层
        if self.buffer_num > 0 and self.training and input.requires_grad:  # some layers are frozen!
            # ------------- 计算当前 batch μ 和 v 和 σ -------------
            cur_mu = y.mean(dim=1)
            # v
            cur_meanx2 = torch.pow(y, 2).mean(dim=1)
            cur_sigma2 = y.var(dim=1)

            # cal partial(μ/Θ)
            dmudw = torch.autograd.grad(cur_mu, weight, self.ones, retain_graph=True)[0]
            # cal partial(v/Θ)
            dmeanx2dw = torch.autograd.grad(cur_meanx2, weight, self.ones, retain_graph=True)[0]

            # ------------- 统计 最近迭代的 μ -------------
            mu_all = torch.stack(
                # ---------- 泰勒公式 -------------
                [cur_mu, ] + [tmp_mu + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for
                              tmp_mu, tmp_d, tmp_w in zip(self.pre_mu, self.pre_dmudw, self.pre_weight)])

            # ------------- 统计 最近迭代的 v -------------
            meanx2_all = torch.stack(
                # ---------- 泰勒公式 -------------
                [cur_meanx2, ] + [tmp_meanx2 + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for
                                  tmp_meanx2, tmp_d, tmp_w in
                                  zip(self.pre_meanx2, self.pre_dmeanx2dw, self.pre_weight)])

            # ------------- 统计 最近迭代的 σ -------------
            sigma2_all = meanx2_all - torch.pow(mu_all, 2)

            # ------------- 防止 值 修改 -------------
            re_mu_all = mu_all.clone()
            re_meanx2_all = meanx2_all.clone()

            re_mu_all[sigma2_all < 0] = 0
            re_meanx2_all[sigma2_all < 0] = 0

            count = (sigma2_all >= 0).sum(dim=0).float()

            # ------------- 估计 当前的 μ 和 σ  -------------
            mu = re_mu_all.sum(dim=0) / count
            sigma2 = re_meanx2_all.sum(dim=0) / count - torch.pow(mu, 2)

            # ------------- 只保留 buffer_num 迭代的 μ 和 v 和 σ 和 Θ -------------
            self.pre_mu = [cur_mu.detach(), ] + self.pre_mu[:(self.buffer_num - 1)]
            self.pre_meanx2 = [cur_meanx2.detach(), ] + self.pre_meanx2[:(self.buffer_num - 1)]
            self.pre_dmudw = [dmudw.detach(), ] + self.pre_dmudw[:(self.buffer_num - 1)]
            self.pre_dmeanx2dw = [dmeanx2dw.detach(), ] + self.pre_dmeanx2dw[:(self.buffer_num - 1)]

            # ------------- 防止 值 修改 -------------
            tmp_weight = torch.zeros_like(weight.data)
            tmp_weight.copy_(weight.data)
            self.pre_weight = [tmp_weight.detach(), ] + self.pre_weight[:(self.buffer_num - 1)]

        else:
            x = y
            mu = x.mean(dim=1)
            cur_mu = mu
            sigma2 = x.var(dim=1)
            cur_sigma2 = sigma2

        # ------------ 计算 normalization ------------
        if not self.training or self.FROZEN:
            y = y - self.running_mean.view(-1, 1)
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (self.running_var.view(-1, 1) + self.eps) ** .5
            else:
                y = y / (self.running_var.view(-1, 1) ** .5 + self.eps)

        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * cur_mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * cur_sigma2
            y = y - mu.view(-1, 1)
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (sigma2.view(-1, 1) + self.eps) ** .5
            else:
                y = y / (sigma2.view(-1, 1) ** .5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'buffer={max_buffer_num}, burnin={burnin}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)
