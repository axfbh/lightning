import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

torch.set_printoptions(precision=3, sci_mode=False)


def _pad_1x1_to_3x3_tensor(kernel1x1):
    return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])


def _fuse_bn_without_conv(bn, groups):
    input_dim = bn.num_features // groups
    weight = torch.zeros((bn.num_features, input_dim, 3, 3), dtype=torch.float32, device=bn.weight.device)
    x = torch.arange(bn.num_features)
    y = torch.arange(input_dim).repeat(groups)
    weight[x, y, 1, 1] = 1
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return weight * t, beta - running_mean * gamma / std


class _RepConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch,
                              kernel_size=kernel_size,
                              groups=groups,
                              padding=padding,
                              stride=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def train(self, mode: bool = True):
        super().train(mode)

        if not mode:
            conv_fuse = utils.fuse_conv_bn_eval(self.conv, self.bn)
            self.fuse_weight = conv_fuse.weight
            self.fuse_bias = conv_fuse.bias

    def forward(self, x):
        return self.bn(self.conv(x))


class _RepBn(nn.Module):
    def __init__(self, in_ch, groups):
        super().__init__()
        self.in_ch = in_ch
        self.groups = groups
        self.bn = nn.BatchNorm2d(in_ch)

    def train(self, mode: bool = True):
        super().train(mode)

        if not mode:
            weight, bias = _fuse_bn_without_conv(self.bn, self.groups)
            self.fuse_weight = weight
            self.fuse_bias = bias

    def forward(self, x):
        return self.bn(x)


class RepConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, groups=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.rep_conv3x3 = _RepConv2d(in_ch, out_ch, 3, groups)
        self.rep_conv1x1 = _RepConv2d(in_ch, out_ch, 1, groups)
        self.bn = None if in_ch != out_ch else _RepBn(out_ch, groups)

    def _fuse_conv(self, x):
        kernel3x3 = self.rep_conv3x3.fuse_weight
        bias3x3 = self.rep_conv3x3.fuse_bias

        kernel1x1 = self.rep_conv1x1.fuse_weight
        bias1x1 = self.rep_conv1x1.fuse_bias

        kernelid = self.bn.fuse_weight if self.bn is not None else 0
        biasid = self.bn.fuse_bias if self.bn is not None else 0

        kernel = kernel3x3 + _pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        bias = bias3x3 + bias1x1 + biasid

        return F.conv2d(x, kernel, bias,
                        stride=self.rep_conv3x3.conv.stride,
                        padding=self.rep_conv3x3.conv.padding,
                        dilation=self.rep_conv3x3.conv.dilation,
                        groups=self.rep_conv3x3.conv.groups)

    def forward(self, x):
        if self.training:
            x1 = self.rep_conv3x3(x)
            x2 = self.rep_conv1x1(x)
            x3 = self.bn(x) if self.bn is not None else 0
            return x1 + x2 + x3

        return self._fuse_conv(x)
