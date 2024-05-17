import torch.nn as nn


class AdaptiveGroupNorm(nn.Module):
    def __init__(self,
                 num_channels: int,
                 num_groups: int = None,
                 eps: float = 1e-5,
                 affine: bool = True,
                 device=None,
                 dtype=None) -> None:
        super(AdaptiveGroupNorm, self).__init__()

        if num_groups is None:
            for num in range(32, 1, - 1):
                if num_channels % num == 0:
                    self.norm_layer = nn.GroupNorm(num_groups=num,
                                                   num_channels=num_channels,
                                                   eps=eps,
                                                   affine=affine,
                                                   device=device,
                                                   dtype=dtype)
                    break
            else:
                self.norm_layer = nn.GroupNorm(num_groups=num_channels,
                                               num_channels=num_channels,
                                               eps=eps,
                                               affine=affine,
                                               device=device,
                                               dtype=dtype)
        else:
            self.norm_layer = nn.GroupNorm(num_groups=num_groups,
                                           num_channels=num_channels,
                                           eps=eps,
                                           affine=affine,
                                           device=device,
                                           dtype=dtype)

    def forward(self, x):
        if x.size(0) > 32:
            raise ValueError('Batch Size bigger than 32, must using BN')
        return self.norm_layer(x)
