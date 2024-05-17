import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.ops.focal_loss import sigmoid_focal_loss as _sigmoid_focal_loss
from torchvision.ops.focal_loss import sigmoid_focal_loss


def sigmoid_one_hot_focal_loss(inputs,
                               targets,
                               num_classes,
                               gamma=2.0,
                               alpha=0.25,
                               reduction='mean'):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        targets (Tensor): The number of classes, [background,...], must account the background class
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    targets = F.one_hot(targets, num_classes=num_classes)

    targets = targets.type_as(inputs)

    return sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction)
