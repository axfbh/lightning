import torch
from torch import Tensor


def _ignore_background(preds: Tensor, target: Tensor):
    """Ignore the background class in the computation assuming it is the first, index 0."""
    preds = preds[:, 1:] if preds.shape[1] > 1 else preds
    target = target[:, 1:] if target.shape[1] > 1 else target
    return preds, target


def _preprocess_preds_targets(
        preds: Tensor,
        targets: Tensor,
        num_classes: int,
        include_background=False):
    if (preds.bool() != preds).any():  # preds is an index tensor
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
    if (targets.bool() != targets).any():  # target is an index tensor
        targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).movedim(-1, 1)

    if not include_background:
        preds, targets = _ignore_background(preds, targets)

    return preds, targets
