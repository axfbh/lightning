import torch
from ops.models.backbone import cspdarknet
from ops.models.backbone import darknet
from ops.models.backbone import elandarknet
from typing import Optional, List, Union
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter


def _cspdarknet_extractor(
        backbone: Union[cspdarknet.CSPDarknetV1, cspdarknet.CSPDarknetV2],
        trainable_layers: int,
        returned_layers: Optional[List[int]] = None):
    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 6:
        raise ValueError(f"Trainable layers should be in the range [0,6], got {trainable_layers}")
    layers_to_train = ["crossStagePartial4",
                       "crossStagePartial3",
                       "crossStagePartial2",
                       "crossStagePartial1",
                       "stem"][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,5]. Got {returned_layers}")
    return_layers = {f"crossStagePartial{k}": str(v) for v, k in enumerate(returned_layers)}

    return IntermediateLayerGetter(backbone, return_layers)


def _darknet_extractor(
        backbone: darknet.DarkNet,
        trainable_layers: int,
        returned_layers: Optional[List[int]] = None):
    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["feature4",
                       "feature3",
                       "feature2",
                       "feature1",
                       "stem", ][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,5]. Got {returned_layers}")
    return_layers = {f"feature{k}": str(v) for v, k in enumerate(returned_layers)}

    return IntermediateLayerGetter(backbone, return_layers)


def _elandarknet_extractor(
        backbone: elandarknet.ElanDarkNet,
        trainable_layers: int,
        returned_layers: Optional[List[int]] = None):
    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["stage4",
                       "stage3",
                       "stage2",
                       "stage1",
                       "stem", ][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,5]. Got {returned_layers}")
    return_layers = {f"stage{k}": str(v) for v, k in enumerate(returned_layers)}

    return IntermediateLayerGetter(backbone, return_layers)
