from typing import Optional, List, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models._utils import IntermediateLayerGetter

import ops
from ops.models.backbone import cspdarknet
from ops.models.backbone import darknet
from ops.models.backbone import elandarknet
from ops.utils.torch_utils import NestedTensor


def _cspdarknet_extractor(
        backbone: Union[cspdarknet.CSPDarknetV4, cspdarknet.CSPDarknetV5, cspdarknet.CSPDarknetV8],
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
                       "stem"][:trainable_layers]

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
                       "stem"][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,5]. Got {returned_layers}")
    return_layers = {f"stage{k}": str(v) for v, k in enumerate(returned_layers)}

    return IntermediateLayerGetter(backbone, return_layers)


def _mobilenet_extractor(
        backbone,
        trainable_layers: int):
    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 1:
        raise ValueError(f"Trainable layers should be in the range [0,1], got {trainable_layers}")
    layers_to_train = ["features"][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {f"features": '0'}

    return IntermediateLayerGetter(backbone, return_layers)


def _shufflenet_extractor(
        backbone,
        trainable_layers: int):
    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 6:
        raise ValueError(f"Trainable layers should be in the range [0,6], got {trainable_layers}")
    layers_to_train = ["conv1",
                       'maxpool',
                       'stage2',
                       'stage3',
                       'stage4',
                       'conv5'][:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {"conv1": '0',
                     "maxpool": '1',
                     "stage2": '2',
                     "stage3": '3',
                     "stage4": '4',
                     'conv5': '5'}

    return IntermediateLayerGetter(backbone, return_layers)


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, layers_to_train: List, return_interm_layers: Dict):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_interm_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self,
                 name: str,
                 layers_to_train: List,
                 return_interm_layers: Dict,
                 norm_layer=nn.BatchNorm2d,
                 pretrained=True):
        backbone = getattr(ops.models.backbone, name)(pretrained=pretrained, norm_layer=norm_layer)
        super().__init__(backbone, layers_to_train, return_interm_layers)
