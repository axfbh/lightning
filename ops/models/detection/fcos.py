import torch
import torch.nn as nn
import math
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from ops.models.head.fcos_head import FCOSHead


class FCOS(nn.Module):
    def __init__(self, num_classes, anchors, aspect_ratios):
        super(FCOS, self).__init__()

        weights_backbone = ResNet50_Weights.IMAGENET1K_V1

        backbone = resnet50(weights=weights_backbone, progress=True)

        is_trained = weights_backbone is not None

        trainable_backbone_layers = _validate_trainable_layers(is_trained, None, 5, 3)

        self.backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers,
                                              returned_layers=[2, 3, 4],
                                              extra_blocks=LastLevelP6P7(256, 256))

        self.head = FCOSHead([256, 256, 256, 256, 256], anchors, aspect_ratios, num_classes)

    def forward(self, x):
        _, _, H, W = x.size()

        fpn_out = self.backbone(x)

        p7 = fpn_out['p7']
        p6 = fpn_out['p6']
        p5 = fpn_out['2']
        p4 = fpn_out['1']
        p3 = fpn_out['0']

        return self.head([p3, p4, p5, p6, p7], H, W)
