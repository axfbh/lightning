import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from ops.models.backbone.utils import _mobilenet_extractor


class LiteSeg(nn.Module):
    def __init__(self, num_classes):
        super(LiteSeg, self).__init__()
        self.backbone = _mobilenet_extractor(mobilenet_v3_large(pretrained=True), 1)
