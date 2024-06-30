from torch import nn

from ops.models.neck.panet import PanNetTopDown
from ops.models.head.yolo_head import YoloHead
from ops.models.backbone.utils import _darknet_extractor
from ops.models.backbone.darknet import DarkNet
from ops.models.detection.utils import Yolo
from ops.loss.yolo_loss import YoloLossV3


class ConvolutionalLayer(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=None, activation_layer=None):
        activation_layer = nn.LeakyReLU if activation_layer is None else activation_layer
        p = (k - 1) // 2 if p is None else p
        super(ConvolutionalLayer, self).__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_layer(),
        )


class YVBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(YVBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_ch, out_ch, 1),
            ConvolutionalLayer(out_ch, in_ch, 3),
            ConvolutionalLayer(in_ch, out_ch, 1),
            ConvolutionalLayer(out_ch, in_ch, 3),
            ConvolutionalLayer(in_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.conv(x)


class YoloV3(nn.Module):
    def __init__(self, num_classes):
        super(YoloV3, self).__init__()

        self.backbone = _darknet_extractor(DarkNet(), 5)

        self.neck = PanNetTopDown([256, 128], YVBlock, use_cat=True)

        self.head = YoloHead([256, 512, 1024], num_classes)

    def forward(self, x):
        x = self.backbone(x)

        h_52 = x['0']
        h_26 = x['1']
        h_13 = x['2']

        sample = [h_13, h_26, h_52]

        neck = self.neck(sample)

        head = self.head(neck)

        detection_52 = head[0]
        detection_26 = head[1]
        detection_13 = head[2]

        return detection_13, detection_26, detection_52


def get_model(args):
    model = YoloV3(num_classes=3 * (args.num_classes + 5))
    return model
