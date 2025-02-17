import torch
from torch import nn
from torch.functional import F

from torchvision.ops.misc import FrozenBatchNorm2d

from ops.loss.detr_loss import SetCriterion
from ops.utils.torch_utils import NestedTensor
from ops.models.backbone.utils import Backbone
from ops.models.head.detr_head import DetrHead
from ops.models.detection.detr.model import DetrModel
from ops.models.detection.detr.matcher import HungarianMatcher
from ops.models.misc.position_encoding import PositionEmbeddingSine


class DeformableDETR(DetrModel):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 dim_feedforward,
                 enc_layers,
                 dec_layers,
                 strides,
                 num_channels,
                 num_queries,
                 num_classes,
                 *args,
                 **kwargs):
        super(DeformableDETR, self).__init__(*args, **kwargs)

        self.backbone = Backbone(name='resnet50',
                                 layers_to_train=['layer2', 'layer3', 'layer4'],
                                 return_interm_layers={"layer2": "0", "layer3": "1", "layer4": "2"},
                                 norm_layer=FrozenBatchNorm2d)

        num_backbone_outs = len(strides)
        input_proj_list = []
        for i in range(num_backbone_outs):  # 3个1x1conv
            in_channels = num_channels[i]  # 512  1024  2048
            input_proj_list.append(nn.Sequential(  # conv1x1  -> 256 channel
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        in_channels = num_channels[-1]
        input_proj_list.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),  # 3x3conv s=2 -> 256channel
            nn.GroupNorm(32, hidden_dim),
        ))
        self.input_proj = nn.ModuleList(input_proj_list)

        N_steps = hidden_dim // 2

        self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

        self.dec_layers = dec_layers

        self.num_classes = num_classes

        # self.transformer = Transformer(
        #     d_model=hidden_dim,
        #     nhead=num_heads,
        #     dim_feedforward=dim_feedforward,
        #     num_encoder_layers=enc_layers,
        #     num_decoder_layers=dec_layers,
        # )
        #
        # self.head = DetrHead(hidden_dim, hidden_dim, 4, 3, num_classes + 1)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)

    def forward(self, samples: NestedTensor, orig_target_sizes):
        features = self.backbone(samples)

        srcs = []
        masks = []
        pos = []
        for l, feat in enumerate(features.values()):
            pos.append(self.position_embedding(feat))
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)

        srcs.append(self.input_proj[-1](features['2'].tensors))
        m = samples.mask
        masks.append(F.interpolate(m[None].float(), size=srcs[-1].shape[-2:]).to(torch.bool)[0])
        pos.append(self.position_embedding(NestedTensor(srcs[-1], masks[-1])))

        # hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]

        return self.head(hs, orig_target_sizes)

    def on_fit_start(self) -> None:
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(self.num_classes,
                                      matcher=matcher,
                                      weight_dict=self.hyp['weight_dict'],
                                      eos_coef=0.1,
                                      losses=losses)
