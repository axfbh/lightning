import torch
from typing import List
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator as AG


class AnchorGenerator(AG):

    def forward(self, image_sizes: torch.Tensor, feature_maps: torch.Tensor):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        strides = [
            [
                image_sizes[0] // g[0],
                image_sizes[1] // g[1],
            ]
            for g in grid_sizes
        ]

        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        # anchors: List[List[torch.Tensor]] = []
        # for _ in range(len(feature_maps)):
        #     anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
        #     anchors.append(anchors_in_image)
        # anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        strides = [torch.tensor(st, device=device) for st in strides]
        return anchors_over_all_feature_maps, strides

