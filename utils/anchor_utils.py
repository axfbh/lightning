import torch
from typing import List
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator as AG


class AnchorGenerator(AG):
    def __init__(
            self,
            sizes=((128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        if not isinstance(sizes[0], tuple):
            # TODO change this
            sizes = tuple(tuple(s,) for s in sizes)
        if not isinstance(aspect_ratios[0], tuple):
            aspect_ratios = tuple(tuple(s, ) for s in aspect_ratios)

        super(AnchorGenerator, self).__init__(sizes, aspect_ratios)

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
        return anchors_over_all_feature_maps, torch.tensor(strides, dtype=dtype, device=device)

#
# # if __name__ == '__main__':
# #     anchor_generator = AnchorGenerator([8, 16, 32, 64, 128])
# #     anchor_generator([800, 1216], [[100, 152], [50, 76], [25, 38], [13, 19], [7, 10]])
