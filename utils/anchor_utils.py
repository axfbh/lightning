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
            sizes = tuple((s,) for s in sizes)

        if not isinstance(aspect_ratios[0], tuple):
            aspect_ratios = tuple((s,) for s in aspect_ratios)

        super(AnchorGenerator, self).__init__(sizes, aspect_ratios)

    def forward(self, image_sizes: torch.Tensor, feature_maps: torch.Tensor):
        # grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
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


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


# if __name__ == '__main__':
    # anchor_generator = AnchorGenerator([8, 16, 32, 64, 128], [1, 1, 1, 1, 1])
    # an1 = anchor_generator([800, 1216],
    #                        [torch.rand((100, 152)),64*
    #                         torch.rand((50, 76)),
    #                         torch.rand((25, 38)),
    #                         torch.rand((13, 19)),
    #                         torch.rand((7, 10))])

    # anchor_generator = AnchorGenerator([10, 20, 40], [1, 1, 1])
    # an1, _ = anchor_generator([640, 640],
    #                           [torch.rand((64, 64)),
    #                            torch.rand((32, 32)),
    #                            torch.rand((16, 16))])
    # print(an1)
    #
    # an2, _ = make_anchors([torch.rand((1, 3, 64, 64)),
    #                        torch.rand((1, 3, 32, 32)),
    #                        torch.rand((1, 3, 16, 16))], [10, 20, 40], grid_cell_offset=0.5)
    # print(an2)
