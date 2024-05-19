import random
from typing import List, Tuple, Union, Dict, Any

import numpy as np

import cv2
from albumentations import DualTransform

from ops.transform.pad_maker import PaddingImage
import ops.cv.io as io


def resize_boxes_ratio(boxes, ratio_height, ratio_width):
    xmin, ymin, xmax, ymax = boxes

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return xmin, ymin, xmax, ymax


def resize_boxes(boxes, original_size, new_size):
    ratio_height = new_size[0] / original_size[0]
    ratio_width = new_size[1] / original_size[1]
    return resize_boxes_ratio(boxes, ratio_height, ratio_width)


class ResizeShortLongest(DualTransform):
    def __init__(self, image_size: List[int],
                 always_apply=False,
                 p=0.5):
        super(ResizeShortLongest, self).__init__(always_apply, p)
        self.image_size = image_size
        self.min_size, self.max_size = sorted(image_size)

    def apply(self, img: np.ndarray, min_size=0, max_size=0, **params) -> np.ndarray:
        original_size = (params['rows'], params['cols'])
        original_min_size = min(original_size)
        original_max_size = max(original_size)
        ratio = round(min(min_size / original_min_size, max_size / original_max_size), 5)
        resize_image = cv2.resize(img, None, fx=ratio, fy=ratio)
        return resize_image

    def apply_to_bbox(self, bbox, min_size=0, max_size=0, **params) -> Tuple:
        return bbox

    def apply_to_mask(self, mask, min_size=0, max_size=0, **params) -> np.ndarray:
        return mask

    def get_params(self) -> Dict[str, Any]:
        return {
            "min_size": self.min_size,
            "max_size": self.max_size
        }

    def get_transform_init_args_names(self):
        return (
            "image_size",
        )
