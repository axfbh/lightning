import random
from typing import List, Tuple, Union, Dict, Any
from enum import Enum

import numpy as np

import cv2
from albumentations import DualTransform
from ops.augmentations.functional import bbox_shift_scale_rotate, shift_scale_rotate_matrix


class RandomShiftScaleRotate(DualTransform):
    class PositionType(Enum):
        CENTER = "center"
        TOP_LEFT = "top_left"
        TOP_RIGHT = "top_right"
        BOTTOM_LEFT = "bottom_left"
        BOTTOM_RIGHT = "bottom_right"

    def __init__(self,
                 scale_limit=(0.5, 1.5),
                 shift_limit_x=(0.4, 0.6),
                 shift_limit_y=(0.4, 0.6),
                 rotate_limit=(-45, 45),
                 position=PositionType.CENTER,
                 border_mode: int = cv2.BORDER_REFLECT_101,
                 value=None,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)
        self.rotate_limit = rotate_limit
        self.scale_limit = scale_limit
        self.shift_limit_x = shift_limit_x
        self.shift_limit_y = shift_limit_y
        self.position = position
        self.border_mode = border_mode
        self.value = value

    def apply(
            self,
            img: np.ndarray,
            angle: float = 0,
            scale: float = 0,
            dx: int = 0,
            dy: int = 0,
            **params
    ) -> np.ndarray:
        height, width = params['rows'], params['cols']
        center = self.__update_center_params(width, height)
        M = shift_scale_rotate_matrix(center, angle, scale, dx, dy, height, width)
        return cv2.warpAffine(img, M[:2], dsize=(width, height), borderMode=self.border_mode, borderValue=self.value)

    def apply_to_bbox(
            self,
            bbox: Tuple,
            angle: float = 0,
            scale: float = 0,
            dx: int = 0,
            dy: int = 0,
            **params
    ) -> Tuple:
        height, width = params['rows'], params['cols']
        center = self.__update_center_params(width, height)
        x_min, y_min, x_max, y_max = bbox_shift_scale_rotate(bbox, center, angle, scale, dx, dy, height, width)
        return max(x_min, 0), max(y_min, 0), min(x_max, width), min(y_max, height)

    def apply_to_mask(
            self,
            mask: np.ndarray,
            angle: float = 0,
            scale: float = 0,
            dx: int = 0,
            dy: int = 0,
            **params
    ) -> np.ndarray:
        height, width = params['rows'], params['cols']
        center = self.__update_center_params(width, height)
        M = shift_scale_rotate_matrix(center, angle, scale, dx, dy, height, width)
        return cv2.warpAffine(mask, M[:2], dsize=(width, height), borderMode=self.border_mode, borderValue=self.value)

    def get_params(self) -> Dict[str, Any]:
        return {
            "angle": random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
            "scale": random.uniform(self.scale_limit[0], self.scale_limit[1]),
            "dx": random.uniform(self.shift_limit_x[0], self.shift_limit_x[1]),
            "dy": random.uniform(self.shift_limit_y[0], self.shift_limit_y[1]),
        }

    def get_transform_init_args_names(self):
        return (
            "scale_limit",
            "shift_limit_x",
            "shift_limit_y",
            "position",
            "border_mode",
            "value",
        )

    def __update_center_params(self, width, height):
        if self.position == RandomShiftScaleRotate.PositionType.TOP_LEFT:
            return 0, 0

        elif self.position == RandomShiftScaleRotate.PositionType.TOP_RIGHT:
            return width, 0

        elif self.position == RandomShiftScaleRotate.PositionType.BOTTOM_LEFT:
            return 0, height

        elif self.position == RandomShiftScaleRotate.PositionType.BOTTOM_RIGHT:
            return width, height

        elif self.position == RandomShiftScaleRotate.PositionType.CENTER:
            return width / 2, height / 2
