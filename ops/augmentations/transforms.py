import random
from typing import List, Tuple, Union, Dict, Any
from enum import Enum

import numpy as np

import cv2
from albumentations import DualTransform
from albumentations.core.bbox_utils import convert_bboxes_to_albumentations

import ops.augmentations.functional as F


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
        M = F.shift_scale_rotate_matrix(center, angle, scale, dx, dy, height, width)
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
        x_min, y_min, x_max, y_max = F.bbox_shift_scale_rotate(bbox, center, angle, scale, dx, dy, height, width)
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
        M = F.shift_scale_rotate_matrix(center, angle, scale, dx, dy, height, width)
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


class Mosaic(DualTransform):
    """Mosaic augmentation arranges randomly selected four images into single one like the 2x2 grid layout.

    Note:
        This augmentation requires additional helper targets as sources of additional
        image and bboxes.
        The targets are:
        - `image_cache`: list of images or 4 dimensional np.nadarray whose first dimension is batch size.
        - `bboxes_cache`: list of bounding boxes. The bounding box format is specified in `bboxes_format`.
        You should make sure that the bounding boxes of i-th image (image_cache[i]) are given by bboxes_cache[i].

        Here is a typical usage:
        ```
        data = transform(image=image, image_cache=image_cache)
        # or
        data = transform(image=image, image_cache=image_cache, bboxes=bboxes, bboxes_cache=bboxes_cache)
        ```

        You can set `image_cache` whose length is less than 3. In such a case, the same image will be selected
        multiple times.
        Note that the image specified by `image` argument is always included.

    Args:
        height (int)): height of the mosaiced image.
        width (int): width of the mosaiced image.
        fill_value (int): padding value.
        replace (bool): whether to allow replacement in sampling or not. When the value is `True`, the same image
            can be selected multiple times. When False, the length of `image_cache` (and `bboxes_cache`) should
            be at least 3.
            This replacement rule is applied only to `image_cache`. So, if the `image_cache` contains the same image as
            the one specified in `image` argument, it can make image that includes duplication for the `image` even if
            `replace=False` is set.
        bboxes_forma (str)t: format of bounding box. Should be on of "pascal_voc", "coco", "yolo".

    Targets:
        image, mask, bboxes, image_cache, mask_cache, bboxes_cache

    Image types:
        uint8, float32

    Reference:
    [Bochkovskiy] Bochkovskiy A, Wang CY, Liao HYM. （2020） "YOLOv 4 : Optimal speed and accuracy of object detection.",
    https://arxiv.org/pdf/2004.10934.pdf

    """

    def __init__(
            self,
            height,
            width,
            fill_value=0,
            always_apply=False,
            p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.fill_value = fill_value
        self.__target_dependence = {}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "height", "width", "replace", "fill_value", "bboxes_cache_format"

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        self.update_target_dependence(**kwargs)
        return super().__call__(force_apply=force_apply, **kwargs)

    @property
    def target_dependence(self) -> Dict:
        return self.__target_dependence

    @target_dependence.setter
    def target_dependence(self, value):
        self.__target_dependence = value

    def update_target_dependence(self, **kwargs):
        """Update target dependence dynamically."""
        self.target_dependence = {}
        if "image" in kwargs:
            self.target_dependence["image"] = {"image_cache": kwargs["image_cache"]}
        if "mask" in kwargs:
            self.target_dependence["mask"] = {"mask_cache": kwargs["mask_cache"]}
        if "bboxes" in kwargs:
            self.target_dependence["bboxes"] = {"bboxes_cache": kwargs["bboxes_cache"]}

    def apply(self, image, image_cache=None, height=0, width=0, fill_value=114, **params):
        image_cache.append(image)
        image, self.padh_cache, self.padw_cache = F.mosaic4(image_cache, height, width, fill_value)
        return image

    def apply_to_mask(self, mask, mask_cache=None, height=0, width=0, fill_value=114, **params):
        mask_cache.append(mask)
        mask, *_ = F.mosaic4(mask_cache, height, width, fill_value)
        return mask

    def apply_to_bbox(self, bbox, padh=0, padw=0, height=0, width=0, **params):
        return F.bbox_mosaic4(bbox, padh, padw, height, width)

    def apply_to_bboxes(
            self,
            bboxes,
            bboxes_cache=None,
            height=0,
            width=0,
            **params
    ):
        bboxes_cache.append(bboxes)
        new_bboxes = []
        self.new_classes = []
        for bbox, classes, padh, padw in zip(bboxes_cache, self.classes_cache, self.padh_cache, self.padw_cache, ):
            for box, cls in zip(bbox, classes):
                new_bbox = self.apply_to_bbox(box, padh, padw, height, width)
                new_bboxes.append(new_bbox)
                self.new_classes.append(cls)
        return new_bboxes

    def apply_to_keypoint(self, **params):
        pass  # TODO

    def get_params(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "fill_value": self.fill_value,
        }

    def apply_with_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        self.classes_cache = kwargs['classes_cache']
        self.classes_cache.append(kwargs['classes'])
        res = super(Mosaic, self).apply_with_params(params, **kwargs)
        res['classes'] = self.new_classes
        return res
