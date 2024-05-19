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


class Resize(DualTransform):
    def __init__(self, image_size: Union[List[int], Tuple[int]],
                 always_apply=False,
                 p=0.5):
        super(Resize, self).__init__(always_apply, p)
        self.image_size = image_size
        self.height, self.width = image_size

    def apply(self, img: np.ndarray, new_size, **params) -> np.ndarray:
        resize_image = cv2.resize(img, (new_size[1], new_size[0]))
        return resize_image

    def apply_to_bbox(self, bbox, new_size, **params) -> Tuple:
        original_size = (params['rows'], params['cols'])
        bbox = resize_boxes(bbox, original_size, new_size)
        return bbox

    def apply_to_mask(self, mask, new_size, **params) -> np.ndarray:
        mask = cv2.resize(mask, (new_size[1], new_size[0]))
        return mask

    def get_params(self) -> Dict[str, Any]:
        return {
            "new_size": (self.height, self.width)
        }

    def get_transform_init_args_names(self):
        return (
            "image_size",
        )


class ResizeShortLongest(DualTransform):
    def __init__(self, image_size: List[int],
                 always_apply=False,
                 p=0.5):
        super(ResizeShortLongest, self).__init__(always_apply, p)
        self.image_size = image_size
        self.min_size, self.max_size = sorted(image_size)

    def apply(self, img: np.ndarray, min_size, max_size, **params) -> np.ndarray:
        original_size = (params['rows'], params['cols'])
        original_min_size = min(original_size)
        original_max_size = max(original_size)
        ratio = round(min(min_size / original_min_size, max_size / original_max_size), 5)
        resize_image = cv2.resize(img, None, fx=ratio, fy=ratio)
        return resize_image

    def apply_to_bbox(self, bbox, min_size, max_size, **params) -> Tuple:
        original_size = (params['rows'], params['cols'])
        original_min_size = min(original_size)
        original_max_size = max(original_size)
        ratio = round(min(min_size / original_min_size, max_size / original_max_size), 5)
        bbox = resize_boxes_ratio(bbox, ratio, ratio)
        return bbox

    def apply_to_mask(self, mask, min_size, max_size, **params) -> np.ndarray:
        original_size = (params['rows'], params['cols'])
        original_min_size = min(original_size)
        original_max_size = max(original_size)
        ratio = round(min(min_size / original_min_size, max_size / original_max_size), 5)
        mask = cv2.resize(mask, None, fx=ratio, fy=ratio)
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


class ResizeLongestPaddingShort(DualTransform):

    def __init__(self, image_size: List[int], shuffle: bool = False, color=(114, 114, 114),
                 always_apply=False,
                 p=0.5):
        super(ResizeLongestPaddingShort, self).__init__(always_apply, p)
        """
        填充边界，防止图像缩放变形，基于短边
        :param shuffle: True 随机填充边界, False 对半填充边界

        :return:
        """
        self.image_size = image_size
        super(ResizeLongestPaddingShort, self).__init__()
        self.min_size, self.max_size = sorted(image_size)
        self.shuffle = shuffle
        self.color = color
        self.resize_short_longest_func = ResizeShortLongest(image_size, always_apply=True)
        self.padding_func = PaddingImage(0, 0, 0, 0, always_apply=True)

    def apply(self, img: np.ndarray, min_size, max_size, **params) -> np.ndarray:
        img = self.resize_short_longest_func.apply(img, min_size, max_size, **params)

        h, w = img.shape[:2]

        image_size = max(h, w)

        self.gap_h = 0 if h == image_size else (image_size - h)
        self.gap_w = 0 if w == image_size else (image_size - w)

        if self.shuffle:
            self.pad_t = random.randint(0, self.gap_h)
            self.pad_l = random.randint(0, self.gap_w)
            self.pad_b = self.gap_h - self.pad_t
            self.pad_r = self.gap_w - self.pad_l
        else:
            self.pad_t = self.gap_h // 2
            self.pad_l = self.gap_w // 2
            self.pad_b = self.gap_h - self.pad_t
            self.pad_r = self.gap_w - self.pad_l

        return self.padding_func.apply(img, self.color, self.pad_l, self.pad_t, self.pad_r, self.pad_b, **params)

    def apply_to_bbox(self, bbox, min_size, max_size, **params) -> Tuple:
        bbox = self.resize_short_longest_func.apply_to_bbox(bbox, min_size, max_size, **params)
        bbox = self.padding_func.apply_to_bbox(bbox, self.pad_l, self.pad_t, self.pad_r, self.pad_b, **params)
        return bbox

    def apply_to_mask(self, mask, min_size, max_size, **params) -> np.ndarray:
        mask = self.resize_short_longest_func.apply_to_mask(mask, min_size, max_size, **params)
        mask = self.padding_func.apply_to_mask(mask, self.pad_l, self.pad_t, self.pad_r, self.pad_b, **params)
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

#
# if __name__ == '__main__':
#     image = io.imread(r"D:\cgm\dataset\VOC2007\JPEGImages\000005.jpg")
#     print(image.shape)
#     for _ in range(3):
#         x0, y0, x1, y1 = 25, 12, 430, 310
#         var = ResizeLongestPaddingShort(image_size=[416, 600], shuffle=True)(image,
#                                                                              bboxes=np.array([[x0, y0, x1, y1]],
#                                                                                              dtype=float))
#         var1 = cv2.rectangle(var['image'].copy(),
#                              tuple(var['bboxes'][0, [0, 1]].astype(int)),
#                              tuple(var['bboxes'][0, [2, 3]].astype(int)),
#                              (255, 255, 0), 1)
#         print(var1.shape)
#         io.show('ad', var1)
