from typing import Dict, Any, Tuple

import cv2
import numpy as np

from albumentations import ImageOnlyTransform, DualTransform
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox

import ops.augmentations.functional as F
import ops.cv.io as io


class SaltPepperNoise(ImageOnlyTransform):
    def __init__(self,
                 color: int = 180,
                 n: int = 1000,
                 noise_scale: int = 5,
                 border_scale: float = 0.15,
                 salience: bool = False,
                 always_apply=False,
                 p: float = 0.5):
        """
        椒盐噪音制作
        :param n: 产生多少个噪点
        :param color: 噪音颜色
        :param noise_scale: 噪音缩小放大因子
        :param border_scale: 边缘的噪音点消除比例
        :param p: 噪音出现概率
        :return:
        """

        super(SaltPepperNoise, self).__init__(always_apply, p)

        self.salience = salience
        self.noise_scale = noise_scale
        self.border_scale = border_scale
        self.n = n
        self.color = color

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
            self.target_dependence["image"] = {"bboxes": kwargs["bboxes"]}

    def apply(self, image: np.ndarray, bboxes=None, **params):
        h, w = image.shape[:2]
        salience_area = F.bbox_salience_area(bboxes) if self.salience else np.array([-1, -1, -1, -1])

        noise_image, noise_mask = F.bbox_salience_salt_pepper_noise(image, salience_area, self.n, self.color)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.noise_scale, self.noise_scale))
        noise_mask = cv2.dilate(noise_mask, kernel)
        noise_mask = cv2.rectangle(noise_mask, (0, 0), (w, h), 0, int(min(h, w) * self.border_scale))
        noise_image[noise_mask > 0] = self.color

        return noise_image

    def get_transform_init_args_names(self):
        return "color", "n", "noise_scale", "border_scale", "saline"


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
            self.target_dependence["bboxes"] = {"bboxes_cache": kwargs["bboxes_cache"],
                                                "classes_cache": kwargs['classes_cache']}

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
            classes_cache=None,
            height=0,
            width=0,
            **params
    ):
        new_bboxes = []
        padh1, padw1 = self.padh_cache.pop(-1), self.padw_cache.pop(-1)

        for bbox, classes, padh, padw in zip(bboxes_cache, classes_cache, self.padh_cache, self.padw_cache):
            for box, cls in zip(bbox, classes):
                new_bbox = self.apply_to_bbox(tuple(tuple(box) + tuple([cls])), padh, padw, height, width)
                new_bbox = normalize_bbox(new_bbox, height, width)
                new_bboxes.append(new_bbox)

        for box in bboxes:
            box = denormalize_bbox(box, params['rows'], params['cols'])
            new_bbox = self.apply_to_bbox(box, padh1, padw1, height, width)
            new_bbox = normalize_bbox(new_bbox, height, width)
            new_bboxes.append(new_bbox)

        return new_bboxes

    def apply_to_keypoint(self, **params):
        pass  # TODO

    def get_params(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "fill_value": self.fill_value,
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "height", "width", "fill_value", "bboxes_cache_format"

# if __name__ == '__main__':
#     image = io.imread(r"D:\cgm\dataset\VOC2007\JPEGImages\000005.jpg")
#     print(image.shape)
#     for _ in range(5):
#         x0, y0, x1, y1 = 25, 12, 430, 310
#
#         var = SaltPepperNoise(p=1, salience=True, border_scale=0)(image=image,
#                                                                   bboxes=np.array([[x0, y0, x1, y1]], dtype=float))
#         var1 = cv2.rectangle(var['image'].copy(),
#                              var['bboxes'][0, [0, 1]].astype(int),
#                              var['bboxes'][0, [2, 3]].astype(int),
#                              (255, 255, 0), 1)
#         print(var1.shape)
#         io.show('ad', var1)
