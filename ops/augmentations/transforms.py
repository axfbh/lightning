from typing import Dict, Any, Tuple, Callable, Union
import random
import cv2
import numpy as np

from albumentations import ImageOnlyTransform, DualTransform, BasicTransform, BaseCompose, Compose, Crop, BboxParams
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox

import ops.augmentations.functional as F
from ops.augmentations.geometric.transforms import RandomShiftScaleRotate
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
            raise KeyError("You have to pass cfg to augmentations as named arguments, for example: aug(image=image)")
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


class _Mosaic(DualTransform):
    """
        Mosaic augmentation arranges randomly selected four images into single one like the 2x2 grid layout.
    """

    def __init__(
            self,
            height,
            width,
            read_fn: Union[BasicTransform, BaseCompose, Callable],
            reference_data,
            value=0,
            mask_value=0,
            always_apply=False,
            p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        if len(reference_data) < 3:
            raise ValueError('Mosaic must transform with 4 images.')

        self.read_fn = read_fn
        self.reference_data = reference_data
        self.height = height
        self.width = width
        self.value = value
        self.mask_value = mask_value

    def apply(self,
              image,
              mosaic_data=None,
              x_center=0,
              y_center=0,
              height=0,
              width=0,
              **params):
        image_cache = [data['image'] for data in mosaic_data]
        image_cache.append(image)
        image, self.padh_cache, self.padw_cache = F.mosaic4(image_cache, x_center, y_center, height, width, self.value)
        return image

    def apply_to_mask(self,
                      masks,
                      mosaic_data=None,
                      x_center=0,
                      y_center=0,
                      height=0,
                      width=0,
                      **params):
        # TODO
        mask_cache = [data['mask'] for data in mosaic_data]
        mask_cache.append(masks)
        mask, *_ = F.mask_mosaic4(mask_cache, x_center, y_center, height, width, self.mask_value)
        return mask

    def apply_to_bbox(self, bbox, padh=0, padw=0, height=0, width=0, **params):
        return F.bbox_mosaic4(bbox, padh, padw, height, width)

    def apply_to_bboxes(self,
                        bboxes,
                        mosaic_data=None,
                        height=0,
                        width=0,
                        **params
                        ):
        new_bboxes = []
        padh1, padw1 = self.padh_cache.pop(-1), self.padw_cache.pop(-1)

        for data, padh, padw in zip(mosaic_data, self.padh_cache, self.padw_cache):
            bbox, classes = data['bboxes'], data['classes']
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

    def get_params(self) -> Dict[str, Union[None, float, Dict[str, Any]]]:
        mosaic_idx = random.choices(range(len(self.reference_data)), k=3)

        mosaic_data = [self.read_fn(**self.reference_data[i]) for i in mosaic_idx]

        return {"mosaic_data": mosaic_data,
                'x_center': int(random.uniform(self.height // 2, self.height - self.height // 2)),
                'y_center': int(random.uniform(self.width // 2, self.width - self.width // 2)),
                "height": self.height * 2,
                "width": self.width * 2}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "reference_data", "height", "width", "fill_value"


class Mosaic:
    def __init__(
            self,
            height,
            width,
            read_fn: Union[BasicTransform, BaseCompose, Callable],
            reference_data,
            scale=0.5,
            translate=0.1,
            value=0,
            mask_value=0,
            bbox_params=BboxParams(format='pascal_voc', label_fields=['classes']),
            p=0.5):
        self._mosaic = Compose([
            _Mosaic(
                height=height,
                width=width,
                read_fn=read_fn,
                reference_data=reference_data,
                value=value,
                mask_value=mask_value,
                always_apply=True),
            RandomShiftScaleRotate(
                scale_limit=(1 - scale, 1 + scale),
                shift_limit_x=(0.5 - translate, 0.5 + translate),
                shift_limit_y=(0.5 - translate, 0.5 + translate),
                rotate_limit=(0, 0),
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                mask_value=mask_value,
                position=RandomShiftScaleRotate.PositionType.TOP_LEFT,
                always_apply=True),
            Crop(x_max=width,
                 y_max=height,
                 always_apply=True),
        ], bbox_params, p=p)

    def __call__(self, *args, **kwargs):
        return self._mosaic(*args, **kwargs)
