from typing import Dict, Any, Tuple, Callable, Union
import random
import cv2
import numpy as np

from albumentations import ImageOnlyTransform, DualTransform, BasicTransform, BaseCompose
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
    """
        Mosaic augmentation arranges randomly selected four images into single one like the 2x2 grid layout.
    """

    def __init__(
            self,
            height,
            width,
            read_fn: Union[BasicTransform, BaseCompose, Callable],
            reference_data,
            fill_value=0,
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
        self.fill_value = fill_value

    def apply(self,
              image,
              mosaic_data=None,
              x_center=0,
              y_center=0,
              height=0,
              width=0,
              fill_value=114,
              **params):
        image_cache = [data['image'] for data in mosaic_data]
        image_cache.append(image)
        image, self.padh_cache, self.padw_cache = F.mosaic4(image_cache, x_center, y_center, height, width, fill_value)
        return image

    def apply_to_masks(self,
                       masks,
                       mosaic_data=None,
                       x_center=0,
                       y_center=0,
                       height=0,
                       width=0,
                       fill_value=114,
                       **params):
        # TODO
        mask_cache = [data['mask'] for data in mosaic_data]
        mask_cache.append(masks)
        return F.mosaic4(mask_cache, x_center, y_center, height, width, fill_value)

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
                'x_center': int(random.uniform(self.height // 4, self.height - self.height // 4)),
                'y_center': int(random.uniform(self.width // 4, self.width - self.width // 4)),
                "height": self.height,
                "width": self.width,
                "fill_value": self.fill_value}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "reference_data", "height", "width", "fill_value"

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
