from typing import Dict, Any

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import ops.augmentations.functional as F


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
        salience_area = F.cal_salience_area(bboxes) if self.salience else np.array([-1, -1, -1, -1])

        noise_image, noise_mask = F.cal_salience_salt_pepper_noise(image, salience_area, self.n, self.color)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.noise_scale, self.noise_scale))
        noise_mask = cv2.dilate(noise_mask, kernel)
        noise_mask = cv2.rectangle(noise_mask, (0, 0), (w, h), 0, int(min(h, w) * self.border_scale))
        noise_image[noise_mask > 0] = self.color

        return noise_image

    def get_transform_init_args_names(self):
        return "color", "n", "noise_scale", "border_scale", "saline"

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
