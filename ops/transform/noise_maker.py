import random
import cv2
import numpy as np
from ops.transform.basic_transform import SalienceTransform
from typing import List
import ops.cv.io as io


def _salience_salt_pepper_noise(image, salience_area, n, color):
    x0, y0, x1, y1 = salience_area
    h, w = image.shape[:2]
    noise_mask = np.zeros((h, w))
    noise_image = image.copy()
    for i in range(n):
        x = np.random.randint(1, w)
        y = np.random.randint(1, h)
        # bbox 区域类，不做任何操作
        while (x0 <= x <= x1) and (y0 <= y <= y1):
            x = np.random.randint(1, w)
            y = np.random.randint(1, h)
        noise_mask[y, x] = color
    return noise_image, noise_mask


class SaltPepperNoise(SalienceTransform):
    def __init__(self,
                 color: int = 180,
                 n: int = 1000,
                 noise_scale: int = 5,
                 border_scale: float = 0.15,
                 saline: bool = False,
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

        super(SaltPepperNoise, self).__init__(saline, p)

        self.noise_scale = noise_scale
        self.border_scale = border_scale
        self.n = n
        self.color = color

    def __call__(self,
                 image: np.ndarray,
                 mask: np.ndarray = None,
                 bbox_params: np.ndarray = None):
        """

        :param image: np.array[h,w,3] 增强图片
        :param mask: np.array[h,w] 增强 mask
        :param bbox_params: np.array[[x0,y0,x1,y1],...] 增强 bbox
        :return:
        """
        image, mask, bbox_params = super(SalienceTransform, self).__call__(image, mask, bbox_params)

        if self.salience and mask is None and bbox_params is None:
            raise ValueError('salience 为 True 的时候，至少要有 mask 或 bbox_params')

        if random.random() <= self.p:
            image = self.apply(image, bbox_params)

        return {"image": image,
                "mask": mask,
                "bbox_params": bbox_params}

    def apply(self,
              image: np.ndarray,
              bbox_params=None):

        h, w = image.shape[:2]
        salience_area = bbox_params.salience_area() if self.salience else np.array([-1, -1, -1, -1])

        noise_image, noise_mask = _salience_salt_pepper_noise(image, salience_area, self.n, self.color)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.noise_scale, self.noise_scale))
        noise_mask = cv2.dilate(noise_mask, kernel)
        noise_mask = cv2.rectangle(noise_mask, (0, 0), (w, h), 0, int(min(h, w) * self.border_scale))
        noise_image[noise_mask > 0] = self.color

        return noise_image


if __name__ == '__main__':
    image = io.imread(r"D:\cgm\dataset\VOC2007\JPEGImages\000005.jpg")
    print(image.shape)
    for _ in range(5):
        x0, y0, x1, y1 = 25, 12, 430, 310

        var = SaltPepperNoise(p=1, saline=False, border_scale=0)(image,
                                                                 bbox_params=np.array([[x0, y0, x1, y1]], dtype=float))
        var1 = cv2.rectangle(var['image'].copy(),
                             var['bbox_params'][0, [0, 1]].astype(int),
                             var['bbox_params'][0, [2, 3]].astype(int),
                             (255, 255, 0), 1)
        print(var1.shape)
        io.show_window('ad', var1)
