import random
import numpy as np
from ops.transform.basic_transform import DualTransform
import cv2
from typing import List
import ops.cv.io as io
import torch
from ops.transform.pad_maker import PaddingImage


def resize_boxes(boxes, original_size, new_size):
    ratios = [torch.tensor(s, dtype=torch.float32) / torch.tensor(s_orig, dtype=torch.float32)
              for s, s_orig in zip(new_size, original_size)]

    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = torch.from_numpy(boxes).unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1).numpy()


class ResizeBasicTransform(DualTransform):
    def __call__(self,
                 image: np.ndarray,
                 mask: np.ndarray = None,
                 bbox_params: np.ndarray = None):

        image, mask, bbox_params = super(DualTransform, self).__call__(image, mask, bbox_params)

        h, w = image.shape[:-1]
        image = self.apply(image)
        if bbox_params is not None:
            bbox_params = self.apply_to_bbox(bbox_params, (h, w), image.shape[:-1])
        if mask is not None:
            mask = self.apply_to_mask(mask, image.shape[:-1])

        return {"image": image,
                "mask": mask,
                "bbox_params": bbox_params}


class Resize(ResizeBasicTransform):
    def __init__(self, image_size: List[int]):
        super(Resize, self).__init__()
        self.h, self.w = image_size

    def apply(self, image: np.ndarray):
        resize_image = cv2.resize(image, (self.w, self.h))
        return resize_image

    def apply_to_bbox(self, bbox_params,
                      original_size,
                      new_size):
        bbox_params = resize_boxes(bbox_params, original_size, new_size)
        return bbox_params

    def apply_to_mask(self, mask, new_size):
        mask = cv2.resize(mask, new_size)
        return mask


class ResizeShortLongest(ResizeBasicTransform):
    def __init__(self, image_size: List[int]):
        super(ResizeShortLongest, self).__init__()
        self.min_size, self.max_size = image_size

    def apply(self, image: np.ndarray):
        im_shape = image.shape[:-1]
        min_size = min(im_shape)
        max_size = max(im_shape)
        ratio = round(min(self.min_size / min_size, self.max_size / max_size), 5)
        resize_image = cv2.resize(image, None, fx=ratio, fy=ratio)

        return resize_image

    def apply_to_bbox(self, bbox_params,
                      original_size,
                      new_size):
        bbox_params = resize_boxes(bbox_params, original_size, new_size)
        return bbox_params

    def apply_to_mask(self, mask, new_size):
        mask = cv2.resize(mask, new_size)
        return mask


class ResizeLongestPaddingShort(DualTransform):
    def __init__(self, image_size: List[int], shuffle: bool, color=(114, 114, 114)):
        """
        填充边界，防止图像缩放变形，基于短边
        :param shuffle: True 随机填充边界, False 对半填充边界

        :return:
        """
        super(ResizeLongestPaddingShort, self).__init__()
        self.image_size = image_size
        self.shuffle = shuffle
        self.color = color

    def __call__(self,
                 image: np.ndarray,
                 mask: np.ndarray = None,
                 bbox_params: np.ndarray = None):
        resize_info = ResizeShortLongest(self.image_size)(image=image,
                                                          mask=mask,
                                                          bbox_params=bbox_params)
        image = resize_info['image']
        mask = resize_info['mask']
        bbox_params = resize_info['bbox_params']

        h, w = image.shape[:2]

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

        return PaddingImage(self.pad_l, self.pad_t, self.pad_r, self.pad_b, self.color)(image=image,
                                                                                        mask=mask,
                                                                                        bbox_params=bbox_params)


if __name__ == '__main__':
    image = io.imread(r"D:\cgm\dataset\VOC2007\JPEGImages\000005.jpg")
    print(image.shape)
    for _ in range(3):
        x0, y0, x1, y1 = 25, 12, 430, 310
        var = ResizeLongestPaddingShort(image_size=[416, 600], shuffle=True)(image,
                                                                             bbox_params=np.array([[x0, y0, x1, y1]],
                                                                                                  dtype=float))
        var1 = cv2.rectangle(var['image'].copy(),
                             tuple(var['bbox_params'][0, [0, 1]].astype(int)),
                             tuple(var['bbox_params'][0, [2, 3]].astype(int)),
                             (255, 255, 0), 1)
        print(var1.shape)
        io.show_window('ad', var1)
