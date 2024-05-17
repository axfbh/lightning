import cv2
from typing import List
import random
import numpy as np


def _mask_to_bbox_params(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bbox_params = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # 坐标的时候表示 x,y,w,h. 表格的时候表示 y,x,h,w
        x0 = y
        x1 = y + h
        y0 = x
        y1 = x + w
        bbox_params.append(np.array([x0, y0, x1, y1], dtype=float))
    bbox_params = np.array(bbox_params, dtype=float)
    return bbox_params


class BboxParams(np.ndarray):

    def get_xywh(self):
        xywh = self.copy()
        xywh[:, 2:] = xywh[:, 2:] - xywh[:, :2]
        return xywh

    def get_x0y0x1y1(self):
        x0y0x1y1 = self.copy()
        x0y0x1y1[:, 2:] = x0y0x1y1[:, 2:] + x0y0x1y1[:, :2]
        return x0y0x1y1

    def salience_area(self):
        xmin, ymin = self.min(0)[[0, 1]]
        xmax, ymax = self.max(0)[[2, 3]]
        return np.array([xmin, ymin, xmax, ymax], dtype=float)


class BasicTransform:
    def __init__(self,
                 p: float = 0.5):
        """

        :param p: 触发概率
        """
        self.p = p

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

        # ------------ 判断 bbox_params 是否满足要求 -------------
        if bbox_params is not None:
            if isinstance(bbox_params, float):
                raise TypeError('请输入 bbox_params 的类型必须是 float numpy 数组')

            if bbox_params.ndim > 2 and bbox_params[0].shape[1] == 4:
                raise TypeError('请输入 bbox_params 数组内的 shape 必须是 2维，且每个维度包含 4 个参数')

        if isinstance(mask, np.uint8) and mask is not None and mask.ndim > 2:
            raise TypeError('请输入 mask 的类型必须是 uint8 numpy 数组,并且 shape 必须是 2 维')

        # -------------- 如果 mask Not None，但是 bbox_params 是 None ，则 mask 生成 bbox_params --------------
        if mask is not None and bbox_params is None:
            bbox_params, xywh = _mask_to_bbox_params(mask)
            bbox_params = np.array(bbox_params, dtype=float).view(BboxParams)
        elif bbox_params is not None:
            bbox_params = bbox_params.view(BboxParams)

        return image, mask, bbox_params

    def apply(self, *args) -> np.ndarray:
        pass

    def apply_to_bbox(self, *args):
        pass

    def apply_to_mask(self, *args):
        pass

    def get_params_dependent_on_targets(self, *args):
        pass


class DualTransform(BasicTransform):
    def __init__(self):
        """

        :param p: 触发概率
        """
        super(DualTransform, self).__init__(1)


class SalienceTransform(BasicTransform):
    def __init__(self,
                 salience: bool,
                 p: float = 0.5):
        """

        :param salience: 增强后是否保留完整的 mask 黑 bbox 信息
        :param p: 触发概率
        """
        super(SalienceTransform, self).__init__(p)
        self.salience = salience

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
            return self.apply(image, mask, bbox_params)

        return {"image": image,
                "mask": mask,
                "bbox_params": bbox_params}
