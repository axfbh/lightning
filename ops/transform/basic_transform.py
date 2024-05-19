import cv2
from typing import List
import random
import numpy as np


def _mask_to_bbox(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # 坐标的时候表示 x,y,w,h. 表格的时候表示 y,x,h,w
        x0 = y
        x1 = y + h
        y0 = x
        y1 = x + w
        bboxes.append(np.array([x0, y0, x1, y1], dtype=float))
    bboxes = np.array(bboxes, dtype=float)
    return bboxes


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
                 bboxes: np.ndarray = None):
        """

        :param image: np.array[h,w,3] 增强图片
        :param mask: np.array[h,w] 增强 mask
        :param bboxes: np.array[[x0,y0,x1,y1],...] 增强 bbox
        :return:
        """

        # ------------ 判断 bboxes 是否满足要求 -------------
        if bboxes is not None:
            if isinstance(bboxes, float):
                raise TypeError('请输入 bboxes 的类型必须是 float numpy 数组')

            if bboxes.ndim > 2 and bboxes[0].shape[1] == 4:
                raise TypeError('请输入 bboxes 数组内的 shape 必须是 2维，且每个维度包含 4 个参数')

        if isinstance(mask, np.uint8) and mask is not None and mask.ndim > 2:
            raise TypeError('请输入 mask 的类型必须是 uint8 numpy 数组,并且 shape 必须是 2 维')

        # -------------- 如果 mask Not None，但是 bboxes 是 None ，则 mask 生成 bboxes --------------
        if mask is not None and bboxes is None:
            bboxes, xywh = _mask_to_bbox(mask)
            bboxes = np.array(bboxes, dtype=float).view(BboxParams)
        elif bboxes is not None:
            bboxes = bboxes.view(BboxParams)

        return image, mask, bboxes

    def apply(self, *args) -> np.ndarray:
        pass

    def apply_to_bbox(self, *args):
        pass

    def apply_to_mask(self, *args):
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
                 bboxes: np.ndarray = None):
        """

        :param image: np.array[h,w,3] 增强图片
        :param mask: np.array[h,w] 增强 mask
        :param bboxes: np.array[[x0,y0,x1,y1],...] 增强 bbox
        :return:
        """
        image, mask, bboxes = super(SalienceTransform, self).__call__(image, mask, bboxes)

        if self.salience and mask is None and bboxes is None:
            raise ValueError('salience 为 True 的时候，至少要有 mask 或 bboxes')

        if random.random() <= self.p:
            return self.apply(image, mask, bboxes)

        return {"image": image,
                "mask": mask,
                "bboxes": bboxes}
