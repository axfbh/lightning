import numpy as np
import random
import cv2
from ops.transform.basic_transform import SalienceTransform
from ops.transform.resize_maker import Resize
import ops.cv.io as io


def _valid_up_size(image, salience_area, scale):
    h, w = image.shape[:-1]

    x0, y0, x1, y1 = salience_area

    # ----------- 长宽最少可以放大的距离 -------------
    gap = min(min(y0 - 0, h - y1), min(x0 - 0, w - x1))

    scale = int(gap * scale)

    ratio = random.randint(scale // 2, scale)

    ratio = ratio if ratio % 2 == 0 else ratio - 1

    return h + ratio, w + ratio


class ObjectRandomUp(SalienceTransform):
    def __init__(self,
                 scale: float = 0.3,
                 p: float = 0.5):
        """
        目标放大
        :param scale: 缩放比
        :return:
        """
        super(ObjectRandomUp, self).__init__(True, p)

        self.scale = scale

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

        if mask is None and bboxes is None:
            raise ValueError('使用 salience ，至少要有 mask 或 bboxes')

        salience_area = bboxes.salience_area()

        if random.random() <= self.p:
            h, w = image.shape[:-1]

            scale_h, scale_w = _valid_up_size(image, salience_area, self.scale)
            resize_info = Resize([scale_h, scale_w])(image, mask, bboxes)
            image = resize_info['image']
            mask = resize_info['mask']
            bboxes = resize_info['bboxes']
            image = self.apply(image, (h, w), (scale_h, scale_w))
            if bboxes is not None:
                bboxes = self.apply_to_bbox(bboxes, (h, w), (scale_h, scale_w))

        return {"image": image,
                "mask": mask,
                "bboxes": bboxes}

    def apply(self, image: np.ndarray, original_size, new_size):
        image = image.copy()
        cut_h = (new_size[0] - original_size[0]) // 2
        cut_w = (new_size[1] - original_size[1]) // 2

        if cut_h == 0 and cut_w == 0:
            return image

        if cut_h != 0:
            image = image[cut_h:-cut_h, :]

        if cut_w != 0:
            image = image[:, cut_w:-cut_w]

        return image

    def apply_to_bbox(self, bboxes, original_size, new_size):
        bboxes = bboxes.copy()
        cut_h = (new_size[0] - original_size[0]) // 2
        cut_w = (new_size[1] - original_size[1]) // 2
        bboxes = bboxes - np.array([cut_w, cut_h, 0, 0])
        return bboxes

    def apply_to_mask(self, mask, original_size, new_size):
        mask = mask.copy()
        cut_h = (new_size[0] - original_size[0]) // 2
        cut_w = (new_size[1] - original_size[1]) // 2

        if cut_h == 0 and cut_w == 0:
            return mask

        if cut_h != 0:
            mask = mask[cut_h:-cut_h, :]

        if cut_w != 0:
            mask = mask[:, cut_w:-cut_w]

        return mask

#
# if __name__ == '__main__':
#     image = io.imread(r"D:\cgm\dataset\VOC2007\JPEGImages\000005.jpg")
#     print(image.shape)
#     for _ in range(3):
#         x0, y0, x1, y1 = 25, 12, 430, 310
#
#         var = ObjectRandomUp(scale=0.9, p=1)(image, bboxes=np.array([[x0, y0, x1, y1]], dtype=float))
#         var1 = cv2.rectangle(var['image'].copy(),
#                              tuple(var['bboxes'][0, [0, 1]].astype(int)),
#                              tuple(var['bboxes'][0, [2, 3]].astype(int)),
#                              (255, 255, 0), 1)
#         print(var1.shape)
#         io.show('ad', var1)
