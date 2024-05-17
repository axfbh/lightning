import numpy as np
from ops.transform.basic_transform import DualTransform
import cv2


class PaddingImage(DualTransform):
    def __init__(self, pad_l: int, pad_t: int, pad_r: int, pad_b: int, color=(114, 114, 114)):
        super(PaddingImage, self).__init__()
        self.color = color
        self.pad_b = pad_b
        self.pad_r = pad_r
        self.pad_t = pad_t
        self.pad_l = pad_l

    def __call__(self,
                 image: np.ndarray,
                 mask: np.ndarray = None,
                 bbox_params: np.ndarray = None):

        image = self.apply(image)
        if bbox_params is not None:
            bbox_params = self.apply_to_bbox(bbox_params)
        if mask is not None:
            mask = self.apply_to_mask(mask)

        return {"image": image,
                "mask": mask,
                "bbox_params": bbox_params,
                "pad": (self.pad_l, self.pad_t, self.pad_r, self.pad_b)}

    def apply(self, image: np.ndarray):
        image = image.copy()

        image = cv2.copyMakeBorder(image, self.pad_t, self.pad_b, self.pad_l, self.pad_r, cv2.BORDER_CONSTANT,
                                   value=self.color)

        return image

    def apply_to_bbox(self, bbox_params):
        bbox_params = bbox_params + np.array([self.pad_l, self.pad_t, self.pad_l, self.pad_t])
        return bbox_params

    def apply_to_mask(self, mask):
        mask = mask.copy()
        mask = np.pad(mask, ((self.pad_t, self.pad_b), (self.pad_l, self.pad_r)))
        return mask
