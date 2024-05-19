import numpy as np

import cv2
from albumentations import DualTransform
from typing import Any, Dict, Tuple


class PaddingImage(DualTransform):
    def __init__(self, pad_l: int, pad_t: int, pad_r: int, pad_b: int, color=(114, 114, 114),
                 always_apply=False,
                 p=0.5):
        super(PaddingImage, self).__init__(always_apply, p)

        self.color = color
        self.pad_b = pad_b
        self.pad_r = pad_r
        self.pad_t = pad_t
        self.pad_l = pad_l

    def apply(self, img: np.ndarray, color, pad_l, pad_t, pad_r, pad_b, **params) -> np.ndarray:
        img = cv2.copyMakeBorder(img,
                                 pad_t,
                                 pad_b,
                                 pad_l,
                                 pad_r,
                                 cv2.BORDER_CONSTANT,
                                 value=color)

        return img

    def apply_to_bbox(self, bbox, pad_l, pad_t, pad_r, pad_b, **params) -> Tuple:
        return tuple([v1 + v2 for v1, v2 in zip(bbox, [pad_l, pad_t, pad_r, pad_b])])

    def apply_to_mask(self, mask, pad_l, pad_t, pad_r, pad_b, **params) -> np.ndarray:
        mask = np.pad(mask, ((pad_t, pad_b), (pad_l, pad_r)))
        return mask

    def get_params(self) -> Dict[str, Any]:
        return {
            "color": self.color,
            "pad_b": self.pad_b,
            "pad_r": self.pad_r,
            "pad_t": self.pad_t,
            "pad_l": self.pad_l,
        }

    def get_transform_init_args_names(self):
        return (
            "color",
            "pad_l",
            "pad_t",
            "pad_r",
            "pad_b",
        )
