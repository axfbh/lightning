import numpy as np
from typing import Tuple, List
import random


def bbox_salience_salt_pepper_noise(image, salience_area, n, color):
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


def bbox_salience_area(bboxes: np.ndarray):
    xmin, ymin = bboxes[:, :2].min(0)
    xmax, ymax = bboxes[:, 2:].max(0)
    return xmin, ymin, xmax, ymax


def mosaic4(image_batch: List[np.ndarray], height: int, width: int, fill_value: int = 0):
    """Arrange the images in a 2x2 grid. Images can have different shape.
    This implementation is based on YOLOv5 with some modification:
    https://github.com/ultralytics/yolov5/blob/932dc78496ca532a41780335468589ad7f0147f7/utils/datasets.py#L648

    Args:
        image_batch (List[np.ndarray]): image list. The length should be 4.
        height (int): Height of output mosaic image
        width (int): Width of output mosaic image
        fill_value (int): padding value

    """
    if len(image_batch) != 4:
        raise ValueError(f"Length of image_batch should be 4. Got {len(image_batch)}")

    mosaic_border = [-height // 4, -width // 4]
    yc, xc = (int(random.uniform(-x, y + x)) for x, y in zip(mosaic_border, [height, width]))
    img4 = np.full((height, width, 3), fill_value, dtype=np.uint8)  # base image with 4 tiles
    padw_cache = []
    padh_cache = []
    for i, img in enumerate(image_batch):
        (h, w) = img.shape[:2]

        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, width), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(height, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, width), min(height, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw_cache.append(x1a - x1b)
        padh_cache.append(y1a - y1b)

    return img4, padh_cache, padw_cache


def bbox_mosaic4(bbox: Tuple, padh: int, padw: int, height: int, width: int):
    """Put the given bbox in one of the cells of the 2x2 grid.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)` or `(x_min, y_min, x_max, y_max, label, ...)`.
        rows (int): Height of input image that corresponds to one of the mosaic cells
        cols (int): Width of input image that corresponds to one of the mosaic cells
        position_index (int): Index of the mosaic cell. 0: top left, 1: top right, 2: bottom left, 3: bottom right
        height (int): Height of output mosaic image
        width (int): Width of output mosaic image
    """
    # bbox = denormalize_bbox(bbox, height, width)
    bbox, tail = bbox[:4], tuple(bbox[4:])
    bbox = (
        max(bbox[0] + padw, 0),
        max(bbox[1] + padh, 0),
        min(bbox[2] + padw, width),
        min(bbox[3] + padh, height),
    )
    return tuple(bbox + tail)
