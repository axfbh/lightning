import cv2
import numpy as np
from typing import Sequence, Tuple, List
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox


def shift_scale_rotate_matrix(
        center: Sequence,
        angle: float,
        scale: float,
        dx: int,
        dy: int,
        rows: int,
        cols: int,
) -> np.ndarray:
    height, width = rows, cols
    C = np.eye(3)
    C[0, 2] = -width / 2  # x translation (pixels)
    C[1, 2] = -height / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(center, angle, scale)

    # Translation
    T = np.eye(3)
    T[0, 2] = dx * width  # x translation (pixels)
    T[1, 2] = dy * height  # y translation (pixels)

    M = T @ C @ R
    return M


def bbox_shift_scale_rotate(
        bbox,
        center: Sequence,
        angle: float,
        scale: float,
        dx: int,
        dy: int,
        rows: int,
        cols: int
) -> Tuple:
    height, width = rows, cols
    matrix = shift_scale_rotate_matrix(center, angle, scale, dx, dy, height, width)
    x_min, y_min, x_max, y_max = bbox[:4]
    x = np.array([x_min, x_max, x_max, x_min])
    y = np.array([y_min, y_min, y_max, y_max])
    ones = np.ones(shape=(len(x)))
    points_ones = np.vstack([x, y, ones]).transpose()
    points_ones[:, 0] *= width
    points_ones[:, 1] *= height
    tr_points = matrix.dot(points_ones.T).T
    tr_points[:, 0] /= width
    tr_points[:, 1] /= height

    x_min, x_max = min(tr_points[:, 0]), max(tr_points[:, 0])
    y_min, y_max = min(tr_points[:, 1]), max(tr_points[:, 1])

    return x_min, y_min, x_max, y_max


def resize_boxes_ratio(boxes, ratio_height, ratio_width):
    xmin, ymin, xmax, ymax = boxes

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return xmin, ymin, xmax, ymax


def resize_boxes(boxes, original_size, new_size):
    ratio_height = new_size[0] / original_size[0]
    ratio_width = new_size[1] / original_size[1]
    return resize_boxes_ratio(boxes, ratio_height, ratio_width)


def mosaic4(image_batch: List[np.ndarray], height: int, width: int, fill_value: int = 0) -> np.ndarray:
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

    if len(image_batch[0].shape) == 2:
        out_shape = [height, width]
    else:
        out_shape = [height, width, image_batch[0].shape[2]]

    center_x = width // 2
    center_y = height // 2
    img4 = np.full(out_shape, fill_value, dtype=np.uint8)  # base image with 4 tiles
    for i, img in enumerate(image_batch):
        (h, w) = img.shape[:2]

        # place img in img4
        # this based on the yolo5's implementation
        #
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = (
                max(center_x - w, 0),
                max(center_y - h, 0),
                center_x,
                center_y,
            )  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = (
                center_x,
                max(center_y - h, 0),
                min(center_x + w, width),
                center_y,
            )
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = (
                max(center_x - w, 0),
                center_y,
                center_x,
                min(height, center_y + h),
            )
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = (
                center_x,
                center_y,
                min(center_x + w, width),
                min(height, center_y + h),
            )
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

    return img4


def bbox_mosaic4(bbox: Tuple, rows: int, cols: int, position_index: int, height: int, width: int):
    """Put the given bbox in one of the cells of the 2x2 grid.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)` or `(x_min, y_min, x_max, y_max, label, ...)`.
        rows (int): Height of input image that corresponds to one of the mosaic cells
        cols (int): Width of input image that corresponds to one of the mosaic cells
        position_index (int): Index of the mosaic cell. 0: top left, 1: top right, 2: bottom left, 3: bottom right
        height (int): Height of output mosaic image
        width (int): Width of output mosaic image
    """
    bbox = denormalize_bbox(bbox, rows, cols)
    bbox, tail = bbox[:4], tuple(bbox[4:])
    center_x = width // 2
    center_y = height // 2
    if position_index == 0:  # top left
        shift_x = center_x - cols
        shift_y = center_y - rows
    elif position_index == 1:  # top right
        shift_x = center_x
        shift_y = center_y - rows
    elif position_index == 2:  # bottom left
        shift_x = center_x - cols
        shift_y = center_y
    elif position_index == 3:  # bottom right
        shift_x = center_x
        shift_y = center_y
    bbox = (
        bbox[0] + shift_x,
        bbox[1] + shift_y,
        bbox[2] + shift_x,
        bbox[3] + shift_y,
    )

    # bbox = normalize_bbox(bbox, height, width)
    return tuple(bbox + tail)
