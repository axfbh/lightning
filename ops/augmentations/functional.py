import cv2
import numpy as np
from typing import Sequence, Tuple


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
