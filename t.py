import time

import numpy as np
import cv2
import random
import math

a = random.uniform(-0, 0)
s = random.uniform(1 - 0.5, 1 + 0.5)


def affine(shape=(640, 640), translate=0.1, border=(0, 0)):
    height = shape[0] + border[0] * 2  # shape(h,w,c)
    width = shape[1] + border[1] * 2

    C = np.eye(3)
    C[0, 2] = -shape[1] / 2  # x translation (pixels)
    C[1, 2] = -shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ C @ R  # order of operations (right to left) is IMPORTANT
    print(M[:2])


def shift_scale_rotate(shape=(640, 640), translate_x=(-0.1, 0.0629), translate_y=(-0.1, 0.0629)):
    height, width = shape

    C = np.eye(3)
    C[0, 2] = -shape[1] / 2  # x translation (pixels)
    C[1, 2] = -shape[0] / 2  # y translation (pixels)
    center = (-width / 2 - 0.5, -height / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, a, s)
    # matrix[0, 2] += random.uniform(*translate_x) * width  # x
    # matrix[1, 2] += random.uniform(*translate_y) * height  # y
    print(matrix)


if __name__ == '__main__':
    for _ in range(100):
        affine()
        shift_scale_rotate()
        print(end='\n')
        time.sleep(1)
