import numpy as np


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
