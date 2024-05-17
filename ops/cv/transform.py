import cv2


def rot_point(point, image, angle):
    h, w = image.shape[:2]
    center = (w / 2 - 0.5, h / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    return cv2.transform(point, M)


