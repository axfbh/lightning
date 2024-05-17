import cv2
import torch
from skimage import io
from torchvision.ops import box_convert
import importlib
from matplotlib import pyplot as plt
import matplotlib


def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_FREERATIO)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def imread(path):
    if importlib.import_module('jpeg4py'):
        import jpeg4py as jpeg
        img = jpeg.JPEG(path).decode()
    else:
        img = io.imread(path)
    return img


def visualize(image, bboxes, category_ids, category_id_to_name=None, in_fmt='xyxy'):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        if category_id_to_name is not None:
            class_name = category_id_to_name[category_id]
        else:
            class_name = str(category_id)

        bbox = box_convert(torch.as_tensor(bbox), in_fmt=in_fmt, out_fmt='xyxy').int().numpy()
        img = visualize_bbox(img, bbox, class_name)
    show('ImageWithBox', img)


def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    test_color = (255, 255, 255)  # White

    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=test_color,
        lineType=cv2.LINE_AA,
    )
    return img
