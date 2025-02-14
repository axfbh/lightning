import random
import numpy as np

from ops.models.detection.yolo.trainer import Yolo
from ops.models.detection.detr.trainer import Detr
from ops.utils.torch_utils import init_seeds


if __name__ == '__main__':
    # model = Yolo("yolov5n.yaml")
    # model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], epochs=220, batch=8, box=0.05)

    # yolov4
    # model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], epochs=220, batch=8, box=0.15, cls=1.5, obj=3)

    init_seeds(1)
    model = Detr("detrv1.yaml")
    model.train(data="./cfg/datasets/coco.yaml", imgsz=[640, 640], epochs=100, batch=8)
