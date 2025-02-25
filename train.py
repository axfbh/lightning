import random
import numpy as np
import torch
from ops.models.detection.yolo.trainer import Yolo
from ops.models.detection.detr.trainer import Detr
from ops.utils.torch_utils import init_seeds

if __name__ == '__main__':
    # model = Yolo("yolov5n.yaml")
    # model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], epochs=220, batch=8, box=0.05)

    # yolov4
    # model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], epochs=220, batch=8, box=0.15, cls=1.5, obj=3)

    init_seeds(2)
    model = Detr("deformable_detr.yaml")
    model.train(data="./cfg/datasets/coco.yaml",
                imgsz=[640, 640],
                epochs=300,
                batch=4,
                weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2})
