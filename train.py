import argparse
from ops.utils.logging import print_args
from ops.models.detection.yolo.trainer import Yolo

if __name__ == '__main__':
    model = Yolo("./cfg/models/yolo/v5/yolov5.yaml")
    model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], batch=5, box=0.05)
