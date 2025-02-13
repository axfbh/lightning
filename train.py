from ops.models.detection.yolo.trainer import Yolo
from ops.models.detection.detr.trainer import Detr
import torch
import random
import numpy as np


def set_seed(seed):
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    # 设置 CUDA 的随机种子（如果使用 GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
        torch.backends.cudnn.deterministic = True  # 确保 CUDA 卷积操作是确定性的
        torch.backends.cudnn.benchmark = False  # 关闭 CUDA 的自动优化
    # 设置 Python 的随机种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)


if __name__ == '__main__':
    # model = Yolo("yolov5n.yaml")
    # model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], epochs=220, batch=8, box=0.05)

    # yolov4
    # model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], epochs=220, batch=8, box=0.15, cls=1.5, obj=3)

    set_seed(1)
    model = Detr("detrv1.yaml")
    model.train(data="./cfg/datasets/coco.yaml", imgsz=[640, 640], epochs=100, batch=2)
