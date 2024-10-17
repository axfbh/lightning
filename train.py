from ops.models.detection.yolo.trainer import Yolo

if __name__ == '__main__':
    model = Yolo("./cfg/models/yolo/v5/yolov5.yaml")
    model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], batch=8, box=0.05)
