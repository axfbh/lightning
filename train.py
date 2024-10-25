from ops.models.detection.yolo.trainer import Yolo

if __name__ == '__main__':
    model = Yolo("yolov4n.yaml")
    # model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], epochs=220, batch=8, box=0.05)
    # yolov4
    model.train(data="./cfg/datasets/voc.yaml", imgsz=[640, 640], epochs=220, batch=8, box=0.15, cls=1.5, obj=3)
