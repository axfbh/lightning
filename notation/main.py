from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

# Load a models
if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")  # load a pretrained models (recommended for training)

    # Train the models
    results = model.train(data="VOC.yaml", epochs=100, imgsz=640, amp=False, device=0, batch=4, workers=2)
