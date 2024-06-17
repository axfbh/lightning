from ultralytics import YOLO

# Load a model
if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="VOC.yaml", epochs=100, imgsz=640, amp=False, device=0, cache=False)
