from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(data="dataset/data.yaml", epochs=20, imgsz=640, batch=8)