from ultralytics import YOLO

model=YOLO(r"C:\Users\ananya_sharma\Downloads\object_detection\runs\detect\train\weights\best.pt")
model.predict(source=r"C:\Users\ananya_sharma\Downloads\object_detection\dataset\val\images", save=True)