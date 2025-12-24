from ultralytics import YOLO

model = YOLO("runs/detect/crop_weed_512/weights/best.pt")
model.predict(source="sample_images", imgsz=512, save=True)
