from ultralytics import YOLO

model = YOLO("runs/detect/crop_weed_512/weights/best.pt")
model.val(data="data.yaml", imgsz=512)
