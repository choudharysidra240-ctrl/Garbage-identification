from ultralytics import YOLO
import torch

model = YOLO("yolov8n.pt")

results = model.train(
    data="taco_consolidated/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else "cpu"
)
