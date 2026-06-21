import os
from ultralytics import YOLO
import torch

def train_model():
    """
    Train YOLOv8 model on garbage detection dataset
    """
    # Load a pretrained model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="taco_consolidated/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        patience=10,
        save=True
    )

    # Save the model
    model.save("models/saved_model/garbage_detector.pt")
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
