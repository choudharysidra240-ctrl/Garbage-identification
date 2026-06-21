#!/usr/bin/env python3
"""
Garbage Detection using YOLOv8
This script detects and classifies garbage items in images
"""

import torch
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully!")

def detect_trash(image):
    """
    Detect garbage items in an image
    Args:
        image: Input image (numpy array or path)
    Returns:
        Annotated image with bounding boxes
    """
    # Run inference
    results = model(image)
    # Annotate image
    annotated_image = results[0].plot()
    return annotated_image

def train_model():
    """
    Train the YOLOv8 model on TACO dataset
    """
    print("Starting training...")
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
    print("Training complete!")
    return results

def launch_gradio():
    """
    Launch Gradio web interface
    """
    iface = gr.Interface(
        fn=detect_trash,
        inputs=gr.Image(type="numpy", label="Upload Image"),
        outputs=gr.Image(type="numpy", label="Detected Garbage"),
        title="🗑️ Garbage Detection System",
        description="Upload an image to detect and classify garbage items using YOLOv8",
        examples=[],
        allow_flagging="never"
    )
    iface.launch(share=True, debug=True)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Garbage Detection System")
    print("="*50 + "\n")
    print("Choose an option:")
    print("1. Launch Gradio Interface")
    print("2. Train Model")
    print("3. Test on single image")
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == "1":
        launch_gradio()
    elif choice == "2":
        train_model()
    elif choice == "3":
        img_path = input("Enter image path: ")
        if os.path.exists(img_path):
            result = detect_trash(img_path)
            cv2.imwrite("output.jpg", result)
            print("Output saved as output.jpg")
        else:
            print("File not found!")
    else:
        print("Invalid choice!")
