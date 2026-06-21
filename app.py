import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the trained model
model = YOLO("yolov8n.pt")

def detect_trash(image):
    """
    Detect garbage items in an image
    Args:
        image: Input image
    Returns:
        Annotated image with bounding boxes
    """
    # Run inference
    results = model(image)
    # Annotate image
    annotated_image = results[0].plot()
    return annotated_image

# Create Gradio interface
iface = gr.Interface(
    fn=detect_trash,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Detected Garbage"),
    title="Garbage Detection System",
    description="Upload an image to detect and classify garbage items using YOLOv8",
    examples=[],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=True, debug=True)
