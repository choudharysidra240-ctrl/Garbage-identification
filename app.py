import gradio as gr
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_garbage(image):
    results = model(image)
    return results[0].plot()

iface = gr.Interface(
    fn=detect_garbage,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Garbage Detection System"
)

iface.launch(share=True)
