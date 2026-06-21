import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    """
    Preprocess image for inference
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def draw_boxes(image, boxes, labels, colors=None):
    """
    Draw bounding boxes on image
    """
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[i % len(colors)], 2)
        cv2.putText(image, labels[i], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i % len(colors)], 2)
    return image
