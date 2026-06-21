# Garbage Identification Project

## Overview
This project uses YOLOv8 for object detection to identify and classify garbage items in images.

## Features
- Object Detection: Detects multiple garbage items per image
- 12 Classes: Consolidated from TACO dataset (59 raw classes)
- Web Interface: Gradio-based UI for easy testing
- Fast Inference: YOLOv8 optimized for real-time detection

## Classes
1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash
7. Bottle
8. Can
9. Wrapper
10. Cup
11. Bag
12. Container

## Installation

```bash
git clone https://github.com/choudharysidra240-ctrl/Garbage-identification.git
cd Garbage-identification
pip install -r requirements.txt
```

## Usage

### Run Gradio App
```bash
python app.py
```

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/training.ipynb
```
