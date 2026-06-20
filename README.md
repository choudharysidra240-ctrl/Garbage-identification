# Garbage Identification Project

## Overview
This project uses deep learning to classify different types of garbage and waste materials.

## Features
- Image classification of garbage types
- Web interface using Streamlit
- TensorFlow deep learning model

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

## Dataset
The model is trained on various garbage categories including:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Model Architecture
- Convolutional Neural Network
- 6 classes
- 224x224 input size
- Transfer learning from MobileNetV2
