import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️")

st.title("🗑️ Garbage Classification App")
st.write("Upload an image to classify the type of garbage")

# Load model (you'll add your trained model)
# model = tf.keras.models.load_model("models/garbage_classifier.h5")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    # Add classification logic here
