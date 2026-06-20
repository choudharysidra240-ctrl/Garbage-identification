import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    # Load your dataset here
    print("Loading data...")
    return

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation="softmax")
    ])
    return model

def train_model():
    model = create_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print("Training started...")
    # Add training code here
    model.save("models/garbage_classifier.h5")
    print("Training complete!")

if __name__ == "__main__":
    train_model()
