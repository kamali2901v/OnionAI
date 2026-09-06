"""
Converts the trained Keras model to TensorFlow Lite format.
Proves the model can run fully on-device without a live server.
"""

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

MODEL_PATH = "models/onion_classifier_v1.keras"
TFLITE_PATH = "models/onion_classifier_v1.tflite"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input}
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

original_size = os.path.getsize(MODEL_PATH) / (1024*1024)
tflite_size = len(tflite_model) / (1024*1024)

print(f"Original model: {original_size:.2f} MB")
print(f"TFLite model:   {tflite_size:.2f} MB")
print(f"Saved to: {TFLITE_PATH}")