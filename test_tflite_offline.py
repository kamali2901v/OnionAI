"""
Tests the TFLite model in isolation -- proves it runs without
needing the full TensorFlow/Keras stack or any network call.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import sys

TFLITE_PATH = "models/onion_classifier_v1.tflite"
IMG_SIZE = (224, 224)

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(sys.argv) < 2:
    print("Usage: python test_tflite_offline.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
img_array = np.array(img, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

prob = output[0][0]
prediction = "healthy" if prob > 0.5 else "defective"
confidence = prob if prob > 0.5 else 1 - prob

print(f"Prediction: {prediction.upper()}")
print(f"Confidence: {confidence*100:.1f}%")