import sys
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/onion_classifier_v1.tflite"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["defective", "healthy"]

if len(sys.argv) < 2:
    print("Usage: python demo_offline.py <path_to_onion_image>")
    sys.exit(1)

image_path = sys.argv[1]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
img_array = np.array(img, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])[0][0]
predicted_class = CLASS_NAMES[1] if output > 0.5 else CLASS_NAMES[0]
confidence = output if output > 0.5 else 1 - output

print("=" * 40)
print(f"  OFFLINE PREDICTION (no internet used)")
print(f"  Result: {predicted_class.upper()}")
print(f"  Confidence: {confidence * 100:.1f}%")
print("=" * 40)