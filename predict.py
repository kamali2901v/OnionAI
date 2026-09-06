"""
Phase 6: Predict Healthy vs Defective for a single onion image.
Usage: python predict.py path\to\image.jpg
"""

import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
MODEL_PATH = "models/onion_classifier_v1.keras"
CLASS_NAMES = ["defective", "healthy"]  # must match the order printed during training/eval
CONFIDENCE_THRESHOLD = 0.70  # below this, flag as uncertain

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path\\to\\image.jpg")
        return

    image_path = sys.argv[1]

    # Load model (custom_objects needed because preprocess_input is baked into the model)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
    )

    # Load and prepare the image
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # model expects a batch, so wrap in [ ]

    # NOTE: no manual preprocess_input call here -- it's already inside the model.
    prob = model.predict(img_array, verbose=0)[0][0]

    # prob = probability of class 1 ("healthy"), since that's index 1 in CLASS_NAMES
    predicted_class = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = prob if prob > 0.5 else 1 - prob

    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence * 100:.1f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        print("Status: LOW CONFIDENCE -- manual inspection recommended.")
    else:
        print("Status: OK")

if __name__ == "__main__":
    main()