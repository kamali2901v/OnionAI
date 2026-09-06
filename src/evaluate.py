"""
Phase 5: Evaluate the trained model on the held-out test set.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = "dataset_split"

# custom_objects tells Keras how to resolve the preprocess_input function
# that's baked into the saved model as a Lambda layer.
model = tf.keras.models.load_model(
    "models/onion_classifier_v1.keras",
    custom_objects={"preprocess_input": preprocess_input},
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = test_ds.class_names
print("Class order (0/1):", class_names)

# NOTE: do NOT manually preprocess here -- it's already inside the model.
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred_probs = model.predict(test_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(f"                Predicted {class_names[0]}   Predicted {class_names[1]}")
print(f"Actual {class_names[0]:<10}    {cm[0][0]:<10}          {cm[0][1]}")
print(f"Actual {class_names[1]:<10}    {cm[1][0]:<10}          {cm[1][1]}")

if len(y_true) < 30:
    print(f"\nNote: test set has only {len(y_true)} images -- "
          "these metrics give a rough signal, not a statistically reliable performance estimate.")