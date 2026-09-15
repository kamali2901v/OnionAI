"""
Evaluate the multiclass onion model on the held-out test set.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = "dataset_multiclass_split"

model = tf.keras.models.load_model(
    "models/onion_classifier_multiclass.keras",
    custom_objects={"preprocess_input": preprocess_input},
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = test_ds.class_names
print("Class order:", class_names)

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_true = np.argmax(y_true, axis=1)  # convert one-hot back to class index

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print("Rows = Actual, Columns = Predicted")
print("Classes:", class_names)
print(cm)

print(f"\nTotal test images: {len(y_true)}")