"""
Phase 4: Train a binary Healthy vs Defective onion classifier
using transfer learning on MobileNetV2.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
DATA_DIR = "dataset_split"

# --- Load data from folders ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

print("Class order (0/1):", train_ds.class_names)

# --- Data augmentation: creates varied versions of each training image
# (flip, rotate, zoom, contrast, brightness) so the model learns onion
# features, not our specific background/lighting. Only active during
# training -- Keras automatically disables it during evaluation/prediction.
# NOTE: operates on raw 0-255 images, BEFORE preprocess_input rescales them. ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.15, 0.15),   # NEW: shifts onion position in frame
    tf.keras.layers.RandomContrast(0.3),             # stronger
    tf.keras.layers.RandomBrightness(0.3), 
])

# --- Build the model: augmentation + normalize + MobileNetV2 base + our classification head ---
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,       # exclude MobileNet's original 1000-class head
    weights="imagenet",      # use pretrained knowledge
)
base_model.trainable = False  # freeze base -- we don't retrain it yet

model = models.Sequential([
    data_augmentation,
    layers.Lambda(preprocess_input),  # normalize AFTER augmentation, right before the base model
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),  # 1 output: probability of "defective"
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --- Train ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# --- Save the trained model ---
model.save("models/onion_classifier_v1.keras")
print("\nModel saved to models/onion_classifier_v1.keras")