import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "models/onion_classifier_v1.keras"
CLASS_NAMES = ["defective", "healthy"]  # match evaluate.py's printed class order
CONFIDENCE_THRESHOLD = 0.70

st.set_page_config(page_title="OnionAI", page_icon="🧅")
st.title("🧅 OnionAI — Quality Check")
st.write("Take a photo or upload an onion image to check its quality.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
    )

model = load_model()

tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload"])

img_file = None
with tab1:
    camera_img = st.camera_input("Take a photo of the onion")
    if camera_img is not None:
        img_file = camera_img

with tab2:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        img_file = uploaded_img

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Input image", use_container_width=True)

    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    # NOTE: no manual preprocess_input call here -- it's already inside the model.

    prob = model.predict(img_array)[0][0]
    predicted_class = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = prob if prob > 0.5 else 1 - prob

    st.subheader(f"Prediction: {predicted_class.upper()}")
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence * 100:.1f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low confidence — manual inspection recommended.")
    else:
        st.success("✅ High confidence prediction")   