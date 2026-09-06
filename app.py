import streamlit as st
import numpy as np
import tensorflow as tf
import csv
import os
import pandas as pd
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from grading import load_rules, apply_grading

IMG_SIZE = (224, 224)
MODEL_PATH = "models/onion_classifier_v1.keras"
CLASS_NAMES = ["defective", "healthy"]  # match evaluate.py's printed class order
CONFIDENCE_THRESHOLD = 0.70
AUDIT_FILE = "audit_log.csv"

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
rules = load_rules()

def log_to_audit(row):
    file_exists = os.path.isfile(AUDIT_FILE)
    with open(AUDIT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

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

    grading_result = apply_grading(predicted_class, float(confidence), rules)

    st.subheader(f"Grade: {grading_result['grade']}")
    if grading_result["manual_review_required"]:
        st.warning("⚠️ Manual review required before final grading.")

    st.caption(
        "Grading thresholds are currently provisional, pending sourcing of "
        "verified AGMARK/NAFED onion grading standards."
    )

    # Stable ID tied to the photo itself, not the clock — same photo = same id every rerun
    img_bytes = img_file.getvalue()
    sample_id = str(hash(img_bytes))

    st.divider()
    st.write("**Human Review**")
    st.write("Does this AI result look correct to you?")
    col1, col2 = st.columns(2)

    already_logged_key = f"logged_{sample_id}"
    if already_logged_key not in st.session_state:
        st.session_state[already_logged_key] = False

    human_decision = None
    with col1:
        if st.button("✅ Yes, AI is correct", key=f"confirm_{sample_id}"):
            human_decision = predicted_class

    with col2:
        correction = st.selectbox(
            "❌ No, it's actually:",
            ["", "healthy", "defective"],
            key=f"correction_select_{sample_id}",
        )
        if correction and st.button("Submit correction", key=f"submit_correction_{sample_id}"):
            human_decision = correction

    # --- Log ONE row per photo, only once, guarded by session_state ---
    if human_decision is not None and not st.session_state[already_logged_key]:
        log_row = {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "ai_prediction": predicted_class,
            "ai_confidence": round(float(confidence), 4),
            "grade": grading_result["grade"],
            "human_agreed": human_decision == predicted_class,
            "final_decision": human_decision,
        }
        log_to_audit(log_row)
        st.session_state[already_logged_key] = True
        if human_decision == predicted_class:
            st.success(f"✅ Logged: Human confirmed AI's result ({predicted_class.upper()})")
        else:
            st.warning(f"⚠️ Logged: Human corrected AI. AI said {predicted_class.upper()}, human says {human_decision.upper()}")
    elif st.session_state[already_logged_key]:
        st.info("✔️ This sample has already been logged.")
    else:
        st.info("👆 Please confirm or correct the result above to log this sample.")

    with st.expander("📋 View audit log"):
        if os.path.isfile(AUDIT_FILE):
            with open(AUDIT_FILE, "rb") as f:
                st.download_button("Download audit log (CSV)", f, file_name="audit_log.csv")
            log_df = pd.read_csv(AUDIT_FILE)
            log_df = log_df.sort_values("timestamp", ascending=False)
            st.dataframe(log_df)