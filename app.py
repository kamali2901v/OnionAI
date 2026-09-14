import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

from grading import load_rules, apply_grading
from db import (
    init_db, create_batch, create_sample, record_human_decision,
    get_batch_summary, get_all_batches,
)

IMG_SIZE = (224, 224)
MODEL_PATH = "models/onion_classifier_v1.keras"
CLASS_NAMES = ["defective", "healthy"]  # match evaluate.py's printed class order
CONFIDENCE_THRESHOLD = 0.70

init_db()

st.set_page_config(page_title="AgroNex", page_icon="🧅")
st.title("🧅 AgroNex — Onion Procurement Inspection")
st.write("AI-assisted quality assessment for procurement batches.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
    )

model = load_model()
rules = load_rules()

# ---------------- BATCH SELECTION / CREATION ----------------
st.divider()
st.subheader("📦 Batch")

existing_batches = get_all_batches()
batch_options = ["+ Create New Batch"] + [b["batch_id"] for b in existing_batches]
default_index = 0
if "active_batch" in st.session_state and st.session_state["active_batch"] in batch_options:
    default_index = batch_options.index(st.session_state["active_batch"])

selected = st.selectbox("Select or create a batch:", batch_options, index=default_index)

if selected == "+ Create New Batch":
    with st.form("new_batch_form"):
        supplier_name = st.text_input("Supplier / Farmer Name")
        procurement_centre = st.text_input("Procurement Centre")
        onion_variety = st.text_input("Onion Variety", value="Red Onion")
        quantity_received = st.number_input("Quantity Received", min_value=0.0, value=0.0)
        unit = st.selectbox("Unit", ["kg", "quintal", "tonne"])
        intended_use = st.selectbox("Intended Use", ["Storage", "Immediate Dispatch", "Other"])
        submitted = st.form_submit_button("Create Batch")

        if submitted:
            new_batch_id = create_batch(
                supplier_name, procurement_centre, onion_variety,
                quantity_received, unit, intended_use
            )
            st.success(f"✅ Batch created: {new_batch_id}")
            st.session_state["active_batch"] = new_batch_id
            st.rerun()

    st.stop()  # don't show inspection UI until a batch exists
else:
    st.session_state["active_batch"] = selected
    active_batch = selected
    st.info(f"Active batch: **{active_batch}**")

# ---------------- SAMPLE INSPECTION ----------------
st.divider()
st.subheader("🔍 Sample Inspection")

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

    st.subheader(f"AI Prediction: {predicted_class.upper()}")
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

    # Stable key tied to the photo bytes, so a rerun doesn't create a duplicate sample
    img_bytes = img_file.getvalue()
    photo_key = str(hash(img_bytes))
    sample_id_key = f"sample_id_{photo_key}"

    # Create the sample row in the database ONCE per unique photo
    if sample_id_key not in st.session_state:
        new_sample_id = create_sample(
            batch_id=active_batch,
            image_path="(not saved to disk in this MVP)",
            ai_prediction=predicted_class,
            ai_confidence=float(confidence),
            grade=grading_result["grade"],
        )
        st.session_state[sample_id_key] = new_sample_id

    sample_id = st.session_state[sample_id_key]
    st.caption(f"Sample ID: **{sample_id}**")

    st.divider()
    st.write("**Human Review**")
    st.write("Does this AI result look correct to you?")
    col1, col2 = st.columns(2)

    reviewed_key = f"reviewed_{sample_id}"
    if reviewed_key not in st.session_state:
        st.session_state[reviewed_key] = False

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

    if human_decision is not None and not st.session_state[reviewed_key]:
        record_human_decision(sample_id, human_decision)
        st.session_state[reviewed_key] = True
        if human_decision == predicted_class:
            st.success(f"✅ Recorded: Human confirmed AI's result ({predicted_class.upper()})")
        else:
            st.warning(
                f"⚠️ Recorded: Human corrected AI. "
                f"AI said {predicted_class.upper()}, human says {human_decision.upper()}"
            )
    elif st.session_state[reviewed_key]:
        st.info("✔️ This sample has already been reviewed.")
    else:
        st.info("👆 Please confirm or correct the result above to record this sample.")

# ---------------- BATCH SUMMARY ----------------
st.divider()
st.subheader("📊 Batch Summary")
summary = get_batch_summary(active_batch)

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", summary["total_samples"])
col2.metric("Healthy %", f"{summary['healthy_pct']}%")
col3.metric("Defective %", f"{summary['defective_pct']}%")
st.caption(f"Human corrections: {summary['human_corrections']}")