import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os
from datetime import datetime

from db import (
    init_db, create_batch, create_sample, record_human_decision,
    record_size_assessment, get_batch_summary, get_all_batches, get_batch_samples
)
from translations import t
from report_generator import generate_batch_report

IMG_SIZE = (224, 224)
MODEL_PATH = "models/onion_classifier_multiclass.keras"
CLASS_NAMES = ["damaged", "healthy", "rotten"]
CONFIDENCE_THRESHOLD = 0.60
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
init_db()

if "lang" not in st.session_state:
    st.session_state.lang = "en"

lang_display = {"English": "en", "தமிழ் (Tamil)": "ta", "हिन्दी (Hindi)": "hi"}
lang_choice = st.sidebar.selectbox(
    "Language / மொழி / भाषा",
    list(lang_display.keys()),
    index=list(lang_display.values()).index(st.session_state.lang),
)
st.session_state.lang = lang_display[lang_choice]
lang = st.session_state.lang

st.set_page_config(page_title=t("app_title", lang), page_icon="🧅", layout="centered")
st.markdown("""
<style>
    .main {
        background-color: #FAF7F2;
    }
    h1 {
        color: #8B4513;
        font-weight: 700;
    }
    h2, h3 {
        color: #5C4033;
    }
    div.stButton > button {
        background-color: #8B4513;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #A0522D;
        color: white;
    }
    [data-testid="stMetricValue"] {
        color: #8B4513;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F5DEB3;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #E0D5C7;
        border-radius: 10px;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
    )

model = load_model()

st.title(t("app_title", lang))

# --- Sidebar: batch selection/creation ---
st.sidebar.header(t("batch_header", lang))

if "current_batch" not in st.session_state:
    st.session_state.current_batch = None

with st.sidebar.expander(t("create_batch", lang)):
    supplier = st.text_input(t("supplier", lang))
    center = st.text_input(t("centre", lang))
    variety = st.text_input(t("variety", lang))
    quantity = st.number_input(t("quantity", lang), min_value=0.0, value=0.0)
    unit = st.selectbox(t("unit", lang), ["kg", "quintal", "tonnes"])
    intended_use = st.text_input(t("intended_use", lang), value="General")

    if st.button(t("create_batch_btn", lang)):
        if supplier and center:
            batch_id = create_batch(supplier, center, variety, quantity, unit, intended_use)
            st.session_state.current_batch = batch_id
            st.success(f"{t('batch_created', lang)}: {batch_id}")
        else:
            st.error(t("supplier_centre_required", lang))

all_batches = get_all_batches()
batch_options = [b["batch_id"] for b in all_batches]
if batch_options:
    selected = st.sidebar.selectbox(
        t("active_batch", lang),
        batch_options,
        index=batch_options.index(st.session_state.current_batch) if st.session_state.current_batch in batch_options else 0,
    )
    st.session_state.current_batch = selected

if not st.session_state.current_batch:
    st.info(t("select_batch_prompt", lang))
    st.stop()

st.subheader(f"{t('active_batch', lang)}: {st.session_state.current_batch}")

# --- Sample inspection ---
st.markdown(f"### {t('inspect_sample', lang)}")
tab1, tab2 = st.tabs([t("camera_tab", lang), t("upload_tab", lang)])

img_file = None
with tab1:
    camera_img = st.camera_input(t("take_photo", lang))
    if camera_img is not None:
        img_file = camera_img

with tab2:
    uploaded_img = st.file_uploader(t("upload_image", lang), type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        img_file = uploaded_img

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption=t("sample_image", lang), use_container_width=True)

    img_resized = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized), axis=0)
    probs = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    class_key_map = {"healthy": "healthy", "damaged": "damaged", "rotten": "rotten"}

    st.subheader(f"{t('ai_prediction', lang)}: {t(class_key_map[predicted_class], lang).upper()}")
    st.write(f"{t('confidence', lang)}: {confidence * 100:.1f}%")
    for i, name in enumerate(CLASS_NAMES):
        st.write(f"{t(class_key_map[name], lang)}: {probs[i] * 100:.1f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(t("low_confidence", lang))
    else:
        st.success(t("high_confidence", lang))

    if "last_sample_id" not in st.session_state or st.session_state.get("last_img_name") != img_file.name:
        img_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{img_file.name}"
        img_path = os.path.join(UPLOAD_DIR, img_filename)
        img.save(img_path)

        sample_id = create_sample(
            st.session_state.current_batch, img_path, predicted_class, confidence, grade=None
        )
        st.session_state.last_sample_id = sample_id
        st.session_state.last_img_name = img_file.name
        st.session_state.last_prediction = predicted_class

    sample_id = st.session_state.last_sample_id

    # --- Human verification ---
    st.markdown(f"#### {t('human_verification', lang)}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("confirm_ai", lang), key=f"confirm_{sample_id}"):
            record_human_decision(sample_id, predicted_class)
            st.success(t("confirmed", lang))
    with col2:
        correction = st.selectbox(
            t("correct_to", lang),
            CLASS_NAMES,
            format_func=lambda c: t(class_key_map[c], lang),
            key=f"correct_select_{sample_id}",
        )
        if st.button(t("submit_correction", lang), key=f"correct_btn_{sample_id}"):
            reason = st.session_state.get(f"reason_{sample_id}", "")
            record_human_decision(sample_id, correction, correction_reason=reason)
            st.success(f"{t('corrected_to', lang)} {t(class_key_map[correction], lang)}.")

    reason_text = st.text_input(t("correction_reason", lang), key=f"reason_{sample_id}")

    # --- Size assessment ---
    st.markdown(f"#### {t('size_assessment', lang)}")
    size_labels = [t("size_normal", lang), t("size_undersized", lang), t("size_not_assessed", lang)]
    size_internal = ["Normal", "Undersized", "Not Assessed"]
    size_choice_idx = st.radio(
        t("size_label", lang), range(3),
        format_func=lambda i: size_labels[i],
        index=2, key=f"size_{sample_id}", horizontal=True,
    )
    if st.button(t("save_size", lang), key=f"size_btn_{sample_id}"):
        record_size_assessment(sample_id, size_internal[size_choice_idx])
        st.success(f"{t('size_recorded', lang)}: {size_labels[size_choice_idx]}")

st.divider()

# --- Batch summary ---
st.markdown(f"### {t('batch_summary', lang)}")
summary = get_batch_summary(st.session_state.current_batch)
st.write(f"**{t('total_samples', lang)}:** {summary['total_samples']}")

col1, col2, col3 = st.columns(3)
col1.metric(t("healthy", lang), f"{summary['healthy']} ({summary['healthy_pct']}%)")
col2.metric(t("damaged", lang), f"{summary['damaged']} ({summary['damaged_pct']}%)")
col3.metric(t("rotten", lang), f"{summary['rotten']} ({summary['rotten_pct']}%)")

st.write(f"{t('human_corrections', lang)}: {summary['human_corrections']}")
st.write(f"{t('size_assessment', lang)} — {t('size_normal', lang)}: {summary['size_normal']}, "
         f"{t('size_undersized', lang)}: {summary['size_undersized']}, "
         f"{t('size_not_assessed', lang)}: {summary['size_not_assessed']}")

# --- PDF report ---
st.divider()
st.markdown(f"### {t('generate_report', lang)}")
if st.button(t("generate_pdf_btn", lang)):
    pdf_path = generate_batch_report(st.session_state.current_batch)
    with open(pdf_path, "rb") as f:
        st.download_button(
            t("download_pdf", lang),
            data=f,
            file_name=f"{st.session_state.current_batch}_report.pdf",
            mime="application/pdf",
        )
    st.success(f"{t('report_generated', lang)}: {pdf_path}")