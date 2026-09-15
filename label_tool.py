"""
Manual label review tool.
Shows one image from dataset/defective at a time.
You classify it as Rotten / Sprouted / Damaged / Skip / Reject.
Approved images get copied into dataset/reviewed/<class>/.
Everything is tracked in dataset/metadata/images.csv so nothing
is silently guessed or duplicated.
"""

import streamlit as st
import os
import csv
import shutil
from PIL import Image

SOURCE_DIR = "dataset/defective"
REVIEWED_DIR = "dataset/reviewed"
REJECTED_DIR = "dataset/rejected"
METADATA_FILE = "dataset/metadata/images.csv"
CLASSES = ["rotten", "sprouted", "damaged"]

os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
for c in CLASSES:
    os.makedirs(os.path.join(REVIEWED_DIR, c), exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)


def load_reviewed_filenames():
    """Which source images have already been reviewed, so we don't repeat them."""
    if not os.path.isfile(METADATA_FILE):
        return set()
    reviewed = set()
    with open(METADATA_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviewed.add(row["original_filename"])
    return reviewed


def log_decision(filename, final_class, review_status):
    file_exists = os.path.isfile(METADATA_FILE)
    with open(METADATA_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "original_filename", "source", "final_class", "review_status"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "original_filename": filename,
            "source": "dataset/defective",
            "final_class": final_class,
            "review_status": review_status,
        })


st.set_page_config(page_title="Onion Label Review", layout="centered")
st.title("🏷️ Onion Defect Label Review")

all_files = sorted(os.listdir(SOURCE_DIR))
reviewed = load_reviewed_filenames()
remaining = [f for f in all_files if f not in reviewed]

st.write(f"**Progress:** {len(reviewed)} reviewed / {len(all_files)} total "
         f"({len(remaining)} remaining)")

if not remaining:
    st.success("✅ All images reviewed!")
    st.stop()

current_file = remaining[0]
current_path = os.path.join(SOURCE_DIR, current_file)

img = Image.open(current_path)
st.image(img, caption=current_file, use_container_width=True)

st.write("What defect does this image show?")
col1, col2, col3 = st.columns(3)

def handle_decision(label, status):
    if status == "APPROVED":
        dest = os.path.join(REVIEWED_DIR, label, current_file)
    else:
        dest = os.path.join(REJECTED_DIR, current_file)
    shutil.copy2(current_path, dest)
    log_decision(current_file, label if status == "APPROVED" else "N/A", status)
    st.rerun()

with col1:
    if st.button("🟤 Rotten"):
        handle_decision("rotten", "APPROVED")
with col2:
    if st.button("🌱 Sprouted"):
        handle_decision("sprouted", "APPROVED")
with col3:
    if st.button("💥 Damaged"):
        handle_decision("damaged", "APPROVED")

col4, col5 = st.columns(2)
with col4:
    if st.button("⏭️ Skip / Unclear"):
        log_decision(current_file, "N/A", "NEEDS_REVIEW")
        st.rerun()
with col5:
    if st.button("🗑️ Reject (not a valid onion image)"):
        handle_decision("N/A", "REJECTED")