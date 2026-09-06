"""
Quick sanity check: can we load our onion images with OpenCV/Pillow,
and what do they look like (size, format, corrupt files)?
"""

import os
import cv2
from PIL import Image

DATASET_DIR = "dataset"
CLASSES = ["healthy", "defective"]

for class_name in CLASSES:
    folder = os.path.join(DATASET_DIR, class_name)
    files = os.listdir(folder)
    print(f"\n[{class_name}] {len(files)} files found")

    for fname in files[:3]:
        path = os.path.join(folder, fname)

        img_cv = cv2.imread(path)
        if img_cv is None:
            print(f"  WARNING - OpenCV FAILED to load: {fname}")
            continue

        try:
            img_pil = Image.open(path)
            img_pil.verify()
        except Exception as e:
            print(f"  WARNING - Pillow FAILED to load: {fname} ({e})")
            continue

        h, w, channels = img_cv.shape
        print(f"  OK - {fname}: {w}x{h}, {channels} channels")

print("\nDone checking sample images.")