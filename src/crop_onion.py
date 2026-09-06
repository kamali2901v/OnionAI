"""
Auto-crops onion images to remove excess background, using simple
thresholding to find the onion's bounding box. This reduces the model's
ability to 'cheat' by learning background/lighting instead of the onion itself.
"""

import cv2
import numpy as np
import os

SOURCE_DIRS = ["dataset/healthy", "dataset/defective"]
PADDING = 20  # extra pixels around the detected onion, so we don't crop too tight

def crop_onion(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipped (couldn't read): {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contour found, keeping original: {image_path}")
        cv2.imwrite(output_path, img)
        return False

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    h_img, w_img = img.shape[:2]
    x0 = max(0, x - PADDING)
    y0 = max(0, y - PADDING)
    x1 = min(w_img, x + w + PADDING)
    y1 = min(h_img, y + h + PADDING)

    cropped = img[y0:y1, x0:x1]
    cv2.imwrite(output_path, cropped)
    return True

def main():
    for folder in SOURCE_DIRS:
        out_folder = folder + "_cropped"
        os.makedirs(out_folder, exist_ok=True)
        count_ok, count_fail = 0, 0
        for fname in os.listdir(folder):
            src = os.path.join(folder, fname)
            dst = os.path.join(out_folder, fname)
            ok = crop_onion(src, dst)
            count_ok += ok
            count_fail += not ok
        print(f"{folder}: {count_ok} cropped successfully, {count_fail} kept/skipped -> saved in {out_folder}")

if __name__ == "__main__":
    main()