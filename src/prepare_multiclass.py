"""
Splits the 3-class onion dataset (healthy/rotten/damaged) into
train/val/test folders, same 70/15/15 logic as the original binary split.
"""

import os
import shutil
import random

random.seed(42)

SOURCE_DIR = "dataset_multiclass"
OUTPUT_DIR = "dataset_multiclass_split"
CLASSES = ["healthy", "rotten", "damaged"]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"'{OUTPUT_DIR}' already exists -- delete it first if you want to re-split.")
        return

    for split in SPLIT_RATIOS:
        for class_name in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

    for class_name in CLASSES:
        src_folder = os.path.join(SOURCE_DIR, class_name)
        files = os.listdir(src_folder)
        random.shuffle(files)

        n = len(files)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, split_files in splits.items():
            for fname in split_files:
                src_path = os.path.join(src_folder, fname)
                dst_path = os.path.join(OUTPUT_DIR, split, class_name, fname)
                shutil.copy2(src_path, dst_path)

        print(f"[{class_name}] total={n} -> train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

    print("\nDone. Split copies are in 'dataset_multiclass_split/'.")

if __name__ == "__main__":
    main()