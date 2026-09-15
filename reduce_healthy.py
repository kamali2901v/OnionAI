import os
import random
import shutil

random.seed(42)

BASE = "dataset_multiclass_split"
TARGETS = {
    "train": 385,
    "val": 82,
    "test": 83,
}

for split, target in TARGETS.items():
    src_dir = os.path.join(BASE, split, "healthy")
    unused_dir = os.path.join(BASE, "healthy_unused", split)
    os.makedirs(unused_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    current = len(files)

    if current <= target:
        print(f"{split}: already at or below target ({current} <= {target}), skipping")
        continue

    to_remove = current - target
    random.shuffle(files)
    remove_files = files[:to_remove]

    for fname in remove_files:
        shutil.move(os.path.join(src_dir, fname), os.path.join(unused_dir, fname))

    remaining = current - len(remove_files)
    print(f"{split}: {current} -> {remaining} (moved {len(remove_files)} to {unused_dir})")

print("Done.")