import os
import csv
import random

data_dir = "../mclaurin/individuals_8_tagless"
output_csv = "penguinID_dataset_splits_8_tagless.csv"
test_ratio = 0.2
random.seed(42)

rows = []

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(("jpg", "jpeg", "png"))
    ]

    full_paths = [os.path.join(class_name, f) for f in images]

    if len(images) <= 5:
        # Put low-sample individuals entirely in validation as unknown
        for f in full_paths:
            rows.append({
                "filename": f,
                "label": class_name,
                "split": "val",
                "known_status": "unknown"
            })
    else:
        # Enough images to split into train/val
        random.shuffle(full_paths)
        split_idx = int(len(images) * test_ratio)
        val_imgs = full_paths[:split_idx]
        train_imgs = full_paths[split_idx:]

        for f in train_imgs:
            rows.append({
                "filename": f,
                "label": class_name,
                "split": "train",
                "known_status": "known"
            })

        for f in val_imgs:
            rows.append({
                "filename": f,
                "label": class_name,
                "split": "val",
                "known_status": "known"
            })

# Save CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("CSV created!")