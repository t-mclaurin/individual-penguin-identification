import os
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import build_embedding_model

# --- CONFIGURATION ---
MODEL_PATH = "3rd_checkpoints_search/best_config2.weights.h5"
DATA_CSV = "penguinID_dataset_splits_8_tagless.csv"
DATA_DIR = "../mclaurin/individuals_8"
INPUT_SHAPE = (224, 224, 3)
EMBEDDING_DIM = 512
BATCH_SIZE = 64
THRESHOLDS = np.linspace(1, 4, 30)  # similarity thresholds

# --- MODEL SETUP ---
model = build_embedding_model(
    input_shape=INPUT_SHAPE,
    embedding_dim=EMBEDDING_DIM,
    base_trainable=False
)
model.load_weights(MODEL_PATH)

# --- IMAGE LOADING ---
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, INPUT_SHAPE[:2])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def batch_generator(filenames, labels, batch_size):
    for i in range(0, len(filenames), batch_size):
        batch_paths = filenames[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        images = [load_and_preprocess_image(os.path.join(DATA_DIR, p)) for p in batch_paths]
        images = tf.stack(images)
        yield images, batch_labels

# --- LOAD DATA ---
df = pd.read_csv(DATA_CSV)
unknown_df = df[(df["split"] == "val") & (df["known_status"] == "unknown")]
filenames = unknown_df["filename"].values
true_labels = unknown_df["label"].values

true_pop = unknown_df["label"].nunique()

print(true_pop)

# --- COMPUTE EMBEDDINGS ---
all_embeddings = []
for batch_imgs, _ in tqdm(batch_generator(filenames, true_labels, BATCH_SIZE), desc="Embedding"):
    embs = model(batch_imgs, training=False).numpy()
    all_embeddings.append(embs)
all_embeddings = np.vstack(all_embeddings)

# --- THRESHOLD SWEEP ---
results = []
for thresh in tqdm(THRESHOLDS, desc="Threshold sweep"):
    reference_embeddings = []
    reference_labels = []
    assigned_ids = []
    next_id = 0
    fp, fn = 0, 0

    for i, query_emb in enumerate(all_embeddings):
        true_label = true_labels[i]

        if len(reference_embeddings) == 0:
            reference_embeddings.append(query_emb)
            reference_labels.append(true_label)
            assigned_ids.append(next_id)
            next_id += 1
            continue

        dists = cdist([query_emb], reference_embeddings, metric="euclidean")[0]
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist <= thresh:
            predicted_label = reference_labels[min_idx]
            assigned_ids.append(reference_labels.index(predicted_label))
            if predicted_label != true_label:
                fp += 1
        else:
            reference_embeddings.append(query_emb)
            reference_labels.append(true_label)
            assigned_ids.append(next_id)
            if true_label in reference_labels[:-1]:
                fn += 1
            next_id += 1

    total = len(true_labels)
    estimated_pop = len(set(assigned_ids))
    results.append({
        "threshold": thresh,
        "estimated_population": estimated_pop,
        "est_prop_pop" : estimated_pop / true_pop,
        "false_positive_rate": fp / total,
        "false_negative_rate": fn / total
    })

# Print total
print(total)
    
# --- PLOT RESULTS ---
thresholds = [r["threshold"] for r in results]
population = [r["estimated_population"] for r in results]
prop_pop = [r["est_prop_pop"] for r in results]
fp_rate = [r["false_positive_rate"] for r in results]
fn_rate = [r["false_negative_rate"] for r in results]

plt.figure(figsize=(12, 6))
plt.plot(thresholds, prop_pop, label="Estimated Population Size (Proportion)")
plt.plot(thresholds, fp_rate, label="False Positive Rate")
plt.plot(thresholds, fn_rate, label="False Negative Rate")
plt.xlabel("Similarity Threshold")
plt.ylabel("Metric Value")
plt.title("Population Estimate and Error Rates vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("population_estimation_threshold_sweep.png")
plt.show()

