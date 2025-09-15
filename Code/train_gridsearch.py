# train_v1_gridsearch.py

import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from model import build_embedding_model
from utils.augmentation import get_static_augmentation, get_progressive_augmentation
from utils.mining import batch_hard_triplet_indices, semi_hard_triplet_indices
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import cdist

import csv
from datetime import datetime
import itertools
#from tqdm import tqdm # Could try progress bar

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_SHAPE = (224, 224, 3)
EMBEDDING_DIM = 512
BATCH_SIZE = 32
EPOCHS = 300
CHECKPOINT_DIR = "Penguin_Photos/checkpoints"
DATA_DIR = "Penguin_Photos/individuals"

EARLY_STOPPING_PATIENCE = 10    # e.g. stop if no improvement in 10 epochs
EARLY_STOPPING_DELTA = 1e-6    # e.g. require at least this much improvement

param_grid = {
    'LEARNING_RATE': [1e-4],
    'LEARNING_RATE_2': [2e-5],
    'MARGIN': [0.5, 0.1],
    'WARMUP_LENGTH': [5, 15, 30],
    'DROPOUT_RATE' : [0, 0.1]
}

keys, values = zip(*param_grid.items())
configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create CSV log file
log_filename = os.path.join(
    CHECKPOINT_DIR,
    f"grid_search_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
with open(log_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "config_id",
        "epoch",
        "avg_loss",
        "triplet_count",
        "mean_pos_dist",
        "mean_neg_dist",
        "mean_known_dist",
        "mean_unknown_dist",
        "top1_acc",
        "top2_acc",
        "top3_acc",
        "top4_acc",
        "top5_acc",
        "threshold_known_acc",
        "unknown_detection_rate",
        "gallery_top1_acc",
        "gallery_top2_acc",
        "gallery_top3_acc",
        "gallery_top4_acc",
        "gallery_top5_acc",
        "LEARNING_RATE",
        "LEARNING_RATE_2",
        "MARGIN",
        "WARMUP_LENGTH",
        "DROPOUT_RATE",
        "maP@1"    
        
    ])

# -----------------------------
# ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help="Resume training from latest checkpoint")
parser.add_argument('--progressive', action='store_true', help="Use progressive augmentation")
args = parser.parse_args()

import pandas as pd

# Load records
df = pd.read_csv("Penguin_Photos/penguinID_dataset_splits.csv")

# Get treining data
train_df = df[
    (df["split"] == "train") &
    (df["known_status"] == "known")
]

class_names = sorted(train_df["label"].unique())
class_to_index = {name: i for i, name in enumerate(class_names)}

# Map labels to integers
train_df["label_index"] = train_df["label"].map(class_to_index)

val_known_df = df[
    (df["split"] == "val") &
    (df["known_status"] == "known")
]

val_unknown_df = df[
    (df["split"] == "val") &
    (df["known_status"] == "unknown")
]

def load_image(filename, label):
    path = tf.strings.join([DATA_DIR, filename], separator=os.sep)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, INPUT_SHAPE[:2])
    image = image / 255.0
    return image, label

def load_image_with_meta(filename, label, known_status):
    path = tf.strings.join([DATA_DIR, filename], separator=os.sep)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, INPUT_SHAPE[:2])
    image = image / 255.0
    return image, label, known_status

train_ds = tf.data.Dataset.from_tensor_slices((
    train_df["filename"].values,
    train_df["label_index"].values
))

train_ds = train_ds.shuffle(1000)
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_known_ds = tf.data.Dataset.from_tensor_slices((
    val_known_df.filename.values,
    val_known_df.label.values
))
val_known_ds = val_known_ds.map(load_image).batch(BATCH_SIZE)

# -----------------------------
# TRAINING LOOP FOR GRID SEARCH
# -----------------------------
SKIP = 0

for config_id, config in enumerate(configs, start=1):
    K.clear_session()
        
    if config_id <= SKIP:
        print(f"Skipping config {config_id}")
        continue

    print(f"\n=== Running config {config_id}/{len(configs)} ===")
    print(config)

    # Parameters for this run
    LEARNING_RATE = config['LEARNING_RATE']
    LEARNING_RATE_2 = config['LEARNING_RATE_2']
    MARGIN = config['MARGIN']
    WARMUP_LENGTH = config['WARMUP_LENGTH']
    DROPOUT_RATE = config['DROPOUT_RATE']
    
    #     steps_per_epoch = train_gen.samples // BATCH_SIZE
    num_examples = len(train_df)
    steps_per_epoch = int(np.ceil(num_examples / BATCH_SIZE))


    # Always start with backbone frozen
    model = build_embedding_model(
        input_shape=INPUT_SHAPE,
        embedding_dim=EMBEDDING_DIM,
        base_trainable=False,
        dropout_rate=DROPOUT_RATE
    )
    optimizer = Adam(learning_rate=LEARNING_RATE)

    # Define triplet loss with the margin from the grid search
    def triplet_loss(y_true, y_pred):
        anchor, positive, negative = tf.unstack(
            tf.reshape(y_pred, [-1, 3, EMBEDDING_DIM]), 3, axis=1
        )
        pos_dist = tf.norm(anchor - positive, axis=1)
        neg_dist = tf.norm(anchor - negative, axis=1)
        
        basic_loss = pos_dist - neg_dist + MARGIN
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return loss

    model.compile(optimizer=optimizer, loss=triplet_loss)

    # Optionally resume
    if args.resume:
            model.load_weights('checkpoints_search/best_config1.weights.h5')

    # Track losses
    loss_history = []
    best_loss = float('inf')
    best_epoch = None
    epochs_without_improvement = 0

    # Track if backbone is unfrozen
    backbone_unfrozen = False

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        if args.progressive:
            idg = get_progressive_augmentation(epoch + 1, EPOCHS)

        total_loss = 0
        triplet_count = 0
        pos_dists = []
        neg_dists = []

        for step, (x_batch, y_batch) in enumerate(train_ds.take(steps_per_epoch)):
            #x_batch, y_batch = next(train_gen)
            
            #embeddings = model.predict(x_batch)
            embeddings = model(x_batch, training=False).numpy()
            triplets = batch_hard_triplet_indices(y_batch, embeddings)
            # To avoid OOM errors
            #MAX_TRIPLETS = 10

            #if len(triplets) > MAX_TRIPLETS:
            #    triplets = triplets[:MAX_TRIPLETS]
    
            if not triplets:
                print(f"  Step {step+1}/{steps_per_epoch}: No valid triplets found.")
                continue
            

            # Compute distances for logging
            a_idx, p_idx, n_idx = zip(*triplets)

            anchor = embeddings[list(a_idx)]
            positive = embeddings[list(p_idx)]
            negative = embeddings[list(n_idx)]

            pos_d = tf.norm(anchor - positive, axis=1).numpy()
            neg_d = tf.norm(anchor - negative, axis=1).numpy()

            pos_dists.extend(pos_d)
            neg_dists.extend(neg_d)

            # Prepare batched triplets
            indices = []
            for a, p, n in triplets:
                indices.extend([a, p, n])

            x_batch_np = x_batch.numpy()
            x_triplets = x_batch_np[indices]

            y_dummy = np.zeros((x_triplets.shape[0],))

            loss = model.train_on_batch(x_triplets, y_dummy)
            total_loss += loss
            triplet_count += len(triplets)
            
            print(
                f"  Step {step+1}/{steps_per_epoch} | "
                f"Triplets: {len(triplets)} | "
                f"Loss: {loss:.4f} | "
                f"PosDist: {np.mean(pos_d):.4f} | "
                f"NegDist: {np.mean(neg_d):.4f}"
            )

        avg_loss = total_loss / triplet_count if triplet_count > 0 else float('nan')
        mean_pos_dist = np.nanmean(pos_dists) if len(pos_dists) else float('nan')
        mean_neg_dist = np.nanmean(neg_dists) if len(neg_dists) else float('nan')
        #mean_pos_dist = np.mean(pos_dists) if pos_dists else float('nan')
        #mean_neg_dist = np.mean(neg_dists) if neg_dists else float('nan')

        print(f"Config {config_id} | Epoch {epoch+1} | "
              f"Loss: {avg_loss:.4f} | "
              f"Triplets: {triplet_count} | "
              f"PosDist: {mean_pos_dist:.4f} | "
              f"NegDist: {mean_neg_dist:.4f}")

        loss_history.append(avg_loss)

        # Save continuous weights
        #cont_path = os.path.join(
        #    CHECKPOINT_DIR, f"config{config_id}_epoch{epoch+1}.weights.h5"
        #)
        #model.save_weights(cont_path)

        # Save best weights
        if avg_loss < best_loss - EARLY_STOPPING_DELTA:
            best_loss = avg_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            best_path = os.path.join(
                CHECKPOINT_DIR, f"best_config{config_id}.weights.h5"
            )
            model.save_weights(best_path)
            print(f"New best model saved with loss {best_loss:.4f} to {best_path}")
        else:
            epochs_without_improvement += 1

        # --- UNFREEZE LOGIC ---
        if epoch+1 == WARMUP_LENGTH:
            print(f" Unfreezing backbone and continuing training.")
            model.base_model.trainable = True
            optimizer = Adam(learning_rate=LEARNING_RATE_2)  # create a new instance
            model.compile(optimizer=optimizer, loss=triplet_loss)

            backbone_unfrozen = True
            epochs_without_improvement = 0  # Optionally reset early stopping after unfreezing

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
        # Save embeddings each epoch
        #x_val_batch, y_val_batch = next(train_gen)
        
        # Grab a batch from train_ds to visualize embeddings
        for x_val_batch, y_val_batch in train_ds.take(1):
            val_embeddings = model(x_val_batch, training=False).numpy()

        embeddings_path = os.path.join(
            CHECKPOINT_DIR,
            f"val_embeddings_config{config_id}_epoch{epoch+1}.npy"
        )
        labels_path = os.path.join(
            CHECKPOINT_DIR,
            f"val_labels_config{config_id}_epoch{epoch+1}.npy"
        )

        np.save(embeddings_path, val_embeddings)
        np.save(labels_path, y_val_batch.numpy())
        print(f"Saved embeddings to {embeddings_path}")
    
        # Write log row for this epoch
        with open(log_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                config_id,
                epoch+1,
                avg_loss,
                triplet_count,
                mean_pos_dist,
                mean_neg_dist,
                "", "", "", "", "", "", ""
            ])
        
    # Load best weights before validation
    model.load_weights(best_path)
    print("Loaded best model for validation evaluation.")
    # Compute gallery embeddings (all known training images)
    gallery_embeddings = []
    gallery_labels = []

    for x_batch, y_batch in train_ds:
        emb_batch = model(x_batch, training=False).numpy()
        gallery_embeddings.append(emb_batch)
        gallery_labels.extend(y_batch.numpy())

    gallery_embeddings = np.vstack(gallery_embeddings)
    gallery_labels = np.array(gallery_labels)


    # Compute centroids for all known classes
    embeddings_by_class = {}

    for x_batch, y_batch in train_ds:
        emb_batch = model(x_batch, training=False).numpy()
        for emb, label in zip(emb_batch, y_batch.numpy()):
            if label not in embeddings_by_class:
                embeddings_by_class[label] = []
            embeddings_by_class[label].append(emb)

    centroids = {
        label: np.mean(emb_list, axis=0)
        for label, emb_list in embeddings_by_class.items()
    }
    
    # Prepare validation dataset
    val_df = df[df.split == "val"]

    val_ds = tf.data.Dataset.from_tensor_slices((
        val_df["filename"].values,
        val_df["label"].values,
        val_df["known_status"].values
    ))

    val_ds = val_ds.map(load_image_with_meta).batch(BATCH_SIZE)

    # Metrics
    correct_topN = {n: 0 for n in range(1, 6)}
    total_known = 0
    
    # New: Top-K accuracy based on gallery
    correct_gallery_topN = {n: 0 for n in range(1, 6)}

    correct_threshold_known = 0
    total_threshold_known = 0

    unknowns_correct = 0
    total_unknowns = 0

    known_dists = []
    unknown_dists = []

    THRESHOLD = 1.2   # adjust later!

    for x_batch, labels_batch, known_status_batch in val_ds:
        embeddings = model(x_batch, training=False).numpy()
        
        # Compute distances to all gallery embeddings
        dists_to_gallery = cdist(embeddings, gallery_embeddings)

        # For each query embedding...
        top_k_labels = []
        for dists in dists_to_gallery:
            # Track closest distance per identity
            closest_per_identity = {}
            for idx, dist in enumerate(dists):
                identity = gallery_labels[idx]
                if identity not in closest_per_identity or dist < closest_per_identity[identity]:
                    closest_per_identity[identity] = dist

            # Sort unique identities by closest distance
            sorted_identities = sorted(closest_per_identity.items(), key=lambda x: x[1])
            top_k = [identity for identity, _ in sorted_identities[:5]]
            top_k_labels.append(top_k)

        top_k_labels = np.array(top_k_labels)


        for i, (emb, true_label, known_flag) in enumerate(zip(
            embeddings,
            labels_batch.numpy(),
            known_status_batch.numpy()
        )):
            dists = {
                label: np.linalg.norm(emb - centroid)
                for label, centroid in centroids.items()
            }

            # Sort centroids by distance
            sorted_labels = sorted(
                dists.items(),
                key=lambda x: x[1]
            )
            ranked_labels = [label for label, dist in sorted_labels]
            nearest_label, nearest_dist = sorted_labels[0]

            if known_flag == b"known":
                total_known += 1
                if isinstance(true_label, (bytes, np.bytes_)):
                    true_label_idx = class_to_index[true_label.decode()]
                else:
                    true_label_idx = class_to_index[str(true_label)]

                # Check Top-N hits
                for n in range(1, 6):
                    top_n = ranked_labels[:n]
                    if true_label_idx in top_n:
                        correct_topN[n] += 1
                        
                for n in range(1, 6):
                    if true_label_idx in top_k_labels[i][:n]:
                        correct_gallery_topN[n] += 1

                known_dists.append(nearest_dist)

                # Threshold-based known accuracy
                if nearest_dist < THRESHOLD:
                    predicted_label = nearest_label
                    if predicted_label == true_label_idx:
                        correct_threshold_known += 1

            else:
                total_unknowns += 1
                unknown_dists.append(nearest_dist)

                if nearest_dist >= THRESHOLD:
                    unknowns_correct += 1
    
    # 1) Collect unknown-only embeddings & labels from the validation set
    unknown_embs = []
    unknown_labels = []   # must be the novel identity labels (ground truth), not "known/unknown" flags
    
    def norm_label(lbl):
    # works for tf tensors, numpy scalars, bytes, or python strings
        try:
            # tf.Tensor -> numpy scalar
            if hasattr(lbl, "numpy"):
                lbl = lbl.numpy()
        except Exception:
            pass
        if isinstance(lbl, (bytes, np.bytes_)):
            return lbl.decode()
        return str(lbl)
    
    for x_batch, labels_batch, known_status_batch in val_ds:
        embs = model(x_batch, training=False).numpy()
        for emb, lbl, kflag in zip(embs, labels_batch, known_status_batch):
            if isinstance(kflag, (bytes, np.bytes_)):
                kflag = kflag.decode()
            elif hasattr(kflag, "numpy"):
                kflag = kflag.numpy().decode()  # if it's a tf string tensor
            if kflag == "unknown":
                unknown_embs.append(emb)
                unknown_labels.append(norm_label(lbl))
    
    unknown_embs = np.vstack(unknown_embs) if len(unknown_embs) else np.empty((0, embed_dim))
    unknown_labels = np.array(unknown_labels, dtype=object)

    # 2) If too few unknowns (or only singletons), guard:
    if len(unknown_embs) >= 2:
        # 3) Pairwise distances among unknowns
        D = cdist(unknown_embs, unknown_embs)  # shape (U, U)

        def ap_at_k_for_query(i, K):
            # rank neighbors excluding self
            idxs = np.argsort(D[i])
            idxs = idxs[idxs != i]  # drop self
            ranked = idxs  # nearest first

            # relevance vector for this query
            q_label = unknown_labels[i]
            rel = (unknown_labels[ranked] == q_label).astype(np.int32)

            R = rel.sum()  # number of available positives
            if R == 0:
                return None  # skip singletons

            K_eff = min(K, len(rel))
            hits = 0
            prec_sum = 0.0
            denom = min(K, R)  # normalize by min(K, #positives) for AP@K
            for k in range(K_eff):
                if rel[k] == 1:
                    hits += 1
                    prec_sum += hits / (k + 1)
            return prec_sum / denom if denom > 0 else None

        def mean_ap_at_k(K):
            vals = []
            for i in range(len(unknown_embs)):
                ap = ap_at_k_for_query(i, K)
                if ap is not None:
                    vals.append(ap)
            return float(np.mean(vals)) if len(vals) > 0 else float('nan')

        mAP_at_1 = mean_ap_at_k(1)
        mAP_at_5 = mean_ap_at_k(5)
    else:
        mAP_at_1 = float('nan')
        mAP_at_5 = float('nan')
    

    # Compute metrics
    topN_accuracies = {
        n: correct_topN[n] / total_known if total_known > 0 else float('nan')
        for n in range(1, 6)
    }
    
    gallery_topN_accuracies = {
        n: correct_gallery_topN[n] / total_known if total_known > 0 else float('nan')
        for n in range(1, 6)
    }

    threshold_known_acc = (
        correct_threshold_known / total_known if total_known > 0 else float('nan')
    )

    unknown_detection_rate = (
        unknowns_correct / total_unknowns if total_unknowns > 0 else float('nan')
    )

    mean_known_dist = np.mean(known_dists) if known_dists else float('nan')
    mean_unknown_dist = np.mean(unknown_dists) if unknown_dists else float('nan')

    # Print results
    for n in range(1, 6):
        print(f"Validation Top-{n} Accuracy: {topN_accuracies[n]*100:.2f}%")
        
    for n in range(1, 6):
        print(f"Gallery Top-{n} Accuracy: {gallery_topN_accuracies[n]*100:.2f}%")

    print(f"Threshold-based Known Accuracy: {threshold_known_acc*100:.2f}%")
    print(f"Unknown Detection Rate: {unknown_detection_rate*100:.2f}%")
    print(f"Mean Known Distance: {mean_known_dist:.4f}")
    print(f"Mean Unknown Distance: {mean_unknown_dist:.4f}")
    print(f"Unknown-only mAP@1: {mAP_at_1*100:.2f}%")

    with open(log_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            config_id,
            best_epoch,
            best_loss,
            "", 
            "", 
            "", 
            mean_known_dist,
            mean_unknown_dist,
            topN_accuracies[1],
            topN_accuracies[2],
            topN_accuracies[3],
            topN_accuracies[4],
            topN_accuracies[5],
            threshold_known_acc,
            unknown_detection_rate,
            gallery_topN_accuracies[1],
            gallery_topN_accuracies[2],
            gallery_topN_accuracies[3],
            gallery_topN_accuracies[4],
            gallery_topN_accuracies[5],
            LEARNING_RATE,
            LEARNING_RATE_2,
            MARGIN,
            WARMUP_LENGTH,
            DROPOUT_RATE,
            mAP_at_1
        ])


    if len(loss_history) > 0:
        # Plot loss curve for this config
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
        plt.title(f"Loss - Config {config_id}")
        plt.xlabel("Epoch")
        plt.ylabel("Average Triplet Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_DIR, f"loss_curve_config{config_id}.png"))
        plt.close()
    else:
        print(f"No loss history for config {config_id}. Skipping plot.")


print("Grid search complete. Logs saved.")
