"""
visualise_individual_poles.py

Purpose
-------
Visualise embeddings for a *single individual* folder (e.g., data/ID1) just like the
previous coloured plot, *and* highlight three special points in the embedding space:
  1) the medoid (central-most point)
  2) the point farthest from the medoid (first pole)
  3) the point farthest from that first pole (second pole)

Those three are drawn in black on top of the colour-coded scatter, and their file
paths are printed and saved to disk for later inspection.

Outputs (under ./embedding_maps/<timestamp>_INDIV)
-------------------------------------------------
- scatter_colored_with_poles.png
- coords_2d.csv (columns: filename,x,y,category,special)
- special_points.json (paths + indices for medoid, pole1, pole2)
- embeddings.npy
- run_meta.json

Notes
-----
- Category colouring still uses your CSV (train, val•known, val•unknown, unmatched).
- Robust filename matching (full path → relative → unique basename) like before.
- Medoid is computed with a memory-safe chunked method (O(N^2) but limited RAM).
"""
from __future__ import annotations

import os
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Optional UMAP support
try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.spatial.distance import cdist

# ----------------------
# === USER SETTINGS ===
MODEL_PATH = "3rd_checkpoints_search/best_config9.weights.h5"
INDIVIDUAL_DIR = "../mclaurin/individuals_8_tagless/Ron_burgundy"
LABELS_CSV = "penguinID_dataset_splits_8_tagless.csv"  # <-- set this to your CSV with columns: filename/path, known_status, split


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EMBEDDING_L2_NORMALIZE = True

REDUCER = "tsne"           # "tsne" or "umap"
TSNE_PERPLEXITY = 30
TSNE_ITER = 1000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Behaviour for items not found in CSV
DROP_UNMATCHED = False  # if True, drop points not found in CSV; if False, keep and colour grey

OUTPUT_ROOT = Path("embedding_maps")
# ----------------------

# Import your model factory (same signature as your project)
try:
    from model2 import build_embedding_model
except Exception as e:
    print("ERROR: Could not import build_embedding_model from model2.py\n", e, file=sys.stderr)
    sys.exit(1)

# ----------------------
# Helpers
# ----------------------

def configure_tf() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def find_images(root: str | Path, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[Path]:
    root = Path(root)
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def preprocess_image(path: Path) -> np.ndarray:
    img = load_img(path, target_size=IMAGE_SIZE)
    arr = img_to_array(img)
    arr = arr / 255.0
    return arr


def batched(iterable: List[Path], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def load_model(weights_path: str | Path):
    model = build_embedding_model()
    weights_path = str(weights_path)
    if os.path.isdir(weights_path):
        model = tf.keras.models.load_model(weights_path)
    else:
        model.load_weights(weights_path)
    model.trainable = False
    return model


def compute_embeddings(model, image_paths: List[Path]) -> np.ndarray:
    feats = []
    for batch_paths in batched(image_paths, BATCH_SIZE):
        batch = np.stack([preprocess_image(p) for p in batch_paths], axis=0)
        emb = model.predict(batch, verbose=0)
        if isinstance(emb, (list, tuple)):
            emb = emb[0]
        emb = np.asarray(emb)
        if EMBEDDING_L2_NORMALIZE:
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms
        feats.append(emb)
    return np.concatenate(feats, axis=0)


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    n = embeddings.shape[0]
    if REDUCER.lower() == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=42)
        coords = reducer.fit_transform(embeddings)
    else:
        perplexity = min(TSNE_PERPLEXITY, max(5, (n - 1) // 3)) if n > 3 else 2
        tsne = TSNE(n_components=2, init="pca", perplexity=perplexity, max_iter=TSNE_ITER, learning_rate="auto", random_state=42)
        coords = tsne.fit_transform(embeddings)
    return coords

# ----------------------
# CSV → category mapping (same as before)
# ----------------------
CATEGORY_NAMES = {
    "train": "train",
    "val_known": "val • known",
    "val_unknown": "val • unknown",
    "unmatched": "unmatched (not in CSV)",
}

CATEGORY_COLORS = {
    "train": "#1f77b4",       # blue
    "val_known": "#2ca02c",    # green
    "val_unknown": "#d62728",  # red
    "unmatched": "#7f7f7f",    # grey
}


def _norm_path_str(p: str | Path) -> str:
    return str(p).replace("\\", "/").lower()


def _choose_filename_col(df: pd.DataFrame) -> str:
    candidates = ["filename", "file", "path", "filepath", "image", "img_path"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("Could not find a filename/path column in CSV. Expected one of: " + ", ".join(candidates))


def build_path_to_category(images: List[Path], labels_df: pd.DataFrame, dataset_root: Path) -> Tuple[np.ndarray, Dict[int, str]]:
    full_map: Dict[str, int] = {}
    rel_map: Dict[str, int] = {}
    base_counts: Dict[str, int] = {}
    base_map: Dict[str, int] = {}

    dataset_root = dataset_root.resolve()

    for i, p in enumerate(images):
        full_key = _norm_path_str(p.resolve())
        full_map[full_key] = i
        try:
            rel = p.resolve().relative_to(dataset_root)
            rel_map[_norm_path_str(rel)] = i
        except Exception:
            pass
        b = p.name.lower()
        base_counts[b] = base_counts.get(b, 0) + 1

    for i, p in enumerate(images):
        b = p.name.lower()
        if base_counts.get(b, 0) == 1:
            base_map[b] = i

    fname_col = _choose_filename_col(labels_df)

    categories = np.array(["unmatched"] * len(images), dtype=object)
    matched = 0

    for _, row in labels_df.iterrows():
        raw_path = str(row[fname_col])
        ks = str(row.get("known_status", "")).strip().lower()
        sp = str(row.get("split", "")).strip().lower()

        if sp == "train":
            cat = "train"
        elif sp == "val" and ks == "known":
            cat = "val_known"
        elif sp == "val" and ks == "unknown":
            cat = "val_unknown"
        else:
            continue

        key_full = _norm_path_str(raw_path)
        key_rel = None
        try:
            rel = Path(raw_path)
            if rel.is_absolute():
                try:
                    rel = rel.relative_to(dataset_root)
                except Exception:
                    pass
            key_rel = _norm_path_str(rel)
        except Exception:
            pass
        key_base = Path(raw_path).name.lower()

        idx = None
        if key_full in full_map:
            idx = full_map[key_full]
        elif key_rel and key_rel in rel_map:
            idx = rel_map[key_rel]
        elif key_base in base_map:
            idx = base_map[key_base]

        if idx is not None:
            categories[idx] = cat
            matched += 1

    stats = {
        0: f"Matched {matched}/{len(images)} images to CSV entries.",
        1: f"Counts → train: {(categories=='train').sum()}, val_known: {(categories=='val_known').sum()}, val_unknown: {(categories=='val_unknown').sum()}, unmatched: {(categories=='unmatched').sum()}"
    }
    return categories, stats

# ----------------------
# Medoid and poles
# ----------------------

def medoid_index(embeddings: np.ndarray, metric: str = "euclidean", chunk_size: int = 2048) -> int:
    """Return index of the medoid (row with minimal sum of distances to all points).
    Chunked to limit memory. Complexity ~O(N^2) but avoids N×N full matrix in RAM.
    """
    n = embeddings.shape[0]
    if n == 0:
        raise ValueError("No embeddings to compute a medoid.")
    best_idx = 0
    best_sum = np.inf
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        d = cdist(embeddings[start:end], embeddings, metric=metric)
        sums = d.sum(axis=1)
        j = int(np.argmin(sums))
        if sums[j] < best_sum:
            best_sum = float(sums[j])
            best_idx = start + j
    return best_idx


def farthest_index(from_vec: np.ndarray, embeddings: np.ndarray, metric: str = "euclidean") -> int:
    d = cdist([from_vec], embeddings, metric=metric)[0]
    return int(np.argmax(d))

# ----------------------
# Saving & plotting
# ----------------------

def save_outputs(out_dir: Path, image_paths: List[Path], embeddings: np.ndarray, coords2d: np.ndarray, categories: np.ndarray,
                 med_idx: int, pole1_idx: int, pole2_idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally drop unmatched
    if DROP_UNMATCHED:
        keep = categories != "unmatched"
        # Keep also the special indices even if unmatched
        keep_special = np.zeros_like(keep, dtype=bool)
        for k in [med_idx, pole1_idx, pole2_idx]:
            if 0 <= k < len(keep):
                keep_special[k] = True
        keep = keep | keep_special

        image_paths = [p for p, k in zip(image_paths, keep) if k]
        coords2d = coords2d[keep]
        embeddings = embeddings[keep]
        categories = categories[keep]

        # Re-map special indices after filtering
        index_map = np.cumsum(keep) - 1
        med_local = int(index_map[med_idx])
        pole1_local = int(index_map[pole1_idx])
        pole2_local = int(index_map[pole2_idx])
        med_idx, pole1_idx, pole2_idx = med_local, pole1_local, pole2_local

    # Save raw embeddings
    np.save(out_dir / "embeddings.npy", embeddings)

    # Save mapping CSV (filename, x, y, category, special)
    csv_path = out_dir / "coords_2d.csv"
    specials = np.array([""] * len(image_paths), dtype=object)
    for i, tag in [(med_idx, "medoid"), (pole1_idx, "pole1"), (pole2_idx, "pole2")]:
        if 0 <= i < len(specials):
            specials[i] = tag
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x", "y", "category", "special"])  # header
        for p, (x, y), c, s in zip(image_paths, coords2d, categories, specials):
            writer.writerow([str(p), float(x), float(y), str(c), str(s)])

    # Save metadata + special points (paths)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(MODEL_PATH),
        "individual_dir": str(INDIVIDUAL_DIR),
        "labels_csv": str(LABELS_CSV),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "normalized": EMBEDDING_L2_NORMALIZE,
        "reducer": REDUCER,
        "tsne": {"perplexity": TSNE_PERPLEXITY, "iter": TSNE_ITER},
        "umap": {"n_neighbors": UMAP_N_NEIGHBORS, "min_dist": UMAP_MIN_DIST, "available": UMAP_AVAILABLE},
        "num_images": len(image_paths),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "drop_unmatched": bool(DROP_UNMATCHED),
        "counts": {
            "train": int((categories=="train").sum()),
            "val_known": int((categories=="val_known").sum()),
            "val_unknown": int((categories=="val_unknown").sum()),
            "unmatched": int((categories=="unmatched").sum()),
        },
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    specials_json = {
        "medoid": {"index": int(med_idx), "path": str(image_paths[med_idx])},
        "pole1": {"index": int(pole1_idx), "path": str(image_paths[pole1_idx])},
        "pole2": {"index": int(pole2_idx), "path": str(image_paths[pole2_idx])},
    }
    with open(out_dir / "special_points.json", "w", encoding="utf-8") as f:
        json.dump(specials_json, f, indent=2)

    # Plot: colour by category + overlay special points in black
    fig = plt.figure(figsize=(9, 7))
    for key in ["train", "val_known", "val_unknown", "unmatched"]:
        if key == "unmatched" and DROP_UNMATCHED:
            continue
        mask = categories == key
        if not np.any(mask):
            continue
        plt.scatter(
            coords2d[mask, 0], coords2d[mask, 1],
            s=12, alpha=0.85 if key != "unmatched" else 0.35,
            c=CATEGORY_COLORS[key], label=CATEGORY_NAMES[key]
        )

    # Overlay medoid + poles
    def _draw_point(i: int, label: str):
        plt.scatter(
            coords2d[i, 0], coords2d[i, 1],
            s=140, c="black", edgecolors="white", linewidths=1.4,
            marker="o", zorder=6, label=label
        )

    _draw_point(med_idx, "medoid")
    if pole1_idx != med_idx:
        _draw_point(pole1_idx, "pole1")
    if pole2_idx not in (med_idx, pole1_idx):
        _draw_point(pole2_idx, "pole2")

    plt.title("Embeddings — single individual (medoid + poles)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.legend(title="Legend", markerscale=1.4, frameon=True)
    plt.tight_layout()
    png_path = out_dir / "scatter_colored_with_poles.png"
    plt.savefig(png_path, dpi=220)
    plt.close(fig)

    print("\nSpecial points:")
    print(f"  medoid:  [{med_idx:>4}] {image_paths[med_idx]}")
    print(f"  pole1:   [{pole1_idx:>4}] {image_paths[pole1_idx]}")
    print(f"  pole2:   [{pole2_idx:>4}] {image_paths[pole2_idx]}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {out_dir / 'special_points.json'}")
    print(f"Saved: {out_dir / 'embeddings.npy'}")

# ----------------------
# Main
# ----------------------

def main() -> None:
    # Basic checks
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print(f"ERROR: MODEL_PATH does not exist: {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)
    if not INDIVIDUAL_DIR or not os.path.isdir(INDIVIDUAL_DIR):
        print(f"ERROR: INDIVIDUAL_DIR is not a directory: {INDIVIDUAL_DIR}", file=sys.stderr)
        sys.exit(3)
    if not LABELS_CSV or not os.path.exists(LABELS_CSV):
        print(f"ERROR: LABELS_CSV not found: {LABELS_CSV}", file=sys.stderr)
        sys.exit(4)

    configure_tf()

    # Load images for this individual
    images = find_images(INDIVIDUAL_DIR)
    if not images:
        print(f"No images found in {INDIVIDUAL_DIR}")
        sys.exit(0)
    print(f"Found {len(images)} images in {INDIVIDUAL_DIR}. Loading model…")

    # Load model
    model = load_model(MODEL_PATH)

    # Compute embeddings
    print("Computing embeddings…")
    embeddings = compute_embeddings(model, images)
    print(f"Embeddings shape: {embeddings.shape}")

    # Reduce to 2D
    print("Reducing to 2D with {}…".format("UMAP" if (REDUCER.lower()=="umap" and UMAP_AVAILABLE) else "t-SNE"))
    coords2d = reduce_to_2d(embeddings)

    # Load labels CSV and align categories (even though this is a single individual dir)
    print("Loading labels CSV and assigning categories…")
    labels_df = pd.read_csv(LABELS_CSV)
    categories, stats = build_path_to_category(images, labels_df, Path(INDIVIDUAL_DIR))
    for k in sorted(stats.keys()):
        print(stats[k])

    # Compute medoid + poles
    print("Computing medoid and poles…")
    med_idx = medoid_index(embeddings, metric="euclidean", chunk_size=2048)
    pole1_idx = farthest_index(embeddings[med_idx], embeddings, metric="euclidean")
    pole2_idx = farthest_index(embeddings[pole1_idx], embeddings, metric="euclidean")

    run_dir = OUTPUT_ROOT / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_INDIV")
    save_outputs(run_dir, images, embeddings, coords2d, categories, med_idx, pole1_idx, pole2_idx)

if __name__ == "__main__":
    main()
