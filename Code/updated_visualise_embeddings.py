"""
Updated visualise_embeddings.py

Purpose:
- Map an *unlabelled* folder of images to 2D using a pretrained embedding model.
- No class/flag tagging. Just compute embeddings and project to 2D (t-SNE by default).
- Configure `MODEL_PATH` and `DATASET_DIR` **inside this file**.
- Saves: embeddings (.npy), 2D coordinates (.csv), and a scatter plot (.png).

Notes:
- Assumes you provide a `build_embedding_model()` factory in `model.py` that returns the embedding network.
- If your model expects a specific preprocessing (mean/std, etc.), update `preprocess_image` accordingly.
- t-SNE can be slow for large N (>10k). Consider switching to UMAP if available.
"""
from __future__ import annotations

import os
import sys
import math
import time
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Optional UMAP support (falls back to TSNE if umap-learn not installed)
try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ----------------------
# === USER SETTINGS ===
# Set these two paths for your environment.
MODEL_PATH = "/path/to/your/model_weights.h5"  # e.g. "runs/ckpt-0123.h5" or a SavedModel directory
DATASET_DIR = "/path/to/folder/of/images"       # A folder containing images (recursively scanned)

# Optional tweaks
IMAGE_SIZE = (224, 224)   # Change to what your model expects
BATCH_SIZE = 64
EMBEDDING_L2_NORMALIZE = True   # Set False if your downstream prefers raw embeddings

# Dimensionality reduction
REDUCER = "tsne"           # "tsne" or "umap" (uses UMAP if installed)
TSNE_PERPLEXITY = 30       # Reasonable range: 5..50
TSNE_ITER = 1000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Output directory (auto timestamped inside ./embedding_maps)
OUTPUT_ROOT = Path("embedding_maps")
# ----------------------

# Import after settings (so local model.py can be swapped easily)
try:
    from model import build_embedding_model  # user-provided
except Exception as e:
    print("ERROR: Could not import build_embedding_model from model.py\n", e, file=sys.stderr)
    sys.exit(1)


def configure_tf() -> None:
    # Make TF behave nicely on GPU and keep logs quiet
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
    # Adjust this if your model expects different preprocessing
    img = load_img(path, target_size=IMAGE_SIZE)
    arr = img_to_array(img)
    arr = arr / 255.0  # simple rescale
    return arr


def batched(iterable: List[Path], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def load_model(weights_path: str | Path):
    model = build_embedding_model()
    # Support SavedModel directories as well as weight files
    weights_path = str(weights_path)
    if os.path.isdir(weights_path):
        # Assume SavedModel format
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
            # Avoid division by zero
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
        # Guard t-SNE perplexity per scikit-learn constraints: < n_samples
        perplexity = min(TSNE_PERPLEXITY, max(5, (n - 1) // 3)) if n > 3 else 2
        tsne = TSNE(n_components=2, init="pca", perplexity=perplexity, n_iter=TSNE_ITER, learning_rate="auto", random_state=42)
        coords = tsne.fit_transform(embeddings)
    return coords


def save_outputs(out_dir: Path, image_paths: List[Path], embeddings: np.ndarray, coords2d: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw embeddings
    np.save(out_dir / "embeddings.npy", embeddings)

    # Save mapping CSV (filename, x, y)
    csv_path = out_dir / "coords_2d.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x", "y"])  # header
        for p, (x, y) in zip(image_paths, coords2d):
            writer.writerow([str(p), float(x), float(y)])

    # Quick JSON sidecar with run metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(MODEL_PATH),
        "dataset_dir": str(DATASET_DIR),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "normalized": EMBEDDING_L2_NORMALIZE,
        "reducer": REDUCER,
        "tsne": {"perplexity": TSNE_PERPLEXITY, "iter": TSNE_ITER},
        "umap": {"n_neighbors": UMAP_N_NEIGHBORS, "min_dist": UMAP_MIN_DIST, "available": UMAP_AVAILABLE},
        "num_images": len(image_paths),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save a static scatter plot
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(coords2d[:, 0], coords2d[:, 1], s=8, alpha=0.8)
    plt.title("Image Embeddings (2D)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    png_path = out_dir / "scatter.png"
    plt.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {out_dir / 'embeddings.npy'}")


def main() -> None:
    configure_tf()

    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print(f"ERROR: MODEL_PATH does not exist: {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)
    if not DATASET_DIR or not os.path.isdir(DATASET_DIR):
        print(f"ERROR: DATASET_DIR is not a directory: {DATASET_DIR}", file=sys.stderr)
        sys.exit(3)

    images = find_images(DATASET_DIR)
    if not images:
        print(f"No images found in {DATASET_DIR}")
        sys.exit(0)

    print(f"Found {len(images)} images. Loading model…")
    model = load_model(MODEL_PATH)

    print("Computing embeddings…")
    embeddings = compute_embeddings(model, images)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Reducing to 2D with {}…".format("UMAP" if (REDUCER.lower()=="umap" and UMAP_AVAILABLE) else "t-SNE"))
    coords2d = reduce_to_2d(embeddings)

    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_outputs(run_dir, images, embeddings, coords2d)


if __name__ == "__main__":
    main()
