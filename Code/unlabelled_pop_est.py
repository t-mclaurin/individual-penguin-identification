import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist

# If your project provides this helper, we re-use it.
# It should match the signature used in your existing codebase.
from model import build_embedding_model


def load_and_preprocess_image(path, input_shape):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, input_shape[:2])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def image_paths_from_dir(data_dir, exts=("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    paths = sorted(paths)
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under {data_dir} with extensions {exts}")
    return paths


def batch_image_tensor(paths, batch_size, input_shape):
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        images = [load_and_preprocess_image(p, input_shape) for p in batch_paths]
        images = tf.stack(images)
        yield images


def compute_embeddings(model, paths, batch_size, input_shape):
    all_embeddings = []
    for batch_imgs in tqdm(batch_image_tensor(paths, batch_size, input_shape), desc="Embedding", total=(len(paths) + batch_size - 1)//batch_size):
        embs = model(batch_imgs, training=False).numpy()
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)


def online_cluster_count(embeddings, thresholds):
    """
    Simple online clustering by threshold on Euclidean distance to current cluster representatives.
    For each threshold, walk through the embedding list once and:
      - If the min distance to any representative <= threshold, assign to that cluster.
      - Else create a new cluster with this embedding as representative.
    Returns list of (threshold, num_clusters).
    """
    results = []
    for thresh in tqdm(thresholds, desc="Threshold sweep"):
        reps = []  # representative embeddings (one per discovered cluster)
        for i in range(len(embeddings)):
            e = embeddings[i]
            if len(reps) == 0:
                reps.append(e)
                continue
            dists = cdist([e], reps, metric="euclidean")[0]
            min_dist = dists.min()
            if min_dist <= thresh:
                # assign to that existing cluster (no rep update for simplicity)
                continue
            else:
                reps.append(e)
        results.append((thresh, len(reps)))
    return results


def plot_single_line(results, out_path):
    thresholds = [t for t, _ in results]
    counts = [c for _, c in results]
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, counts, marker="o")
    plt.xlabel("Similarity Threshold (Euclidean)")
    plt.ylabel("Estimated #Clusters (Population)")
    plt.title("Estimated Population vs Threshold (Unlabeled)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep clustering on UNLABELED dataset")
    parser.add_argument("--data_dir", type=str, default="../validated_cam1/valid", help="Root directory containing images (recursively)")
    parser.add_argument("--model_path", type=str, default="3rd_checkpoints_search/best_config2.weights.h5", help="Path to weights for embedding model")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--thresh_min", type=float, default=7.5)
    parser.add_argument("--thresh_max", type=float, default=9.0)
    parser.add_argument("--thresh_steps", type=int, default=30)
    parser.add_argument("--output_png", type=str, default="unlabeled_threshold_sweep.png")
    args = parser.parse_args()

    input_shape = (args.input_h, args.input_w, 3)
    thresholds = np.linspace(args.thresh_min, args.thresh_max, args.thresh_steps)

    # Build & load model
    model = build_embedding_model(
        input_shape=input_shape,
        embedding_dim=args.embedding_dim,
        base_trainable=False
    )
    model.load_weights(args.model_path)

    # Collect images & compute embeddings
    paths = image_paths_from_dir(args.data_dir)
    print(f"Found {len(paths)} images under {args.data_dir}")
    embeddings = compute_embeddings(model, paths, args.batch_size, input_shape)

    # Sweep thresholds (unlabeled; we only report a single curve)
    results = online_cluster_count(embeddings, thresholds)

    # Plot a single line graph: threshold vs estimated population
    plot_single_line(results, args.output_png)

    # Also print a small table to stdout
    print("\nThreshold\tEstimated_Clusters")
    for t, c in results:
        print(f"{t:.4f}\t{c}")


if __name__ == "__main__":
    main()
