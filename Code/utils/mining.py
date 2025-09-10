# utils/mining.py

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def pairwise_distances(embeddings):
    """
    Computes the full pairwise squared Euclidean distance matrix.
    embeddings: (batch_size, embedding_dim)
    returns: (batch_size, batch_size) matrix of distances
    """
    dot_product = np.dot(embeddings, embeddings.T)
    square_norm = np.diagonal(dot_product)
    distances = square_norm[:, None] - 2 * dot_product + square_norm[None, :]
    distances = np.maximum(distances, 0.0)
    return distances


def semi_hard_triplet_indices(labels, embeddings, margin):
    """
    For each anchor, find a semi-hard positive and a semi-hard negative:
    Positive: same label, closer than some negatives
    Negative: different label, not too far (semi-hard)
    """
    labels = np.array(labels)
    distances = pairwise_distances(embeddings)
    batch_size = embeddings.shape[0]

    triplets = []

    for anchor_idx in range(batch_size):
        anchor_label = labels[anchor_idx]
        anchor_distance = distances[anchor_idx]

        # Mask for positives and negatives
        positive_mask = (labels == anchor_label) & (np.arange(batch_size) != anchor_idx)
        negative_mask = labels != anchor_label

        pos_indices = np.where(positive_mask)[0]
        neg_indices = np.where(negative_mask)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue  # Need at least one valid positive and negative

        # Find semi-hard triplets: pos < neg < pos + margin
        for pos_idx in pos_indices:
            pos_dist = anchor_distance[pos_idx]
            semi_hard_negatives = [neg_idx for neg_idx in neg_indices
                                    if pos_dist < anchor_distance[neg_idx] < pos_dist + margin]

            if semi_hard_negatives:
                neg_idx = np.random.choice(semi_hard_negatives)
                triplets.append((anchor_idx, pos_idx, neg_idx))

    return triplets

def batch_hard_triplet_indices(labels, embeddings):
    triplets = []
    labels = np.array(labels)
    distances = euclidean_distances(embeddings, embeddings)

    for i in range(len(embeddings)):
        anchor_label = labels[i]
        anchor_dists = distances[i]

        # Hardest positive: same class, farthest distance
        pos_mask = (labels == anchor_label) & (np.arange(len(labels)) != i)
        if not np.any(pos_mask):
            continue
        hardest_pos_idx = np.argmax(anchor_dists * pos_mask)

        # Hardest negative: different class, smallest distance
        neg_mask = labels != anchor_label
        if not np.any(neg_mask):
            continue
        hardest_neg_idx = np.argmin(anchor_dists + (1 - neg_mask) * 1e6)

        triplets.append((i, hardest_pos_idx, hardest_neg_idx))

    return triplets
