"""
Data generation module for ranking experiments.

Generates synthetic 2D point clouds for bipartite ranking tasks.
Positive class concentrated around (2, 2), negative class around (-2, -2).
"""

import numpy as np


def generate_synthetic_data(n_positive=500, n_negative=500, noise_std=1.5, random_seed=42):
    """
    Generate synthetic 2D bipartite ranking data.
    
    Args:
        n_positive (int): Number of positive samples.
        n_negative (int): Number of negative samples.
        noise_std (float): Standard deviation of noise.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        dict: Dictionary containing:
            - X: All data points (n_total, 2)
            - y: Labels (-1 for negative, +1 for positive)
            - X_positive: Positive samples (n_positive, 2)
            - X_negative: Negative samples (n_negative, 2)
            - true_weights: True weight vector w
    """
    np.random.seed(random_seed)
    
    # Generate positive class around (0.5, 0.5) - Reduced overlap to ~50%
    X_positive = np.random.randn(n_positive, 2) * noise_std + np.array([0.5, 0.5])
    
    # Generate negative class around (-0.5, -0.5)
    X_negative = np.random.randn(n_negative, 2) * noise_std + np.array([-0.5, -0.5])
    
    # True weight vector for score function
    true_weights = np.array([1.0, 1.0])  # s(x) = w^T x
    
    # Combine data
    X = np.vstack([X_positive, X_negative])
    y = np.hstack([np.ones(n_positive), -np.ones(n_negative)])
    
    return {
        'X': X,
        'y': y,
        'X_positive': X_positive,
        'X_negative': X_negative,
        'true_weights': true_weights
    }


def compute_true_scores(X, weights):
    """
    Compute true scores s(x) = w^T x.
    
    Args:
        X (array): Data points (n, d)
        weights (array): Weight vector (d,)
    
    Returns:
        array: Scores (n,)
    """
    return X @ weights


def create_pairwise_data(X, y, random_seed=42):
    """
    Create pairwise training data for ranking.
    
    For each positive-negative pair (x_pos, x_neg), create training pair
    where positive should be ranked higher.
    
    Args:
        X (array): Data points (n, d)
        y (array): Labels (-1 for negative, +1 for positive)
        random_seed (int): Random seed
    
    Returns:
        dict: Dictionary containing:
            - X_pairs: Pairwise features (n_pairs, 2, d) - stack of (x_pos, x_neg)
            - y_pairs: Pairwise labels (n_pairs,) - all +1
            - pos_indices: Indices of positive samples
            - neg_indices: Indices of negative samples
    """
    np.random.seed(random_seed)
    
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == -1)[0]
    
    # Create all positive-negative pairs
    n_pairs = len(pos_indices) * len(neg_indices)
    X_pairs = np.zeros((n_pairs, 2, X.shape[1]))
    
    pair_idx = 0
    for i in pos_indices:
        for j in neg_indices:
            X_pairs[pair_idx, 0, :] = X[i]  # positive sample
            X_pairs[pair_idx, 1, :] = X[j]  # negative sample
            pair_idx += 1
    
    y_pairs = np.ones(n_pairs)  # All pairs: positive > negative
    
    return {
        'X_pairs': X_pairs,
        'y_pairs': y_pairs,
        'pos_indices': pos_indices,
        'neg_indices': neg_indices
    }
