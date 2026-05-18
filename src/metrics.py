"""
Metrics module for ranking evaluation.

Implements ranking metrics from scratch:
- Pairwise misranking error
- AUC (Area Under the Curve)
- ROC curve points
- Margin loss
"""

import numpy as np


def pairwise_misranking_error(X_pairs, y_pairs, scores):
    """
    Compute pairwise misranking error.
    
    For each pair (x_pos, x_neg) with y=1, count misranking if score(x_neg) >= score(x_pos).
    
    Args:
        X_pairs (array): Pairwise data (n_pairs, 2, d)
        y_pairs (array): Pairwise labels (n_pairs,) - all +1 for our case
        scores (array): Computed scores for all individual points
                       or function that takes X_pairs and returns scores
    
    Returns:
        float: Fraction of misranked pairs
    """
    n_pairs = X_pairs.shape[0]
    
    # If scores is a function, compute scores for all pairs
    if callable(scores):
        scores_pos = scores(X_pairs[:, 0, :])
        scores_neg = scores(X_pairs[:, 1, :])
    else:
        # scores is precomputed (n_pairs, 2) array
        scores_pos = scores[:, 0]
        scores_neg = scores[:, 1]
    
    # Count misranked pairs: negative score >= positive score
    misranked = np.sum(scores_neg >= scores_pos)
    
    return misranked / n_pairs if n_pairs > 0 else 0.0


def auc_from_scratch(X_pos, X_neg, score_func):
    """
    Compute AUC (Area Under ROC Curve) from scratch.
    
    AUC = P(score(x_pos) > score(x_neg)) for random x_pos, x_neg.
    
    Args:
        X_pos (array): Positive samples (n_pos, d)
        X_neg (array): Negative samples (n_neg, d)
        score_func (callable): Function that computes scores
    
    Returns:
        float: AUC value in [0, 1]
    """
    scores_pos = score_func(X_pos)
    scores_neg = score_func(X_neg)
    
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    
    # Count pairs where positive score > negative score
    correct_rankings = 0
    
    for sp in scores_pos:
        # Count how many negative scores are < this positive score
        correct_rankings += np.sum(scores_neg < sp)
    
    total_pairs = n_pos * n_neg
    auc = correct_rankings / total_pairs if total_pairs > 0 else 0.0
    
    return auc


def roc_points_from_scratch(X_pos, X_neg, score_func, n_thresholds=100):
    """
    Compute ROC curve points from scratch.
    
    Returns (FPR, TPR) pairs for different thresholds.
    
    Args:
        X_pos (array): Positive samples (n_pos, d)
        X_neg (array): Negative samples (n_neg, d)
        score_func (callable): Function that computes scores
        n_thresholds (int): Number of threshold points
    
    Returns:
        tuple: (fpr_list, tpr_list) - False Positive Rate and True Positive Rate
    """
    scores_pos = score_func(X_pos)
    scores_neg = score_func(X_neg)
    
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    
    # Collect all unique thresholds
    all_scores = np.concatenate([scores_pos, scores_neg])
    thresholds = np.linspace(all_scores.min() - 1, all_scores.max() + 1, n_thresholds)
    
    fpr_list = []
    tpr_list = []
    
    for threshold in thresholds:
        # True Positives: positive samples with score >= threshold
        tp = np.sum(scores_pos >= threshold)
        tpr = tp / n_pos if n_pos > 0 else 0
        
        # False Positives: negative samples with score >= threshold
        fp = np.sum(scores_neg >= threshold)
        fpr = fp / n_neg if n_neg > 0 else 0
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    return np.array(fpr_list), np.array(tpr_list)


def margin_loss(X_pairs, y_pairs, scores, rho=1.0):
    """
    Compute empirical margin loss.
    
    L_rho(h) = (1/m) * sum_i max(0, rho - y_i * (h(x_i) - h(x'_i)))
    
    For pairwise ranking, y_i = +1 for positive > negative.
    
    Args:
        X_pairs (array): Pairwise data (n_pairs, 2, d)
        y_pairs (array): Pairwise labels (n_pairs,)
        scores (array): Precomputed scores (n_pairs, 2)
        rho (float): Margin parameter
    
    Returns:
        float: Empirical margin loss
    """
    n_pairs = X_pairs.shape[0]
    
    # scores should be (n_pairs, 2) where scores[i, 0] = score(x_pos_i), scores[i, 1] = score(x_neg_i)
    score_diff = scores[:, 0] - scores[:, 1]  # Should be positive for correct ranking
    
    # Margin loss: max(0, rho - score_diff)
    margin_violations = np.maximum(0, rho - score_diff)
    
    return np.mean(margin_violations)


def exponential_loss(X_pairs, y_pairs, scores):
    """
    Compute empirical exponential loss (used in RankBoost).
    
    L_exp(h) = (1/m) * sum_i exp(-y_i * (h(x_i) - h(x'_i)))
    
    Args:
        X_pairs (array): Pairwise data (n_pairs, 2, d)
        y_pairs (array): Pairwise labels (n_pairs,)
        scores (array): Precomputed scores (n_pairs, 2)
    
    Returns:
        float: Empirical exponential loss
    """
    n_pairs = X_pairs.shape[0]
    
    score_diff = scores[:, 0] - scores[:, 1]
    losses = np.exp(-score_diff)  # y_i = +1, so -y_i * diff = -diff
    
    return np.mean(losses)
