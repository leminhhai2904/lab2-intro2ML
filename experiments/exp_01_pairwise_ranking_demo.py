"""
Experiment 1: Pairwise Ranking Demonstration

Visualizes score-based ranking on synthetic 2D data.
Shows scatter plot of positive/negative samples and score contours.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import generate_synthetic_data, compute_true_scores
from visualization import plot_2d_data, plot_score_contours


def main():
    """Run experiment 1: Pairwise ranking demo."""
    print("=" * 70)
    print("Experiment 1: Pairwise Ranking Demonstration")
    print("=" * 70)
    
    # Generate data
    random_seed = 42
    data = generate_synthetic_data(n_positive=100, n_negative=100, 
                                   noise_std=0.5, random_seed=random_seed)
    
    X_pos = data['X_positive']
    X_neg = data['X_negative']
    weights = data['true_weights']
    
    print(f"Generated {len(X_pos)} positive samples around (2, 2)")
    print(f"Generated {len(X_neg)} negative samples around (-2, -2)")
    print(f"True weights: w = {weights}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Raw data
    print("\nPlotting raw data scatter...")
    ax1 = plot_2d_data(X_pos, X_neg, title="2D Bipartite Ranking Data", ax=axes[0])
    
    # Plot 2: Score contours
    print("Plotting score contours with true weights...")
    score_func = lambda X: compute_true_scores(X, weights)
    ax2 = plot_score_contours(X_pos, X_neg, score_func, 
                              title="Score Contours: s(x) = w^T x", ax=axes[1])
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'outputs' / 'figures' / 'pairwise_ranking_demo.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving figure to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Done!")
    
    # Compute some statistics
    scores_pos = compute_true_scores(X_pos, weights)
    scores_neg = compute_true_scores(X_neg, weights)
    
    correct_rankings = np.sum(scores_pos[:, np.newaxis] > scores_neg[np.newaxis, :])
    total_pairs = len(X_pos) * len(X_neg)
    auc_true = correct_rankings / total_pairs
    
    print(f"\nWith true scoring function:")
    print(f"  Mean positive score: {scores_pos.mean():.4f}")
    print(f"  Mean negative score: {scores_neg.mean():.4f}")
    print(f"  AUC (pairwise accuracy): {auc_true:.4f}")


if __name__ == "__main__":
    main()
