"""
Experiment 3: ROC Curve and AUC

Compares AUC and ROC curves between:
- Learned scores from RankBoost
- Random scores (baseline)

Demonstrates that AUC measures ranking accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import generate_synthetic_data, create_pairwise_data
from rankboost import RankBoost
from metrics import auc_from_scratch, roc_points_from_scratch
from visualization import plot_roc_curve, plot_auc_comparison


def main():
    """Run experiment 3: ROC and AUC."""
    print("=" * 70)
    print("Experiment 3: ROC Curve and AUC Analysis")
    print("=" * 70)
    
    # Generate data
    random_seed = 42
    data = generate_synthetic_data(n_positive=100, n_negative=100, 
                                   noise_std=0.5, random_seed=random_seed)
    
    X = data['X']
    y = data['y']
    X_pos = data['X_positive']
    X_neg = data['X_negative']
    
    # Create pairwise data for training
    print("Creating pairwise training data...")
    pair_data = create_pairwise_data(X, y, random_seed=random_seed)
    X_pairs = pair_data['X_pairs']
    y_pairs = pair_data['y_pairs']
    
    # Train RankBoost
    print("Training RankBoost...")
    rankboost = RankBoost(n_rounds=50, random_seed=random_seed)
    rankboost.fit(X_pairs, y_pairs, verbose=False)
    
    # Compute ROC and AUC for learned model
    print("\nComputing ROC and AUC for learned model...")
    fpr_learned, tpr_learned = roc_points_from_scratch(X_pos, X_neg, rankboost.score, 
                                                        n_thresholds=100)
    auc_learned = auc_from_scratch(X_pos, X_neg, rankboost.score)
    
    print(f"  Learned AUC: {auc_learned:.4f}")
    
    # Compute ROC and AUC for random scores
    print("Computing ROC and AUC for random baseline...")
    np.random.seed(random_seed + 100)  # Different seed for random scores
    random_scores_pos = np.random.randn(len(X_pos))
    random_scores_neg = np.random.randn(len(X_neg))
    
    def random_score_func(X):
        """Dummy random score function - doesn't use X"""
        # This is a baseline that just returns random scores
        return np.random.randn(X.shape[0])
    
    # For consistent random baseline comparison
    def random_baseline_score(X):
        """Return random baseline scores using seed"""
        np.random.seed(random_seed + 100)
        all_random_scores = np.random.randn(len(X_pos) + len(X_neg))
        # Positive samples get first n_pos scores
        # Negative samples get remaining scores
        if len(X) == len(X_pos):
            return all_random_scores[:len(X_pos)]
        elif len(X) == len(X_neg):
            return all_random_scores[len(X_pos):]
        else:
            # For other X sizes, just generate new random
            return np.random.randn(X.shape[0])
    
    fpr_random, tpr_random = roc_points_from_scratch(X_pos, X_neg, random_baseline_score, 
                                                      n_thresholds=100)
    auc_random = auc_from_scratch(X_pos, X_neg, random_baseline_score)
    
    print(f"  Random AUC: {auc_random:.4f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: ROC curves
    print("\nPlotting ROC curves...")
    ax1 = plot_roc_curve(fpr_learned, tpr_learned, auc=auc_learned, ax=axes[0])
    axes[0].set_title('ROC Curve (Learned Model)', fontsize=14, fontweight='bold')
    
    # Add random ROC for comparison
    axes[0].plot(fpr_random, tpr_random, 'r--', linewidth=2, alpha=0.7, label='Random Model')
    axes[0].legend(fontsize=11)
    
    # Plot 2: AUC comparison
    print("Plotting AUC comparison...")
    ax2 = plot_auc_comparison(auc_learned, auc_random, ax=axes[1])
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'outputs' / 'figures' / 'roc_auc_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving figure to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Done!")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 70)
    print(f"Learned Model AUC: {auc_learned:.4f}")
    print(f"Random Baseline AUC: {auc_random:.4f}")
    print(f"Improvement: {(auc_learned - auc_random):.4f}")
    print("\nInterpretation:")
    print("- AUC measures the probability that a random positive sample")
    print("  has a higher score than a random negative sample")
    print("- Learned model AUC > 0.5 indicates ranking better than random")
    print("- AUC = 1.0 means perfect ranking (all positives > all negatives)")


if __name__ == "__main__":
    main()
