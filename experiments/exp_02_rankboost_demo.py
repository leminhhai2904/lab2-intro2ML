"""
Experiment 2: RankBoost Training Curves

Trains RankBoost and plots:
- Training pairwise error per round
- Exponential loss per round
- Edge of weak rankers per round
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import generate_synthetic_data, create_pairwise_data
from rankboost import RankBoost
from visualization import plot_training_curves, plot_edge_progression


def main():
    """Run experiment 2: RankBoost training curves."""
    print("=" * 70)
    print("Experiment 2: RankBoost Training Curves")
    print("=" * 70)
    
    # Generate data
    random_seed = 42
    data = generate_synthetic_data(n_positive=100, n_negative=100, 
                                   noise_std=0.5, random_seed=random_seed)
    
    X = data['X']
    y = data['y']
    
    # Create pairwise data
    print("Creating pairwise training data...")
    pair_data = create_pairwise_data(X, y, random_seed=random_seed)
    X_pairs = pair_data['X_pairs']
    y_pairs = pair_data['y_pairs']
    
    print(f"Generated {len(X_pairs)} pairwise training samples")
    
    # Train RankBoost
    print(f"\nTraining RankBoost for 50 rounds...")
    rankboost = RankBoost(n_rounds=50, random_seed=random_seed)
    rankboost.fit(X_pairs, y_pairs, verbose=True)
    
    print(f"\nTraining completed!")
    print(f"  Final training error: {rankboost.training_errors[-1]:.4f}")
    print(f"  Final exp loss: {rankboost.training_exp_losses[-1]:.4f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Training curves
    print("\nPlotting training curves...")
    ax1 = plot_training_curves(rankboost.training_errors, 
                               rankboost.training_exp_losses,
                               ax=axes[0])
    
    # Plot 2: Edge progression
    print("Plotting edge progression...")
    ax2 = plot_edge_progression(rankboost.edges, ax=axes[1])
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'outputs' / 'figures' / 'rankboost_training_curves.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving figure to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Done!")


if __name__ == "__main__":
    main()
