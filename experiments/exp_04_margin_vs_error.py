"""
Experiment 4: Margin vs Error Trade-off

Analyzes the relationship between margin parameter ρ and empirical margin loss.
Demonstrates the margin-based generalization bound concept.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import generate_synthetic_data, create_pairwise_data
from rankboost import RankBoost
from metrics import margin_loss
from visualization import plot_margin_histogram, plot_margin_loss_curve


def main():
    """Run experiment 4: Margin vs error trade-off."""
    print("=" * 70)
    print("Experiment 4: Margin vs Error Trade-off")
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
    print("Training RankBoost...")
    rankboost = RankBoost(n_rounds=50, random_seed=random_seed)
    rankboost.fit(X_pairs, y_pairs, verbose=False)
    
    # Compute ensemble scores on all pairs
    print("Computing ensemble scores...")
    scores = rankboost._compute_ensemble_scores(X_pairs)
    
    # Compute margins
    margins = scores[:, 0] - scores[:, 1]  # h(x_pos) - h(x_neg)
    
    print(f"  Mean margin: {margins.mean():.4f}")
    print(f"  Min margin: {margins.min():.4f}")
    print(f"  Max margin: {margins.max():.4f}")
    
    # Compute margin loss for different rho values
    print("\nComputing margin loss for different margin parameters...")
    rhos = np.linspace(0, 3.0, 50)
    margin_losses = []
    
    for rho in rhos:
        loss = margin_loss(X_pairs, y_pairs, scores, rho=rho)
        margin_losses.append(loss)
        if len(margin_losses) % 10 == 0:
            print(f"  rho = {rho:.2f}, margin loss = {loss:.4f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Margin histogram
    print("\nPlotting margin histogram...")
    ax1 = plot_margin_histogram(margins, rho=1.0, ax=axes[0])
    
    # Plot 2: Margin loss curve
    print("Plotting margin loss curve...")
    ax2 = plot_margin_loss_curve(rhos, margin_losses, ax=axes[1])
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'outputs' / 'figures' / 'margin_vs_error.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving figure to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Done!")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 SUMMARY")
    print("=" * 70)
    print("Margin Analysis:")
    print(f"  Fraction with margin > 0: {np.sum(margins > 0) / len(margins):.4f}")
    print(f"  Fraction with margin > 1: {np.sum(margins > 1) / len(margins):.4f}")
    print(f"  Fraction with margin > 2: {np.sum(margins > 2) / len(margins):.4f}")
    
    print("\nMargin Loss Trade-off:")
    print(f"  Loss at rho=0.0: {margin_losses[0]:.4f}")
    print(f"  Loss at rho=1.0: {margin_losses[np.argmin(np.abs(rhos - 1.0))]:.4f}")
    print(f"  Loss at rho=2.0: {margin_losses[np.argmin(np.abs(rhos - 2.0))]:.4f}")
    
    print("\nInterpretation:")
    print("- Larger margin rho means stricter ranking requirement")
    print("- Larger rho typically increases margin loss L_rho")
    print("- Trade-off: more margin -> better generalization (but higher training loss)")
    print("- Margin-based bounds relate generalization error to this trade-off")


if __name__ == "__main__":
    main()
