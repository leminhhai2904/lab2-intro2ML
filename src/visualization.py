"""
Visualization utilities for ranking experiments.

Includes functions for:
- Scatter plots of 2D data
- Score contour plots
- ROC curves
- Training curves
- Margin histograms
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_2d_data(X_pos, X_neg, title="2D Ranking Data", ax=None):
    """
    Plot 2D scatter plot of positive and negative samples.
    
    Args:
        X_pos (array): Positive samples (n_pos, 2)
        X_neg (array): Negative samples (n_neg, 2)
        title (str): Plot title
        ax: Matplotlib axes object (optional)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', marker='o', s=100, 
               label='Positive', alpha=0.7, edgecolors='darkred', linewidth=2)
    ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', marker='x', s=100, 
               label='Negative', alpha=0.7)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_score_contours(X_pos, X_neg, score_func, title="Score Contours", ax=None):
    """
    Plot score contours and data points.
    
    Args:
        X_pos (array): Positive samples (n_pos, 2)
        X_neg (array): Negative samples (n_neg, 2)
        score_func (callable): Function that computes scores
        title (str): Plot title
        ax: Matplotlib axes object (optional)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid for contour plot
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Compute scores on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = score_func(grid_points).reshape(xx.shape)
    
    # Plot contours
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.6)
    ax.contour(xx, yy, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    plt.colorbar(contour, ax=ax, label='Score')
    
    # Plot data points
    ax.scatter(X_pos[:, 0], X_pos[:, 1], c='darkred', marker='o', s=80, 
               label='Positive', edgecolors='black', linewidth=1.5)
    ax.scatter(X_neg[:, 0], X_neg[:, 1], c='darkblue', marker='x', s=80, 
               label='Negative')
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    return ax


def plot_pairwise_examples(X_pairs, y_pairs, scores, n_examples=6, ax=None):
    """
    Visualize correct and incorrect pairwise rankings.
    
    Args:
        X_pairs (array): Pairwise data (n_pairs, 2, 2)
        y_pairs (array): Labels (n_pairs,)
        scores (array): Computed scores (n_pairs, 2)
        n_examples (int): Number of examples to show
        ax: Matplotlib axes object (optional)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    score_diffs = scores[:, 0] - scores[:, 1]
    correct = score_diffs > 0
    
    # Find examples of correct and incorrect rankings
    correct_indices = np.where(correct)[0]
    incorrect_indices = np.where(~correct)[0]
    
    n_each = n_examples // 2
    example_indices = np.concatenate([
        correct_indices[:n_each],
        incorrect_indices[:n_each]
    ])
    
    for idx, i in enumerate(example_indices):
        x_pos, x_neg = X_pairs[i, 0, :], X_pairs[i, 1, :]
        score_pos, score_neg = scores[i, 0], scores[i, 1]
        
        # Plot pair
        ax.plot([x_pos[0], x_neg[0]], [x_pos[1], x_neg[1]], 'k-', alpha=0.3)
        
        color = 'green' if correct[i] else 'red'
        marker_pos = 'o' if correct[i] else 'x'
        marker_neg = 's' if correct[i] else '+'
        
        ax.scatter(x_pos[0], x_pos[1], c=color, marker=marker_pos, s=200, 
                   edgecolors='black', linewidth=2, alpha=0.7)
        ax.scatter(x_neg[0], x_neg[1], c=color, marker=marker_neg, s=200, 
                   edgecolors='black', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Pairwise Rankings (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold')
    
    green_patch = mpatches.Patch(color='green', label='Correct')
    red_patch = mpatches.Patch(color='red', label='Incorrect')
    ax.legend(handles=[green_patch, red_patch], fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_training_curves(training_errors, training_exp_losses, edges=None, ax=None):
    """
    Plot RankBoost training curves.
    
    Args:
        training_errors (list): Pairwise errors per round
        training_exp_losses (list): Exponential losses per round
        edges (list): Edges per round (optional)
        ax: Matplotlib axes object (optional)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = np.arange(len(training_errors)) + 1
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(rounds, training_errors, 'b-o', linewidth=2, markersize=4, 
                    label='Pairwise Error')
    line2 = ax2.plot(rounds, training_exp_losses, 'r-s', linewidth=2, markersize=4, 
                     label='Exponential Loss')
    
    ax.set_xlabel('Boosting Round', fontsize=12)
    ax.set_ylabel('Pairwise Error', fontsize=12, color='b')
    ax2.set_ylabel('Exponential Loss', fontsize=12, color='r')
    ax.set_title('RankBoost Training Curves', fontsize=14, fontweight='bold')
    
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=11)
    
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_edge_progression(edges, ax=None):
    """
    Plot edge progression during boosting.
    
    Args:
        edges (list): Edge values per round
        ax: Matplotlib axes object (optional)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = np.arange(len(edges)) + 1
    
    ax.bar(rounds, edges, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Boosting Round', fontsize=12)
    ax.set_ylabel('Edge (ε⁺ - ε⁻)', fontsize=12)
    ax.set_title('Weak Ranker Edge per Round', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_roc_curve(fpr, tpr, auc=None, ax=None):
    """
    Plot ROC curve.
    
    Args:
        fpr (array): False Positive Rates
        tpr (array): True Positive Rates
        auc (float): Area Under Curve (optional)
        ax: Matplotlib axes object (optional)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    
    title = 'ROC Curve'
    if auc is not None:
        title += f' (AUC = {auc:.4f})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    return ax


def plot_margin_histogram(margins, rho=1.0, ax=None):
    """
    Plot histogram of pairwise margins.
    
    Args:
        margins (array): Pairwise margin values
        rho (float): Margin parameter for reference
        ax: Matplotlib axes object (optional)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(margins, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=rho, color='red', linestyle='--', linewidth=2, label=f'Margin ρ={rho}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Margin Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Pairwise Margins', fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_margin_loss_curve(rhos, margin_losses, ax=None):
    """
    Plot margin loss as function of margin parameter.
    
    Args:
        rhos (array): Margin values
        margin_losses (array): Margin loss values
        ax: Matplotlib axes object (optional)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rhos, margin_losses, 'b-o', linewidth=2, markersize=6)
    
    ax.set_xlabel('Margin Parameter ρ', fontsize=12)
    ax.set_ylabel('Empirical Margin Loss', fontsize=12)
    ax.set_title('Trade-off: Margin vs Loss', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_auc_comparison(auc_learned, auc_random, ax=None):
    """
    Plot AUC comparison between learned and random scores.
    
    Args:
        auc_learned (float): AUC for learned scores
        auc_random (float): AUC for random scores
        ax: Matplotlib axes object (optional)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Learned\nScores', 'Random\nScores']
    aucs = [auc_learned, auc_random]
    colors = ['green', 'lightcoral']
    
    bars = ax.bar(methods, aucs, color=colors, edgecolor='black', linewidth=2, alpha=0.7)
    
    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC: Learned vs Random Scores', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax
