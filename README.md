# Ranking Experiments

Implementation of ranking algorithms and experiments for demonstrating Chapter 9 (Ranking) concepts from *Foundations of Machine Learning*.

## Overview

This project implements **RankBoost**, a boosting algorithm for ranking, along with associated metrics and visualization tools. All algorithms are implemented from scratch without using pre-built libraries like scikit-learn or libsvm for the algorithmic components.

## Environment

- **Python version**: 3.8+
- **NumPy version**: 1.24.3
- **SciPy version**: 1.11.4
- **Matplotlib version**: 3.8.2
- **Pandas version**: 2.1.3

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone or download this repository

2. Navigate to the project directory:
```bash
cd lab2-intro2ML
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
code/
├── README.md                  # This file
├── requirements.txt           # Python dependencies and versions
├── src/
│   ├── __init__.py
│   ├── data.py               # Synthetic data generation
│   ├── metrics.py            # Ranking metrics (AUC, ROC, pairwise error, etc.)
│   ├── rankboost.py          # RankBoost algorithm implementation
│   └── visualization.py      # Plotting utilities
├── experiments/
│   ├── exp_01_pairwise_ranking_demo.py    # Experiment 1: Data visualization
│   ├── exp_02_rankboost_demo.py           # Experiment 2: Training curves
│   ├── exp_03_auc_roc_demo.py             # Experiment 3: ROC and AUC
│   └── exp_04_margin_vs_error.py          # Experiment 4: Margin analysis
└── outputs/
    └── figures/              # Generated plots
```

## Key Components

### Synthetic Data (`src/data.py`)

Generates 2D bipartite ranking data:
- **Positive class**: Centered around (0.5, 0.5) with Gaussian noise
- **Negative class**: Centered around (-0.5, -0.5) with Gaussian noise
- **Overlap**: ~50% (moderate overlap for challenging but learnable ranking problem)
- **Noise**: σ = 1.5 (empirically tuned for optimal training)
- **Dataset size**: 100 positive + 100 negative samples for fast experiments
- True scoring function: s(x) = w^T x

```python
from src.data import generate_synthetic_data

data = generate_synthetic_data(n_positive=100, n_negative=100, 
                                noise_std=1.5, random_seed=42)
X = data['X']                    # All samples
y = data['y']                    # Labels (-1 or +1)
X_positive = data['X_positive']  # Positive samples
X_negative = data['X_negative']  # Negative samples
```

### Metrics (`src/metrics.py`)

Implements ranking metrics from scratch:
- **Pairwise misranking error**: Fraction of incorrectly ranked pairs
- **AUC**: Area under ROC curve (ranking accuracy)
- **ROC curve**: False positive rate vs true positive rate
- **Margin loss**: Empirical loss with margin parameter ρ
- **Exponential loss**: Loss function used in RankBoost

All metrics are computed from first principles without sklearn utilities.

### RankBoost Algorithm (`src/rankboost.py`)

Complete implementation of RankBoost with enhanced soft decision stumps for pairwise ranking:

```python
from src.rankboost import RankBoost
from src.data import create_pairwise_data

# Create pairwise training data
pair_data = create_pairwise_data(X, y)
X_pairs = pair_data['X_pairs']  # (n_pairs, 2, d)
y_pairs = pair_data['y_pairs']  # (n_pairs,) - all +1

# Train RankBoost
rankboost = RankBoost(n_rounds=50, random_seed=42)
rankboost.fit(X_pairs, y_pairs, verbose=True)

# Make predictions
scores = rankboost.score(X)              # Continuous scores
ranking = rankboost.predict_order(X)     # Ranking order (indices)
```

**Key features:**
- **Base learner**: Soft Decision Stumps with sigmoid smoothing (continuous [0,1] output)
- **Weak ranker selection**: Chooses stump with maximum edge (ε⁺ - ε⁻)
- **Weight update**: Uses exponential loss for distribution update
- **Output**: Weighted ensemble of weak rankers
- **Enhancement**: Soft stumps provide better margin distribution than binary stumps

**Algorithm outline:**
1. Initialize uniform weight distribution over pairwise samples
2. For each round:
   - Find weak ranker with maximum edge
   - Compute weight α based on edge
   - Update weight distribution with exponential loss
3. Output ensemble: g(x) = Σ_t α_t h_t(x)

### Visualization (`src/visualization.py`)

Comprehensive plotting utilities:
- 2D scatter plots with positive/negative classes
- Score contour plots with decision boundaries
- Pairwise ranking examples (correct vs incorrect)
- Training curves (error, exponential loss, edge)
- ROC curves with AUC values
- Margin histograms and loss trade-off curves

## Experiments

### Experiment 1: Pairwise Ranking Demonstration

**Goal**: Visualize score-based ranking on 2D synthetic data

**Run**:
```bash
python experiments/exp_01_pairwise_ranking_demo.py
```

**Output**: 
- `outputs/figures/pairwise_ranking_demo.png`
- Shows scatter plot and score contours

**What it demonstrates**:
- True data distribution (positive vs negative)
- True scoring function and its contour levels
- How score function induces ranking order

---

### Experiment 2: RankBoost Training Curves

**Goal**: Verify that RankBoost reduces training error and exponential loss

**Run**:
```bash
python experiments/exp_02_rankboost_demo.py
```

**Output**:
- `outputs/figures/rankboost_training_curves.png`
- Shows pairwise error and exponential loss decreasing over rounds
- Shows edge (γ) of weak rankers per round

**What it demonstrates**:
- **Empirical error decreases** as rounds increase (with positive edge)
- **Exponential loss follows similar trend** (surrogate loss)
- **Edge exploitation**: RankBoost finds weak rankers with positive advantage
- Theory: With edges γ > 0, error decreases as exp(-2γ²T)

---

### Experiment 3: ROC Curve and AUC

**Goal**: Compare AUC between learned and random ranking

**Run**:
```bash
python experiments/exp_03_auc_roc_demo.py
```

**Output**:
- `outputs/figures/roc_auc_analysis.png`
- ROC curves for learned model and random baseline
- AUC comparison bar chart

**Key results**:
- Learned model AUC >> Random AUC
- AUC = probability that positive score > negative score
- Higher AUC means better ranking performance

**What it demonstrates**:
- **AUC as ranking accuracy**: AUC = pairwise ranking accuracy
- **ROC curve structure**: Trade-off between TPR and FPR at different thresholds
- **Perfect ranking**: AUC = 1.0 means no threshold separates pos/neg

---

### Experiment 4: Margin vs Error Trade-off

**Goal**: Analyze margin-based generalization bounds

**Run**:
```bash
python experiments/exp_04_margin_vs_error.py
```

**Output**:
- `outputs/figures/margin_vs_error.png`
- Histogram of pairwise margins
- Margin loss curve as function of ρ

**What it demonstrates**:
- **Margin distribution**: How many pairs have large margins
- **Trade-off curve**: Larger ρ → higher margin loss L_ρ
- **Margin-based bound relation**: Bound depends on margin violations
- Theory: Error + regularization term involves R_ρ(h) (margin loss)

---

## Reproducibility

All experiments use fixed random seeds for reproducibility:

- **Data generation**: `random_seed=42`
- **RankBoost training**: `random_seed=42`
- **All experiments**: Seeded in each experiment file

### Re-run all experiments:

```bash
python experiments/exp_01_pairwise_ranking_demo.py
python experiments/exp_02_rankboost_demo.py
python experiments/exp_03_auc_roc_demo.py
python experiments/exp_04_margin_vs_error.py
```

### Expected output:
- 4 PNG figures in `outputs/figures/`
- Console output showing metrics and progress

## Algorithm Details

### Decision Stump

Base weak ranker: splits on feature threshold.

```
h(x) = 1 if x[j] >= threshold else 0
```

Output: {0, 1} for each sample

### RankBoost Update

1. **Weak ranker selection**: Among all decision stumps, choose one that maximizes
   - Edge: γ = ε⁺ - ε⁻
   - Where ε⁺ = Σ_i D(i)(1 - h_pos_i), ε⁻ = Σ_i D(i) h_neg_i

2. **Weight α**: 
   ```
   α = (1/2) log(ε⁺ / ε⁻)
   ```

3. **Distribution update**:
   ```
   D_{t+1}(i) ← D_t(i) exp(-α · h(x_pos) + α · h(x_neg)) / Z_t
   ```
   Equivalently: `D_{t+1}(i) ← D_t(i) exp(-α(h(x_pos) - h(x_neg))) / Z_t`

### Output Scoring Function

```
g(x) = Σ_{t=1}^T α_t h_t(x)
```

The ensemble score is the weighted sum of weak ranker outputs.

## Code Snippets for Report

### Synthetic Data Generation

```python
def generate_synthetic_data(n_positive=100, n_negative=100, noise_std=0.5):
    np.random.seed(42)
    X_positive = np.random.randn(n_positive, 2) * noise_std + np.array([2.0, 2.0])
    X_negative = np.random.randn(n_negative, 2) * noise_std + np.array([-2.0, -2.0])
    X = np.vstack([X_positive, X_negative])
    y = np.hstack([np.ones(n_positive), -np.ones(n_negative)])
    return X, y, X_positive, X_negative
```

### Pairwise Error Calculation

```python
def pairwise_misranking_error(X_pairs, scores):
    """
    Compute fraction of pairs where negative score >= positive score.
    """
    scores_pos = scores[:, 0]
    scores_neg = scores[:, 1]
    misranked = np.sum(scores_neg >= scores_pos)
    return misranked / len(X_pairs)
```

### AUC from Scratch

```python
def auc_from_scratch(X_pos, X_neg, score_func):
    """
    AUC = P(score(x_pos) > score(x_neg))
    """
    scores_pos = score_func(X_pos)
    scores_neg = score_func(X_neg)
    
    correct_rankings = 0
    for sp in scores_pos:
        correct_rankings += np.sum(scores_neg < sp)
    
    return correct_rankings / (len(scores_pos) * len(scores_neg))
```

### RankBoost Weight Update (Core Step)

```python
# In each round, after selecting weak ranker h_t:
scores_t = compute_weak_scores(X_pairs, h_t)
score_diffs = scores_t[:, 0] - scores_t[:, 1]

# Update distribution
update_factors = np.exp(-alpha_t * score_diffs)  # y_i = +1
D = D * update_factors
D = D / np.sum(D)  # Normalize
```

## Dependencies

All dependencies are specified in `requirements.txt` with pinned versions:

```
numpy==1.24.3
scipy==1.11.4
matplotlib==3.8.2
pandas==2.1.3
```

**Why these versions?**
- **NumPy 1.24.3**: Stable linear algebra operations
- **SciPy 1.11.4**: Required by numpy, used for scipy.stats if needed
- **Matplotlib 3.8.2**: Stable plotting with 3D support
- **Pandas 2.1.3**: Optional, used for result summary CSV

## Troubleshooting

### Import errors

If you get "ModuleNotFoundError", make sure:
1. You installed requirements: `pip install -r requirements.txt`
2. You're running from the correct directory
3. Python path includes `src/`: experiments automatically add it via `sys.path.insert(0, ...)`

### Visualization not showing

Plots are saved to `outputs/figures/` regardless. If you want to display:
```python
import matplotlib.pyplot as plt
plt.show()
```

### Slow training

RankBoost with 100 rounds on ~10,000 pairs should complete in <3 seconds. 
If slower:
- Reduce `n_rounds` in experiment files (default: 100)
- Reduce number of threshold candidates (currently 15 per feature)
- Use smaller dataset (reduce n_positive/n_negative in data generation)

## Notes on Implementation

- **No sklearn used** for RankBoost, metrics, or data generation
- **NumPy only** for linear algebra and numerical operations
- **Matplotlib only** for visualization
- **Decision stumps**: Simple but effective weak learners
- **Efficiency**: ~O(n_pairs × n_features × n_thresholds) per round

## References

- Mohri, Rostamizadeh, Talwalkar (2018). *Foundations of Machine Learning*
- Chapter 9: Ranking
- Freund et al. (2003). "An efficient algorithm for learning to rank"

---

**Date**: 2026  
**Random Seed for Reproducibility**: 42