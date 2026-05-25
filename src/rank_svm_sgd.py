"""
RankSVM implementation using Stochastic Gradient Descent (SGD).

Implements Section 9.3 (Ranking with SVMs) of Foundations of Machine Learning.
Minimizes the hinge loss over pairwise differences:
L(w) = 1/2 ||w||^2 + C * sum_i max(0, 1 - y_i * w^T (x_pos_i - x_neg_i))
"""

import numpy as np


class RankSVM:
    """
    Pairwise RankSVM trained with Stochastic Gradient Descent (SGD).
    """
    
    def __init__(self, C=1.0, lr=0.01, n_epochs=100, random_seed=42):
        """
        Initialize RankSVM.
        
        Args:
            C (float): Regularization parameter trade-off
            lr (float): Learning rate for SGD
            n_epochs (int): Number of epochs
            random_seed (int): Random seed for reproducibility
        """
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_seed = random_seed
        self.w = None
        self.training_losses = []
        
    def fit(self, X_pairs, y_pairs, verbose=False):
        """
        Train RankSVM on pairwise data.
        
        Args:
            X_pairs (array): Pairwise features (n_pairs, 2, d)
            y_pairs (array): Pairwise labels (n_pairs,) - assumed to be +1 
                            since positive is at index 0, negative at index 1
            verbose (bool): Print progress
            
        Returns:
            self
        """
        np.random.seed(self.random_seed)
        n_pairs, _, d = X_pairs.shape
        
        # Initialize weights
        self.w = np.zeros(d)
        
        # Compute differences: x_pos - x_neg
        # Since y_pairs is all +1, the target margin is: w^T (x_pos - x_neg) >= 1
        X_diffs = X_pairs[:, 0, :] - X_pairs[:, 1, :]
        
        for epoch in range(self.n_epochs):
            # Shuffle data at each epoch
            shuffled_indices = np.random.permutation(n_pairs)
            X_diffs_shuffled = X_diffs[shuffled_indices]
            y_shuffled = y_pairs[shuffled_indices]
            
            epoch_loss = 0.0
            
            for i in range(n_pairs):
                x_diff = X_diffs_shuffled[i]
                y_val = y_shuffled[i]  # Should be +1
                
                # Check margin violation
                margin = y_val * np.dot(self.w, x_diff)
                
                # Subgradient update
                if margin < 1:
                    # Violating: w_next = w - lr * ( (1/n_pairs)*w - C * y_val * x_diff )
                    # We scale C appropriately
                    grad = (1.0 / n_pairs) * self.w - self.C * y_val * x_diff
                    self.w -= self.lr * grad
                    epoch_loss += 1.0 - margin
                else:
                    # Non-violating: w_next = w - lr * (1/n_pairs)*w
                    grad = (1.0 / n_pairs) * self.w
                    self.w -= self.lr * grad
            
            # Record total loss (regularization + hinge loss)
            reg_loss = 0.5 * np.dot(self.w, self.w)
            total_loss = reg_loss + self.C * epoch_loss
            self.training_losses.append(total_loss)
            
            if verbose and (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {total_loss:.4f} (Reg: {reg_loss:.4f}, Hinge: {self.C * epoch_loss:.4f})")
                
        return self
        
    def score(self, X):
        """
        Compute scoring function: s(x) = w^T x.
        
        Args:
            X (array): Input samples (n, d)
            
        Returns:
            array: Scores (n,)
        """
        if self.w is None:
            raise RuntimeError("Model is not trained yet.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.dot(X, self.w)
        
    def predict_order(self, X):
        """
        Predict ranking order (indices sorted by score, highest first).
        
        Args:
            X (array): Input samples (n, d)
            
        Returns:
            array: Sorted indices (descending)
        """
        scores = self.score(X)
        return np.argsort(-scores)
