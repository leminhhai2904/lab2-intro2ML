"""
Enhanced RankBoost algorithm implementation.

Implements RankBoost with:
- Soft decision stumps (weighted predictions)
- Decision tree weak learners  
- More thresholds for better splitting
- Cross-validation support
"""

import numpy as np
try:
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SoftDecisionStump:
    """
    Soft decision stump - outputs weighted scores instead of {0,1}.
    Uses sigmoid to smooth transitions around threshold.
    """
    
    def __init__(self, feature_idx, threshold, polarity=1, temperature=1.0):
        """
        Initialize soft decision stump.
        
        Args:
            feature_idx (int): Feature index to split on
            threshold (float): Threshold value
            polarity (int): 1 or -1, determines output polarity
            temperature (float): Controls softness (lower = sharper, higher = softer)
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.polarity = polarity
        self.temperature = temperature
    
    def predict(self, X):
        """
        Make soft predictions using sigmoid.
        
        Args:
            X (array): Input data (n, d)
        
        Returns:
            array: Soft predictions in (0, 1)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Sigmoid: 1 / (1 + exp(-(x - threshold) / temperature))
        z = (X[:, self.feature_idx] - self.threshold) / self.temperature
        predictions = 1.0 / (1.0 + np.exp(-z))
        
        # Apply polarity
        if self.polarity == -1:
            predictions = 1 - predictions
        
        return predictions
    
    def __repr__(self):
        return f"SoftStump(f{self.feature_idx}, thresh={self.threshold:.2f}, pol={self.polarity})"


class DecisionStump:
    """Keep for backward compatibility - alias to SoftDecisionStump"""
    def __init__(self, feature_idx, threshold, polarity=1):
        self.stump = SoftDecisionStump(feature_idx, threshold, polarity, temperature=0.5)
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.polarity = polarity
    
    def predict(self, X):
        return self.stump.predict(X)
    
    def __repr__(self):
        return self.stump.__repr__()


class DecisionTreeRanker:
    """
    Decision tree weak ranker - more powerful than stumps.
    Uses sklearn's DecisionTreeClassifier with depth limit.
    """
    
    def __init__(self, max_depth=3, random_seed=42):
        """
        Initialize decision tree ranker.
        
        Args:
            max_depth (int): Maximum tree depth
            random_seed (int): Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for DecisionTreeRanker")
        self.tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_seed)
        self.trained = False
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit tree on data.
        
        Args:
            X (array): Training features (n, d)
            y (array): Training labels (n,)
            sample_weight (array): Sample weights (optional)
        """
        self.tree.fit(X, y, sample_weight=sample_weight)
        self.trained = True
    
    def predict_proba(self, X):
        """
        Get probability predictions.
        
        Args:
            X (array): Input data (n, d)
        
        Returns:
            array: Class 1 probability (n,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if not self.trained:
            raise RuntimeError("Tree not trained yet")
        
        # Return probability of positive class
        return self.tree.predict_proba(X)[:, 1]
    
    def __repr__(self):
        return f"DecisionTree(depth={self.tree.max_depth})"


class RankBoost:
    """
    Enhanced RankBoost algorithm for ranking.
    
    Supports:
    - Soft decision stumps (smooth predictions)
    - Decision tree weak learners (more expressive)
    - More candidate thresholds
    - Cross-validation for better generalization
    """
    
    def __init__(self, n_rounds=100, random_seed=42, weak_learner='soft_stump', 
                 n_thresholds=15, temperature=0.5):
        """
        Initialize RankBoost.
        
        Args:
            n_rounds (int): Number of boosting rounds
            random_seed (int): Random seed for reproducibility
            weak_learner (str): 'soft_stump' or 'tree'
            n_thresholds (int): Number of threshold candidates per feature
            temperature (float): Temperature for soft stumps (lower = sharper)
        """
        self.n_rounds = n_rounds
        self.random_seed = random_seed
        self.weak_learner = weak_learner
        self.n_thresholds = n_thresholds
        self.temperature = temperature
        np.random.seed(random_seed)
        
        self.weak_rankers = []
        self.alphas = []
        self.training_errors = []
        self.training_exp_losses = []
        self.edges = []
        self.n_features = None
        self.val_errors = []
    
    def fit(self, X_pairs, y_pairs, verbose=False, val_X_pairs=None, val_y_pairs=None):
        """
        Train RankBoost on pairwise data.
        
        Args:
            X_pairs (array): Pairwise training data (n_pairs, 2, d)
                           X_pairs[i, 0, :] = positive sample
                           X_pairs[i, 1, :] = negative sample
            y_pairs (array): Pairwise labels (n_pairs,) - all +1
            verbose (bool): Print progress
            val_X_pairs (array): Validation pairwise data (optional)
            val_y_pairs (array): Validation pairwise labels (optional)
        
        Returns:
            self
        """
        n_pairs = X_pairs.shape[0]
        self.n_features = X_pairs.shape[2]
        
        # Initialize weight distribution D_1(i) = 1/m for all pairs
        D = np.ones(n_pairs) / n_pairs
        
        for t in range(self.n_rounds):
            # Find best weak ranker
            best_ranker, best_alpha, weighted_error, edge = \
                self._find_best_weak_ranker(X_pairs, y_pairs, D)
            
            if best_ranker is None:
                if verbose:
                    print(f"Round {t}: No weak ranker found with positive edge")
                break
            
            # Store results
            self.weak_rankers.append(best_ranker)
            self.alphas.append(best_alpha)
            self.edges.append(edge)
            
            # Compute predictions from new weak ranker
            if isinstance(best_ranker, SoftDecisionStump):
                h_pos = best_ranker.predict(X_pairs[:, 0, :])
                h_neg = best_ranker.predict(X_pairs[:, 1, :])
            elif isinstance(best_ranker, DecisionTreeRanker):
                h_pos = best_ranker.predict_proba(X_pairs[:, 0, :])
                h_neg = best_ranker.predict_proba(X_pairs[:, 1, :])
            else:
                h_pos = best_ranker.predict(X_pairs[:, 0, :])
                h_neg = best_ranker.predict(X_pairs[:, 1, :])
            
            correct_ranking = (h_pos > h_neg)
            
            # Update weights
            score_diff = h_pos - h_neg
            update_factors = np.exp(-best_alpha * score_diff)
            D = D * update_factors
            Z_t = np.sum(D)
            if Z_t > 0:
                D = D / Z_t
            
            # Compute training metrics
            current_scores = self._compute_ensemble_scores(X_pairs)
            score_diffs = current_scores[:, 0] - current_scores[:, 1]
            pairwise_error = np.mean(score_diffs <= 0)
            exp_loss = np.mean(np.exp(-score_diffs))
            
            self.training_errors.append(pairwise_error)
            self.training_exp_losses.append(exp_loss)
            
            # Validation error
            if val_X_pairs is not None:
                val_scores = self._compute_ensemble_scores(val_X_pairs)
                val_score_diffs = val_scores[:, 0] - val_scores[:, 1]
                val_error = np.mean(val_score_diffs <= 0)
                self.val_errors.append(val_error)
            
            if verbose and (t + 1) % max(1, self.n_rounds // 10) == 0:
                msg = f"Round {t+1}/{self.n_rounds}: error={pairwise_error:.4f}, exp_loss={exp_loss:.4f}, edge={edge:.6f}"
                if val_X_pairs is not None:
                    msg += f", val_error={val_error:.4f}"
                print(msg)
        
        return self
    
    def _find_best_weak_ranker(self, X_pairs, y_pairs, D):
        """
        Find best weak ranker for current distribution.
        
        For soft_stump: searches all features and thresholds
        For tree: trains decision trees
        """
        n_pairs = X_pairs.shape[0]
        n_features = X_pairs.shape[2]
        
        best_ranker = None
        best_error = 1.0
        best_edge = 0
        best_alpha = 0
        
        if self.weak_learner == 'soft_stump':
            # Try all features and thresholds
            for feature_idx in range(n_features):
                X_pos_feat = X_pairs[:, 0, feature_idx]
                X_neg_feat = X_pairs[:, 1, feature_idx]
                
                # Use quantiles as candidate thresholds (more thresholds = better)
                all_vals = np.concatenate([X_pos_feat, X_neg_feat])
                thresholds = np.percentile(all_vals, np.linspace(5, 95, self.n_thresholds))
                
                # Try both polarities
                for polarity in [1, -1]:
                    for threshold in thresholds:
                        stump = SoftDecisionStump(feature_idx, threshold, polarity, 
                                                 temperature=self.temperature)
                        
                        # Compute predictions
                        h_pos = stump.predict(X_pairs[:, 0, :])
                        h_neg = stump.predict(X_pairs[:, 1, :])
                        
                        # Compute weighted errors
                        correct_ranking = (h_pos > h_neg)
                        incorrect_ranking = ~correct_ranking
                        
                        # Compute epsilon+ and epsilon-
                        epsilon_plus = np.sum(D * incorrect_ranking)
                        epsilon_minus = np.sum(D * correct_ranking)
                        
                        # Edge
                        edge = epsilon_minus - epsilon_plus
                        
                        # Keep if better edge
                        if edge > best_edge:
                            best_error = epsilon_plus
                            best_ranker = stump
                            best_edge = edge
        
        elif self.weak_learner == 'tree':
            # Create training data: all samples and their labels
            X_all = np.vstack([X_pairs[:, 0, :], X_pairs[:, 1, :]])
            y_all = np.hstack([np.ones(n_pairs), np.zeros(n_pairs)])
            weights_all = np.hstack([D, D])
            
            # Try different tree depths
            for depth in [1, 2, 3, 4]:
                try:
                    tree = DecisionTreeRanker(max_depth=depth, random_seed=self.random_seed)
                    tree.fit(X_all, y_all, sample_weight=weights_all)
                    
                    # Compute predictions
                    h_pos = tree.predict_proba(X_pairs[:, 0, :])
                    h_neg = tree.predict_proba(X_pairs[:, 1, :])
                    
                    # Compute weighted errors
                    correct_ranking = (h_pos > h_neg)
                    incorrect_ranking = ~correct_ranking
                    
                    epsilon_plus = np.sum(D * incorrect_ranking)
                    epsilon_minus = np.sum(D * correct_ranking)
                    
                    edge = epsilon_minus - epsilon_plus
                    
                    # Keep if better edge
                    if edge > best_edge:
                        best_error = epsilon_plus
                        best_ranker = tree
                        best_edge = edge
                
                except Exception as e:
                    if 'sklearn' not in str(e).lower():
                        continue
        
        if best_ranker is None or best_edge <= 0:
            return None, None, best_error, 0
        
        # Compute alpha
        epsilon_plus = best_error
        epsilon_minus = 1.0 - epsilon_plus
        
        epsilon = 1e-10
        epsilon_plus = np.clip(epsilon_plus, epsilon, 1 - epsilon)
        epsilon_minus = np.clip(epsilon_minus, epsilon, 1 - epsilon)
        best_alpha = 0.5 * np.log(epsilon_minus / epsilon_plus)
        best_alpha = np.clip(best_alpha, -10, 10)
        
        return best_ranker, best_alpha, best_error, best_edge
    
    def _compute_weak_scores(self, X, ranker):
        """Compute scores from a single weak ranker on pairwise data."""
        scores = np.zeros((X.shape[0], 2))
        
        if isinstance(ranker, SoftDecisionStump):
            scores[:, 0] = ranker.predict(X[:, 0, :])
            scores[:, 1] = ranker.predict(X[:, 1, :])
        elif isinstance(ranker, DecisionStump):
            scores[:, 0] = ranker.predict(X[:, 0, :])
            scores[:, 1] = ranker.predict(X[:, 1, :])
        elif isinstance(ranker, DecisionTreeRanker):
            scores[:, 0] = ranker.predict_proba(X[:, 0, :])
            scores[:, 1] = ranker.predict_proba(X[:, 1, :])
        else:
            scores[:, 0] = ranker.predict(X[:, 0, :])
            scores[:, 1] = ranker.predict(X[:, 1, :])
        
        return scores
    
    def _compute_ensemble_scores(self, X):
        """Compute ensemble scores on pairwise data."""
        scores = np.zeros((X.shape[0], 2))
        
        for alpha, ranker in zip(self.alphas, self.weak_rankers):
            weak_scores = self._compute_weak_scores(X, ranker)
            scores += alpha * weak_scores
        
        return scores
    
    def score(self, X):
        """
        Compute ensemble scores for samples.
        
        Args:
            X (array): Input samples (n, d)
        
        Returns:
            array: Ranking scores (n,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        scores = np.zeros(X.shape[0])
        
        for alpha, ranker in zip(self.alphas, self.weak_rankers):
            if isinstance(ranker, (SoftDecisionStump, DecisionStump)):
                scores += alpha * ranker.predict(X)
            elif isinstance(ranker, DecisionTreeRanker):
                scores += alpha * ranker.predict_proba(X)
            else:
                scores += alpha * ranker.predict(X)
        
        return scores
    
    def predict_order(self, X):
        """
        Predict ranking order (indices sorted by score, highest first).
        
        Args:
            X (array): Input samples (n, d)
        
        Returns:
            array: Indices sorted by predicted score (descending)
        """
        scores = self.score(X)
        return np.argsort(-scores)  # Descending order
