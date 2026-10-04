"""Lightweight Random Forest Regressor implementation.

This is a minimal, self-contained RandomForestRegressor that supports
fit(X, y), predict(X) and exposes `feature_importances_` similar to scikit-learn.

Notes:
- Only supports numeric features (the project's preprocessing produces numeric features).
- Single-threaded and intentionally simple for clarity and reproducibility.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional


def _mse(y):
    if len(y) == 0:
        return 0.0
    mean = np.mean(y)
    return float(np.mean((y - mean) ** 2))


class _TreeNode:
    __slots__ = ("is_leaf", "value", "feature", "threshold", "left", "right", "n_samples")

    def __init__(self):
        self.is_leaf = True
        self.value = 0.0
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.n_samples = 0


class DecisionTreeRegressor:
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2,
                 max_features: Optional[int] = None, random_state: Optional[int] = None):
        self.max_depth = max_depth if max_depth is None or max_depth >= 1 else None
        self.min_samples_split = max(2, int(min_samples_split))
        self.max_features = max_features
        self.random_state = np.random.RandomState(random_state)
        self.n_features_in_ = None
        self.feature_importances_ = None
        self._root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        # Determine max_features
        if self.max_features is None:
            max_feats = max(1, int(math.sqrt(n_features)))
        else:
            max_feats = int(self.max_features)

        self._feature_importances = np.zeros(n_features, dtype=float)

        def build_node(idx, depth=0) -> _TreeNode:
            node = _TreeNode()
            node.n_samples = idx.size
            y_node = y[idx]
            node.value = float(np.mean(y_node)) if idx.size > 0 else 0.0
            # stopping
            if (self.max_depth is not None and depth >= self.max_depth) or idx.size < self.min_samples_split or np.all(y_node == y_node[0]):
                node.is_leaf = True
                return node

            best_feat = None
            best_thr = None
            best_imp = 0.0
            best_left_idx = None
            best_right_idx = None

            # sample subset of features
            feats = np.arange(n_features)
            if max_feats < n_features:
                feats = self.random_state.choice(n_features, size=max_feats, replace=False)

            parent_imp = _mse(y_node)

            for f in feats:
                xv = X[idx, f]
                # consider only unique split points
                uniq = np.unique(xv[~np.isnan(xv)])
                if uniq.size <= 1:
                    continue
                # candidate thresholds are midpoints
                thr_candidates = (uniq[:-1] + uniq[1:]) / 2.0
                for thr in thr_candidates:
                    left_mask = xv <= thr
                    right_mask = ~left_mask
                    if left_mask.sum() < 1 or right_mask.sum() < 1:
                        continue
                    left_y = y_node[left_mask]
                    right_y = y_node[right_mask]
                    imp = parent_imp - (left_y.size / y_node.size) * _mse(left_y) - (right_y.size / y_node.size) * _mse(right_y)
                    if imp > best_imp:
                        best_imp = imp
                        best_feat = f
                        best_thr = thr
                        # convert to global indices
                        best_left_idx = idx[left_mask]
                        best_right_idx = idx[right_mask]

            if best_feat is None:
                node.is_leaf = True
                return node

            # record importance contribution (weighted by number of samples)
            self._feature_importances[best_feat] += best_imp * idx.size

            node.is_leaf = False
            node.feature = int(best_feat)
            node.threshold = float(best_thr)
            node.left = build_node(best_left_idx, depth + 1)
            node.right = build_node(best_right_idx, depth + 1)
            return node

        all_idx = np.arange(n_samples)
        self._root = build_node(all_idx, 0)
        # normalize importances
        total = self._feature_importances.sum()
        if total > 0:
            self.feature_importances_ = self._feature_importances / total
        else:
            self.feature_importances_ = np.zeros(n_features, dtype=float)
        return self

    def _predict_one(self, x):
        node = self._root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X)
        out = np.array([self._predict_one(row) for row in X], dtype=float)
        return out


class RandomForestRegressor:
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, max_features: Optional[object] = 'sqrt',
                 bootstrap: bool = True, random_state: Optional[int] = None, n_jobs: int = 1):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_features = max_features
        self.estimators_ = []
        self.feature_importances_ = None

    def _resolve_max_features(self, n_features):
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, n_features))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if self.max_features == 'sqrt' or self.max_features is None:
            return max(1, int(math.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(math.log2(n_features)))
        return max(1, int(self.max_features))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        importances = np.zeros(n_features, dtype=float)
        for i in range(self.n_estimators):
            seed = None if self.random_state is None else rng.randint(0, 2 ** 31 - 1)
            if self.bootstrap:
                sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
            else:
                sample_idx = np.arange(n_samples)
            max_feats = self._resolve_max_features(n_features)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=max_feats, random_state=seed)
            tree.fit(X[sample_idx], y[sample_idx])
            self.estimators_.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        # Normalize across trees
        total = importances.sum()
        if total > 0:
            self.feature_importances_ = importances / total
        else:
            self.feature_importances_ = np.zeros(n_features, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.vstack([est.predict(X) for est in self.estimators_])
        return preds.mean(axis=0)


__all__ = ["RandomForestRegressor", "DecisionTreeRegressor"]
