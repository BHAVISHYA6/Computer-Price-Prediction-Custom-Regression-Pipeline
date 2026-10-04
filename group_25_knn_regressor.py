"""Minimal K-Nearest Neighbors (KNN) Regressor implementation."""
import numpy as np

class KNNRegressor:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for x in X:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            idx = np.argpartition(dists, self.k)[:self.k]
            preds.append(np.mean(self.y_train[idx]))
        return np.array(preds)

__all__ = ["KNNRegressor"]
