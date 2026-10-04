"""Minimal Ridge Regression (L2-regularized linear regression) implementation."""
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        I = np.eye(d + 1)
        I[0, 0] = 0  # Do not regularize bias
        w = np.linalg.pinv(Xb.T @ Xb + self.alpha * I) @ Xb.T @ y
        self.intercept_ = w[0]
        self.coef_ = w[1:]
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

__all__ = ["RidgeRegression"]
