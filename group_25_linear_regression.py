"""Minimal Linear Regression implementation (ordinary least squares) for comparison."""
import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        # Add bias term
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        # Closed-form solution: w = (X^T X)^{-1} X^T y
        w = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
        self.intercept_ = w[0]
        self.coef_ = w[1:]
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

__all__ = ["LinearRegression"]
