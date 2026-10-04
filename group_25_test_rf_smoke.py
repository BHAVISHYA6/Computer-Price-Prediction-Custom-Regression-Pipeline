"""Quick smoke test for the in-repo RandomForestRegressor.

Creates a small synthetic regression problem, fits the RandomForestRegressor
and prints a simple MSE to show it runs end-to-end.
"""
import numpy as np
from group_25_random_forest import RandomForestRegressor

def main():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 6)
    # target is a noisy linear combination
    coef = np.array([1.5, -2.0, 0.5, 0.0, 1.0, -1.0])
    y = X.dot(coef) + rng.randn(200) * 0.5

    rf = RandomForestRegressor(n_estimators=8, max_depth=6, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    mse = float(np.mean((preds - y) ** 2))
    print(f"Smoke test MSE: {mse:.4f}")
    print("feature_importances_ (first 6):", rf.feature_importances_[:6])

if __name__ == '__main__':
    main()
