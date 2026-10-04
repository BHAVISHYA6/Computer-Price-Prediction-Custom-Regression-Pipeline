# Computer Price Prediction — Custom Regression Pipeline

Overview
--------
This repository contains a small end-to-end pipeline to predict laptop/desktop computer prices from a dataset. The project includes lightweight, from-scratch implementations of multiple regressors and an interactive Streamlit tester.

Key points
- Four regressors implemented in-repo: Random Forest, Ordinary Least Squares Linear Regression, Ridge Regression (L2), and a KNN regressor.
- The pipeline performs quick EDA, feature engineering, model training, and saves artifacts (plots, model joblibs, metrics) under `artifacts/`.
- An interactive app (`group_25_app_streamlit.py`) lets you run single-row or batch predictions and compare models.

Files in this repository (current)
----------------------------------
Top-level files and folders (accurate as of this workspace):

- `.venv/`, `.venv-1/` — optional virtual environments (may be present)
- `artifacts/` — output directory with generated plots, metrics and trained models
- `computer_prices_all.csv` — dataset (CSV)
- `group_25_app_streamlit.py` — Streamlit interactive tester (project-local, loads models from `artifacts/`)
- `group_25_run_pipeline.py` — main pipeline: loads data, featurizes, trains models, evaluates and writes artifacts
- `group_25_test_rf_smoke.py` — quick smoke test for the in-repo RandomForest implementation
- `group_25_random_forest.py` — in-repo RandomForest implementation (duplicate/variant naming)
- `group_25_linear_regression.py` — linear regression implementation (duplicate/variant)
- `group_25_ridge_regression.py` — ridge regression implementation (duplicate/variant)
- `group_25_knn_regressor.py` — knn regressor implementation (duplicate/variant)
- `requirements.txt` — Python dependencies

Note: The project also contains model implementation files under names without the `group_25_` prefix in some workflows; the Streamlit app and pipeline in this workspace use the `group_25_`-prefixed scripts.

Contents of `artifacts/` (current)
----------------------------------
Files present in `artifacts/` in this workspace:

- `price_dist.png` — price distribution histogram
- `corr_matrix.png` — correlation heatmap (first 40 numeric features)
- `top_features.png` — top feature importances plot
- `rf_model.joblib` — RandomForest model + feature list
- `lr_model.joblib` — LinearRegression model + feature list
- `ridge_model.joblib` — RidgeRegression model + feature list
- `knn_model.joblib` — KNN model + feature list
- `metrics.txt` — evaluation metrics (MSE, R²) for trained models

Quick start (macOS / Linux)
---------------------------
1. Create and activate a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Run the training & evaluation pipeline (from project root):

```bash
python group_25_run_pipeline.py
# Optionally, set number of trees for the in-repo RF implementation:
# RF_N_ESTIMATORS=50 python group_25_run_pipeline.py
```

3. Launch the Streamlit app for interactive testing:

```bash
streamlit run group_25_app_streamlit.py
```

Useful quick commands
---------------------
- Run a small smoke test for the RandomForest implementation:

```bash
python group_25_test_rf_smoke.py
```

- Inspect generated metrics:

```bash
cat artifacts/metrics.txt
```

What the pipeline produces
-------------------------
- `artifacts/price_dist.png` — price histogram from the dataset
- `artifacts/corr_matrix.png` — correlation heatmap (first 40 numeric features)
- `artifacts/top_features.png` — top feature importances plot
- `artifacts/rf_model.joblib`, `artifacts/lr_model.joblib`, `artifacts/ridge_model.joblib`, `artifacts/knn_model.joblib` — saved model objects (joblib) containing `{'model': <model>, 'features': [...]}`
- `artifacts/metrics.txt` — MSE and R² for each trained model

Streamlit app features
----------------------
- Loads all available models from `artifacts/` and displays a list in the sidebar.
- Single-input mode: pick a model (or "Compare all") and get a predicted price for a manually entered row.
- Batch mode: upload a CSV and generate predictions from one or all models, download results.
- When available, the app shows top features (feature importances or coefficients) for the chosen model.

Notes and troubleshooting
-------------------------
- The in-repo Random Forest is a simple, single-threaded implementation intended for transparency and teaching; it's not optimized for production performance.
- If you get ImportError or the editor shows "module not resolved", run scripts from the project root so local modules can be imported, or add the project root to `PYTHONPATH`.
- If the Streamlit app fails to find models, run `python group_25_run_pipeline.py` first to generate joblib artifacts.
- Virtual environments (`.venv` or `.venv-1`) live in the repo for convenience — delete them if you prefer to recreate fresh environments.

Further work you might consider
------------------------------
- Add cross-validation and hyperparameter tuning (GridSearch or randomized CV).
- Replace simple frequency encoding with target-encoding with appropriate CV to reduce leakage.
- Implement a faster RF (use scikit-learn) or make the custom RF multi-threaded.
- Add unit tests and CI workflow to run smoke tests automatically.

Contact / license
------------------
This repository is an educational/demo project. Feel free to adapt for experiments. If you'd like me to expand the README with performance numbers, LaTeX report generation, or CI configuration, tell me what you want and I'll add it.
