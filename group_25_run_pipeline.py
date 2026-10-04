"""Run a quick EDA, preprocess the dataset and train a Random Forest baseline.
Outputs:
 - console metrics (MSE, R2)
 - top 20 feature importances printed
 - saved model: `rf_model.joblib`
 - saved plots: `price_dist.png`, `corr_matrix.png`, `top_features.png`

This script is intended to be runnable in the project venv.
"""

import os
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from group_25_random_forest import RandomForestRegressor
from group_25_linear_regression import LinearRegression
from group_25_ridge_regression import RidgeRegression
from group_25_knn_regressor import KNNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / 'computer_prices_all.csv'
OUT_DIR = BASE / 'artifacts'
OUT_DIR.mkdir(exist_ok=True)

print('Loading data from', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)

# Quick EDA plots
plt.figure(figsize=(6,4))
sns.histplot(df['price'], bins=80, kde=True)
plt.title('Price distribution')
plt.tight_layout()
plt.savefig(OUT_DIR / 'price_dist.png')
plt.close()

# Feature engineering
# Parse resolution into width/height
def parse_res(r):
    try:
        w,h = r.split('x')
        return int(w), int(h)
    except Exception:
        return np.nan, np.nan

res = df['resolution'].astype(str).apply(parse_res)
df['res_w'] = res.apply(lambda x: x[0])
df['res_h'] = res.apply(lambda x: x[1])

# compute total pixels and approx ppi
df['pixels'] = df['res_w'] * df['res_h']
# ppi ≈ sqrt(w^2 + h^2) / diagonal_inches
with np.errstate(divide='ignore', invalid='ignore'):
    df['ppi_est'] = np.sqrt(df['res_w'].fillna(0)**2 + df['res_h'].fillna(0)**2) / df['display_size_in']

# Convert bluetooth to numeric if it's like 5.1 etc.
try:
    df['bluetooth'] = pd.to_numeric(df['bluetooth'], errors='coerce')
except Exception:
    pass

# Choose features: numeric ones + selected categoricals (brand, cpu_brand, gpu_brand, os, storage_type, display_type)
numeric_cols = [
    'release_year','cpu_tier','cpu_cores','cpu_threads','cpu_base_ghz','cpu_boost_ghz',
    'vram_gb','ram_gb','storage_gb','storage_drive_count','display_size_in','refresh_hz',
    'battery_wh','charger_watts','psu_watts','weight_kg','warranty_months','bluetooth',
    'pixels','ppi_est'
]

# Some numeric columns may be strings; coerce
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Categorical features to one-hot if cardinality small, otherwise frequency-encode
cat_candidates = ['brand','cpu_brand','gpu_brand','os','form_factor','storage_type','display_type','wifi']
cat_small = []
cat_freq = []
for c in cat_candidates:
    if c not in df.columns:
        continue
    nuni = df[c].nunique(dropna=True)
    if nuni <= 20:
        cat_small.append(c)
    else:
        cat_freq.append(c)

print('One-hot encode small categorical:', cat_small)
print('Frequency-encode large categorical:', cat_freq)

# One-hot encode small
if cat_small:
    # sklearn >=1.2 uses `sparse_output` instead of the older `sparse` kwarg
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_arr = ohe.fit_transform(df[cat_small].fillna(''))
    ohe_cols = list(ohe.get_feature_names_out(cat_small))
    df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)
else:
    df_ohe = pd.DataFrame(index=df.index)

# Frequency encode large
df_freq = pd.DataFrame(index=df.index)
for c in cat_freq:
    freq = df[c].fillna('')
    vc = freq.value_counts(dropna=False)
    df_freq[c + '_freq'] = freq.map(vc).fillna(0).astype(float)

# Prepare final DataFrame
X = pd.concat([df[numeric_cols], df_ohe, df_freq], axis=1)
# Drop columns all-NaN
X = X.loc[:, X.columns.notna()]
X = X.dropna(axis=1, how='all')

# Target
y = pd.to_numeric(df['price'], errors='coerce')
# Drop rows with missing target or a high fraction of missing features
mask = ~y.isna()
X = X.loc[mask]
y = y.loc[mask]

# Fill remaining missing numeric values with median
for c in X.columns:
    if X[c].dtype.kind in 'biufc':
        med = X[c].median()
        X[c] = X[c].fillna(med)
    else:
        X[c] = X[c].fillna('')

print('Final feature matrix shape:', X.shape)

# Correlation matrix for numeric features (sample of them to keep plot readable)
num_for_corr = X.select_dtypes(include=[np.number]).iloc[:, :40]
corr = num_for_corr.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation (first 40 numeric features)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'corr_matrix.png')
plt.close()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train/test shapes:', X_train.shape, X_test.shape)



# Train all regressors
try:
    n_est = int(os.environ.get('RF_N_ESTIMATORS', '100'))
except Exception:
    n_est = 100
rf = RandomForestRegressor(n_estimators=n_est, n_jobs=-1, random_state=42)
print('Training RandomForest...')
rf.fit(X_train, y_train)
print('Training complete')

lr = LinearRegression()
print('Training LinearRegression...')
lr.fit(X_train, y_train)
print('Training complete')

ridge = RidgeRegression(alpha=1.0)
print('Training RidgeRegression...')
ridge.fit(X_train, y_train)
print('Training complete')

knn = KNNRegressor(k=5)
print('Training KNNRegressor...')
knn.fit(X_train, y_train)
print('Training complete')

# Evaluate all
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_knn = knn.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print("\nModel Comparison:")
print(f"RandomForest     MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}")
print(f"LinearRegression MSE: {mse_lr:.4f}, R2: {r2_lr:.4f}")
print(f"RidgeRegression  MSE: {mse_ridge:.4f}, R2: {r2_ridge:.4f}")
print(f"KNNRegressor     MSE: {mse_knn:.4f}, R2: {r2_knn:.4f}")


# Feature importances (RF only)
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print('\nTop 20 feature importances (RandomForest):')
print(feat_imp.head(20))

# Plot top features
plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp.head(20).values, y=feat_imp.head(20).index)
plt.title('Top 20 Feature Importances (RandomForest)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'top_features.png')
plt.close()


# Save models and feature list
joblib.dump({'model': rf, 'features': list(X.columns)}, OUT_DIR / 'rf_model.joblib')
joblib.dump({'model': lr, 'features': list(X.columns)}, OUT_DIR / 'lr_model.joblib')
joblib.dump({'model': ridge, 'features': list(X.columns)}, OUT_DIR / 'ridge_model.joblib')
joblib.dump({'model': knn, 'features': list(X.columns)}, OUT_DIR / 'knn_model.joblib')
print('Artifacts saved in', OUT_DIR)


# Save metrics
with open(OUT_DIR / 'metrics.txt', 'w') as fh:
    fh.write(f'RandomForest MSE: {mse_rf}\n')
    fh.write(f'RandomForest R2: {r2_rf}\n')
    fh.write(f'LinearRegression MSE: {mse_lr}\n')
    fh.write(f'LinearRegression R2: {r2_lr}\n')
    fh.write(f'RidgeRegression MSE: {mse_ridge}\n')
    fh.write(f'RidgeRegression R2: {r2_ridge}\n')
    fh.write(f'KNNRegressor MSE: {mse_knn}\n')
    fh.write(f'KNNRegressor R2: {r2_knn}\n')

print('Done')
