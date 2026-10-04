import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# Use project-local paths so the app is portable across platforms
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'artifacts' / 'rf_model.joblib'
DATA_PATH = BASE / 'computer_prices_all.csv'

st.set_page_config(page_title='Computer Price Tester', layout='wide')
st.title('Computer Price Prediction — Interactive Tester')

@st.cache_data
def load_model(path):
    obj = joblib.load(path)
    return obj['model'], obj['features']

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def fit_encoders(df, cat_small):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(df[cat_small].fillna(''))
    return ohe

# Load artifacts
# Load primary dataframe
df_full = load_data(DATA_PATH)


def load_model_file(path):
    try:
        obj = joblib.load(path)
        return obj.get('model'), obj.get('features')
    except Exception:
        return None, None


def load_all_models(base):
    models = {}
    # expected files
    pairs = [
        ('RandomForest', base / 'rf_model.joblib'),
        ('LinearRegression', base / 'lr_model.joblib'),
        ('RidgeRegression', base / 'ridge_model.joblib'),
        ('KNN', base / 'knn_model.joblib'),
    ]
    for name, p in pairs:
        m, feats = load_model_file(p)
        if m is not None:
            models[name] = {'model': m, 'features': feats, 'path': p}
    return models


def read_metrics(path):
    metrics = {}
    try:
        text = Path(path).read_text()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0].endswith('MSE:') is False:
                # handle lines like 'RandomForest MSE: 60456.68'
                pass
        # fallback parsing using simple tokens
        for line in text.splitlines():
            toks = line.split()
            if len(toks) >= 3:
                name = toks[0]
                key = toks[1].rstrip(':')
                try:
                    val = float(toks[2])
                except Exception:
                    continue
                metrics.setdefault(name, {})[key] = val
    except Exception:
        pass
    return metrics


models = load_all_models(Path(__file__).resolve().parent / 'artifacts')
metrics = read_metrics(Path(__file__).resolve().parent / 'artifacts' / 'metrics.txt')

# convenience: determine best model by R2 (if available)
best_model_by_r2 = None
if metrics:
    best = None
    best_val = -1e9
    for mname, vals in metrics.items():
        r2 = vals.get('R2') or vals.get('R2:') or vals.get('R2', None)
        if r2 is None:
            # some files use 'R2:' as token after name; try to find keys
            for k in vals:
                if 'R2' in k:
                    r2 = vals[k]
        if r2 is not None and r2 > best_val:
            best_val = r2
            best = mname
    best_model_by_r2 = best

    model_names = sorted(list(models.keys()))

    st.sidebar.header('Mode')
mode = st.sidebar.selectbox('Choose mode', ['Single input', 'Batch upload (CSV)'])

# same categorical breakdown as pipeline
cat_candidates = ['brand','cpu_brand','gpu_brand','os','form_factor','storage_type','display_type','wifi']
cat_small = [c for c in cat_candidates if c in df_full.columns and df_full[c].nunique() <= 20]
cat_freq = [c for c in cat_candidates if c in df_full.columns and c not in cat_small]

# fit encoder on training data so app uses same encoding rules
if cat_small:
    ohe = fit_encoders(df_full, cat_small)
else:
    ohe = None

# helper preprocessing (mirrors pipeline)
def parse_res(res):
    try:
        w,h = str(res).split('x')
        return int(w), int(h)
    except Exception:
        return np.nan, np.nan

def preprocess_raw(df_raw):
    df = df_raw.copy()
    # resolution
    res = df['resolution'].astype(str).apply(parse_res) if 'resolution' in df.columns else pd.Series([(np.nan, np.nan)] * len(df))
    df['res_w'] = res.apply(lambda x: x[0])
    df['res_h'] = res.apply(lambda x: x[1])
    df['pixels'] = df['res_w'] * df['res_h']
    with np.errstate(divide='ignore', invalid='ignore'):
        df['ppi_est'] = np.sqrt(df['res_w'].fillna(0)**2 + df['res_h'].fillna(0)**2) / df['display_size_in']
    # numeric coercion
    numeric_cols = [
        'release_year','cpu_tier','cpu_cores','cpu_threads','cpu_base_ghz','cpu_boost_ghz',
        'vram_gb','ram_gb','storage_gb','storage_drive_count','display_size_in','refresh_hz',
        'battery_wh','charger_watts','psu_watts','weight_kg','warranty_months','bluetooth',
        'pixels','ppi_est'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    # one-hot small cats
    df_ohe = pd.DataFrame(index=df.index)
    if ohe is not None and cat_small:
        # Ensure all expected categorical columns exist in the incoming dataframe.
        # If a column is missing, create it with empty string values so encoder can transform.
        cats = pd.DataFrame(index=df.index)
        for c in cat_small:
            if c in df.columns:
                cats[c] = df[c].fillna('')
            else:
                cats[c] = ''
        arr = ohe.transform(cats)
        cols = list(ohe.get_feature_names_out(cat_small))
        df_ohe = pd.DataFrame(arr, columns=cols, index=df.index)

    # frequency encode others
    df_freq = pd.DataFrame(index=df.index)
    for c in cat_freq:
        if c in df.columns:
            vc = df_full[c].fillna('').value_counts(dropna=False)
            df_freq[c + '_freq'] = df[c].fillna('').map(vc).fillna(0).astype(float)
        else:
            df_freq[c + '_freq'] = 0.0

    X = pd.concat([df[numeric_cols], df_ohe, df_freq], axis=1)
    # return un-aligned feature matrix; alignment to a model's feature list
    # happens at prediction time so the app can compare models with different
    # feature sets.
    return X


def align_and_fill(X_raw, model_features):
    """Align X_raw to model_features, adding missing columns with 0 and
    filling numeric NAs with medians computed from the original training
    dataframe `df_full` when available."""
    X = X_raw.copy()
    # ensure all model_features present
    for col in model_features:
        if col not in X.columns:
            X[col] = 0.0
    # keep only model features in order
    X = X.loc[:, model_features]
    # fill numeric NA with medians from training data if available
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            if c in df_full.columns:
                med = df_full[c].median()
            else:
                med = X[c].median()
            if not np.isfinite(med):
                med = 0.0
            X[c] = X[c].fillna(med)
        else:
            X[c] = X[c].fillna(0)
    return X

if mode == 'Single input':
    st.header('Single input test')
    # Model selector (or compare all)
    model_names = sorted(list(models.keys()))
    if not model_names:
        st.warning('No trained models found in `artifacts/`. Run the pipeline first to produce model files.')
    model_choice = st.selectbox('Choose model (or Compare all)', options=model_names + ['Compare all'])
    col1, col2 = st.columns(2)
    with col1:
        ram_gb = st.number_input('RAM (GB)', value=int(df_full['ram_gb'].median()))
        storage_gb = st.number_input('Storage (GB)', value=int(df_full['storage_gb'].median()))
        cpu_cores = st.number_input('CPU cores', value=int(df_full['cpu_cores'].median()))
        cpu_threads = st.number_input('CPU threads', value=int(df_full['cpu_threads'].median()))
        display_size_in = st.number_input('Display size (in)', value=float(df_full['display_size_in'].median()))
        resolution = st.text_input('Resolution (e.g. 1920x1080)', value='1920x1080')
        brand = st.selectbox('Brand', options=[''] + list(df_full['brand'].unique())) if 'brand' in df_full.columns else ''
        os_val = st.selectbox('OS', options=[''] + list(df_full['os'].unique())) if 'os' in df_full.columns else ''
    with col2:
        vram_gb = st.number_input('VRAM (GB)', value=int(df_full['vram_gb'].median()))
        storage_drive_count = st.number_input('Storage drive count', value=int(df_full['storage_drive_count'].median()))
        refresh_hz = st.number_input('Refresh Hz', value=int(df_full['refresh_hz'].median()))
        weight_kg = st.number_input('Weight (kg)', value=float(df_full['weight_kg'].median()))
        battery_wh = st.number_input('Battery (Wh)', value=float(df_full['battery_wh'].median()))
        display_type = st.selectbox('Display type', options=[''] + list(df_full['display_type'].unique())) if 'display_type' in df_full.columns else ''
        cpu_brand = st.selectbox('CPU brand', options=[''] + list(df_full['cpu_brand'].unique())) if 'cpu_brand' in df_full.columns else ''

    # build a single-row raw dict with columns used by preprocess
    raw = {
        'ram_gb': ram_gb,
        'storage_gb': storage_gb,
        'cpu_cores': cpu_cores,
        'cpu_threads': cpu_threads,
        'display_size_in': display_size_in,
        'resolution': resolution,
        'brand': brand,
        'os': os_val,
        'vram_gb': vram_gb,
        'storage_drive_count': storage_drive_count,
        'refresh_hz': refresh_hz,
        'weight_kg': weight_kg,
        'battery_wh': battery_wh,
        'display_type': display_type,
        'cpu_brand': cpu_brand
    }
    raw_df = pd.DataFrame([raw])
    X_raw = preprocess_raw(raw_df)

    if model_choice == 'Compare all':
        preds = {}
        for name, info in models.items():
            feats = info.get('features') or []
            if feats is None:
                feats = list(X_raw.columns)
            X_al = align_and_fill(X_raw, feats)
            try:
                p = info['model'].predict(X_al)[0]
            except Exception as e:
                p = None
            preds[name] = p
        df_preds = pd.DataFrame.from_dict(preds, orient='index', columns=['predicted_price']).reset_index().rename(columns={'index':'model'})
        st.subheader('Model predictions (single input)')
        st.table(df_preds)
        if best_model_by_r2:
            st.success(f'Best model by recorded R²: {best_model_by_r2}')
    else:
        # single model prediction
        info = models.get(model_choice)
        if info is None:
            st.error(f'Model {model_choice} not available')
        else:
            feats = info.get('features') or []
            X_al = align_and_fill(X_raw, feats)
            try:
                p = info['model'].predict(X_al)[0]
            except Exception as e:
                st.error(f'Prediction failed: {e}')
                p = None
            if p is not None:
                st.subheader('Predicted price')
                st.metric('Predicted price (USD)', f"${p:,.2f}")
                st.caption(f'Prediction from {model_choice}')

                # show top features / coefficients if available
                m = info['model']
                if hasattr(m, 'feature_importances_'):
                    importances = np.asarray(m.feature_importances_)
                    idx = feats
                    if len(importances) == len(idx):
                        feat_imp = pd.Series(importances, index=idx).sort_values(ascending=False)
                        st.subheader('Top features (model)')
                        st.table(feat_imp.head(10).reset_index().rename(columns={'index':'feature', 0:'importance'}))
                elif hasattr(m, 'coef_'):
                    coefs = np.asarray(m.coef_).ravel()
                    idx = feats
                    if len(coefs) == len(idx):
                        feat_imp = pd.Series(coefs, index=idx).abs().sort_values(ascending=False)
                        st.subheader('Top coefficients (by absolute value)')
                        st.table(feat_imp.head(10).reset_index().rename(columns={'index':'feature', 0:'abs_coef'}))

else:
    st.header('Batch upload')
    # let user pick model for batch or compare all
    batch_choice = st.selectbox('Choose model for batch predictions', options=model_names + ['Compare all']) if model_names else st.text('No models available')
    uploaded = st.file_uploader('Upload CSV with raw columns (like the original dataset)', type=['csv'])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.write('Rows uploaded:', len(df_raw))
        X_raw = preprocess_raw(df_raw)
        df_out = df_raw.copy()
        if batch_choice == 'Compare all':
            for name, info in models.items():
                feats = info.get('features') or list(X_raw.columns)
                X_al = align_and_fill(X_raw, feats)
                try:
                    preds = info['model'].predict(X_al)
                except Exception as e:
                    preds = [None] * len(X_al)
                df_out[f'pred_{name}'] = preds
        else:
            info = models.get(batch_choice)
            if info is None:
                st.error(f'Model {batch_choice} not available')
            else:
                feats = info.get('features') or list(X_raw.columns)
                X_al = align_and_fill(X_raw, feats)
                try:
                    preds = info['model'].predict(X_al)
                except Exception as e:
                    st.error(f'Prediction failed: {e}')
                    preds = [None] * len(X_al)
                df_out['predicted_price'] = preds

        st.download_button('Download predictions as CSV', df_out.to_csv(index=False).encode('utf-8'), file_name='predictions.csv')
        st.dataframe(df_out.head(50))

st.sidebar.markdown('Model artifacts loaded from:')
for name, info in models.items():
    st.sidebar.write(f"- {name}: {info.get('path').name}")
if best_model_by_r2:
    st.sidebar.success(f'Best model by recorded R²: {best_model_by_r2}')
