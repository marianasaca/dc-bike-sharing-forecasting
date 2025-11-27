
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(page_title="DC Bike Sharing (2011–2012) — Interactive Report & Predictor", layout="wide")

# -------------------------
# Data loading & utilities
# -------------------------

from pathlib import Path

@st.cache_data
def load_default():
    # Look for the CSV in common local spots so you don't have to upload it.
    # Priority: script folder -> current working directory -> parent folder
    candidates = [
        Path(__file__).parent / "bike_hourly_with_features.csv",
        Path.cwd() / "bike_hourly_with_features.csv",
        Path.cwd().parent / "bike_hourly_with_features.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if 'dteday' in df.columns:
                df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')
            return df
    # If not found, stop with a clear message
    st.error("Could not find `bike_hourly_with_features.csv`. Place it next to this script or in the project root, or use the uploader.")
    st.stop()

def load_any(upload):
    df = pd.read_csv(upload)
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')
    return df

# ---------- Figure saving helpers ----------
def _figures_dir():
    from pathlib import Path
    out = Path(__file__).parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out

def save_fig(fig, filename_stem: str):
    """Save Plotly figure as PNG (if kaleido installed) and HTML. Returns file paths."""
    out = _figures_dir()
    png_path = out / f"{filename_stem}.png"
    html_path = out / f"{filename_stem}.html"
    saved = []
    # Try PNG first
    try:
        fig.write_image(str(png_path))  # requires kaleido
        saved.append(str(png_path))
    except Exception:
        pass
    # Always save HTML fallback
    try:
        fig.write_html(str(html_path), include_plotlyjs='cdn', full_html=True)
        saved.append(str(html_path))
    except Exception:
        pass
    return saved
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    # Date-derived
    if 'dteday' in d.columns:
        d['year'] = d['dteday'].dt.year
        d['month'] = d['dteday'].dt.month
        d['day'] = d['dteday'].dt.day
        d['dayofweek'] = d['dteday'].dt.dayofweek
        d['dayofyear'] = d['dteday'].dt.dayofyear
        d['is_weekend'] = d['dayofweek'].isin([5,6]).astype(int)
    else:
        if 'mnth' in d.columns: d['month'] = d['mnth']
        if 'weekday' in d.columns: d['dayofweek'] = d['weekday']
        d['is_weekend'] = d.get('dayofweek', pd.Series(np.nan, index=d.index)).isin([5,6]).astype('float').fillna(0).astype(int)

    # Hourly encodings
    if 'hr' in d.columns:
        d['hour'] = d['hr']
    elif 'hour' not in d.columns:
        # Create a dummy hour to avoid downstream errors (if not present)
        d['hour'] = 0
    d['hour_sin'] = np.sin(2*np.pi*d['hour']/24)
    d['hour_cos'] = np.cos(2*np.pi*d['hour']/24)
    d['rush_hour'] = d['hour'].isin([7,8,9,16,17,18]).astype(int)

    # Weather deltas
    if set(['temp','atemp']).issubset(d.columns):
        d['feels_delta'] = d['atemp'] - d['temp']

    # Daylight proxy
    if 'month' in d.columns:
        daylight_map = {1:9.7,2:10.9,3:12.0,4:13.2,5:14.3,6:14.9,7:14.6,8:13.6,9:12.4,10:11.2,11:10.2,12:9.4}
        d['daylight_hours'] = d['month'].map(daylight_map)

    # Target & lags
    target = 'cnt' if 'cnt' in d.columns else ('count' if 'count' in d.columns else None)
    if target is not None:
        # Sort for lag correctness
        sort_cols = []
        if 'dteday' in d.columns: sort_cols.append('dteday')
        if 'hr' in d.columns: sort_cols.append('hr')
        if sort_cols:
            d = d.sort_values(sort_cols)
        for lag in [1, 2, 24]:
            d[f'cnt_lag_{lag}'] = d[target].shift(lag)

    return d

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(transformers=[
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler(with_mean=False))]), num_cols)
    ] + ([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)] if cat_cols else []))
    return pre

def pick_target(df):
    if 'cnt' in df.columns: return 'cnt'
    if 'count' in df.columns: return 'count'
    return None

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload another hourly dataset (optional, CSV)", type=['csv'])
df = load_any(uploaded) if uploaded is not None else load_default()

target = pick_target(df)
if target is None:
    st.error("Target column not found. Expecting 'cnt' or 'count'.")
    st.stop()

# Date filtering
if 'dteday' in df.columns:
    mind, maxd = pd.to_datetime(df['dteday']).min(), pd.to_datetime(df['dteday']).max()
    date_range = st.sidebar.date_input("Date range", value=(mind, maxd), min_value=mind, max_value=maxd)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        df = df[(df['dteday'] >= pd.to_datetime(date_range[0])) & (df['dteday'] <= pd.to_datetime(date_range[1]))]

st.sidebar.header("Features & Model")
save_charts = st.sidebar.checkbox("Save charts to ./figures", value=True)
include_lags = st.sidebar.checkbox("Include lag features (1h, 2h, 24h)", value=True)
model_name = st.sidebar.selectbox("Model", ["RandomForest", "GradientBoosting", "Ridge", "LinearRegression"], index=0)
test_perc = st.sidebar.slider("Test size (% of last data)", 10, 40, 20, 5)

# Model hyperparams in sidebar
if model_name == "RandomForest":
    rf_n = st.sidebar.slider("RF n_estimators", 100, 800, 300, 50)
    rf_depth = st.sidebar.select_slider("RF max_depth", options=[None, 8, 12, 16, 20], value=None)
    rf_leaf = st.sidebar.slider("RF min_samples_leaf", 1, 10, 2, 1)
elif model_name == "GradientBoosting":
    g_n = st.sidebar.slider("GB n_estimators", 100, 800, 300, 50)
    g_lr = st.sidebar.slider("GB learning_rate", 0.01, 0.3, 0.1, 0.01)
    g_depth = st.sidebar.slider("GB max_depth", 2, 6, 3, 1)
elif model_name == "Ridge":
    r_alpha = st.sidebar.slider("Ridge alpha", 0.1, 10.0, 1.0, 0.1)

# -------------------------
# Header
# -------------------------
st.title("Washington D.C. Bike Sharing (2011–2012)")
st.caption("Interactive data quality checks, exploratory analysis, and an hourly demand predictor.")

# -------------------------
# Feature engineering
# -------------------------
df_fe = add_features(df)
if not include_lags:
    drop_lag = [c for c in df_fe.columns if str(c).startswith('cnt_lag_')]
    if drop_lag:
        df_fe = df_fe.drop(columns=drop_lag)

# -------------------------
# Tabs: Data/EDA/Model
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Data Quality", "Exploratory Analysis", "Modeling", "Results & Insights"])

with tab1:
    st.subheader("Schema & Missing Values")
    st.write(f"Shape: {df_fe.shape}")
    st.dataframe(df_fe.head(20), use_container_width=True)
    st.write("Missing values per column:")
    miss = df_fe.isna().sum().reset_index().rename(columns={"index":"column", 0:"missing"})
    miss.columns = ["column","missing"]
    st.dataframe(miss, use_container_width=True)

    st.subheader("Outlier counts (IQR method)")
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    iqr_out = {}
    for c in num_cols:
        q1, q3 = df_fe[c].quantile([0.25, 0.75])
        iqr = q3-q1
        low, up = q1 - 1.5*iqr, q3 + 1.5*iqr
        iqr_out[c] = int(((df_fe[c] < low) | (df_fe[c] > up)).sum())
    st.dataframe(pd.DataFrame({"feature": list(iqr_out.keys()), "outliers": list(iqr_out.values())}).sort_values("outliers", ascending=False), use_container_width=True)

with tab2:
    st.subheader("Time Series")
    if 'dteday' in df_fe.columns:
        fig = px.line(df_fe, x="dteday", y=target, title="Total rides over time")
        if save_charts: save_fig(fig, "time_series_total_rides")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'dteday' column detected to show time line.")

    c1, c2 = st.columns(2)
    with c1:
        if 'hour' in df_fe.columns:
            fig = px.box(df_fe, x="hour", y=target, title="Distribution by hour")
            if save_charts: save_fig(fig, "box_by_hour")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if 'season' in df_fe.columns:
            fig = px.box(df_fe, x="season", y=target, title="Distribution by season")
            if save_charts: save_fig(fig, "box_by_season")
            st.plotly_chart(fig, use_container_width=True)

    # Heatmap hour vs weekday
    if set(['hour','dayofweek']).issubset(df_fe.columns):
        st.subheader("Heatmap: Average rides by weekday × hour")
        heat = df_fe.groupby(['dayofweek','hour'])[target].mean().reset_index()
        heat['weekday_name'] = heat['dayofweek'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
        piv = heat.pivot(index='weekday_name', columns='hour', values=target)
        fig = px.imshow(piv, aspect='auto', title="Avg rides (weekday × hour)")
        if save_charts: save_fig(fig, "heatmap_weekday_hour")
        st.plotly_chart(fig, use_container_width=True)

    # Weather scatter
    st.subheader("Weather relationships")
    for c in ['temp','atemp','hum','windspeed']:
        if c in df_fe.columns:
            fig = px.scatter(df_fe, x=c, y=target, trendline='ols', title=f'{target} vs {c}')
            if save_charts: save_fig(fig, f"scatter_{c}")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Train/Test split (time-ordered)")
    df_model = df_fe.dropna().copy()
    drop_cols = set([target, 'casual','registered']) & set(df_model.columns)
    X = df_model.drop(columns=list(drop_cols)) if drop_cols else df_model.copy()
    if 'instant' in X.columns: X = X.drop(columns=['instant'])
    y = df_model[target]

    n = len(X)
    split_idx = int(n*(1 - test_perc/100.0))
    if split_idx < 1 or split_idx >= n:
        st.error("Invalid split. Adjust test size.")
        st.stop()

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pre = build_preprocessor(X)

    if model_name == "RandomForest":
        mdl = RandomForestRegressor(n_estimators=rf_n, max_depth=rf_depth, min_samples_leaf=rf_leaf, random_state=42, n_jobs=-1)
    elif model_name == "GradientBoosting":
        mdl = GradientBoostingRegressor(random_state=42, n_estimators=g_n, learning_rate=g_lr, max_depth=g_depth)
    elif model_name == "Ridge":
        mdl = Ridge(alpha=r_alpha)
    else:
        mdl = LinearRegression()

    pipe = Pipeline([('pre', pre), ('mdl', mdl)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    from math import sqrt
    rmse = sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("RMSE", f"{rmse:.2f}")
    m3.metric("R²", f"{r2:.3f}")

    st.subheader("Predictions vs Actual")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test.values, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=pred, mode='lines', name=f'Predicted ({model_name})'))
    fig.update_layout(title='Predictions vs Actual (Test period)', xaxis_title='Time index (test)', yaxis_title='Count')
    if save_charts: save_fig(fig, "pred_vs_actual")
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    st.subheader("Residual analysis")
    resid = y_test.values - pred
    fig = px.histogram(resid, nbins=40, title="Residual distribution")
    if save_charts: save_fig(fig, "residual_hist")
    st.plotly_chart(fig, use_container_width=True)
    # Residuals by hour (if available)
    if 'hour' in X_test.columns:
        tmp = pd.DataFrame({'hour': X_test['hour'], 'resid': resid})
        fig = px.box(tmp, x='hour', y='resid', title='Residuals by hour')
        if save_charts: save_fig(fig, "residuals_by_hour")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importances / coefficients
    st.subheader("Feature importance / coefficients")
    try:
        importances = pipe.named_steps['mdl'].feature_importances_
        preproc = pipe.named_steps['pre']
        num_feats = preproc.transformers_[0][2] if preproc.transformers_ else []
        cat_feats = []
        if len(preproc.transformers_) > 1:
            cat_feats = preproc.transformers_[1][2]
        feat_names = list(num_feats) + list(cat_feats)
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(40)
        st.dataframe(imp_df, use_container_width=True)
        st.bar_chart(imp_df.set_index('feature'))
    except Exception as e:
        try:
            coef = pipe.named_steps['mdl'].coef_
            preproc = pipe.named_steps['pre']
            num_feats = preproc.transformers_[0][2] if preproc.transformers_ else []
            cat_feats = []
            if len(preproc.transformers_) > 1:
                cat = preproc.named_transformers_['cat']
                try:
                    cat_names = list(cat.get_feature_names_out(cat_feats))
                except:
                    cat_names = cat_feats
                feat_names = list(num_feats) + cat_names
            else:
                feat_names = list(num_feats)
            coef_df = pd.DataFrame({'feature': feat_names[:len(coef)], 'coefficient': coef[:len(feat_names)]})
            coef_df = coef_df.sort_values('coefficient', key=np.abs, ascending=False).head(40)
            st.dataframe(coef_df, use_container_width=True)
            st.bar_chart(coef_df.set_index('feature'))
        except Exception as e2:
            st.info("This model does not expose importances/coefficients.")

with tab4:
    st.subheader("Key takeaways & recommendations")
st.markdown("""
- Demand shows strong **hourly seasonality** (commute peaks). Provision more bikes during **7–9am** and **4–6pm**.
- **Weekends** and **season** alter patterns; plan for leisure peaks (late morning/afternoon).
- **Weather** matters: higher temperature (to a point) increases demand; humidity/windspeed often reduce it.
- The **RandomForest/GBR** models often perform best out-of-the-box. Tune them with recent data for production.
- Include external features in production (holiday calendars, precipitation, events) and retrain periodically.
""")

#to run: 1- pip install streamlit
# 2 streamlit run Group6_StreamlitSrc_Assignment2.py     make sure you're in the right file path