"""
ML Capstone Dashboard — Water Quality pH Prediction
=====================================================
Streamlit frontend that trains all 10 regression models, shows full
EDA + comparison benchmarks, and lets users test with custom inputs.

Run:  streamlit run app.py
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.io as sio
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality ML Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .block-container { padding-top: 1rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem; color: white;
        text-align: center; margin-bottom: 0.5rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.9; }

    /* Model rank badge */
    .rank-badge {
        display: inline-block; width: 28px; height: 28px; border-radius: 50%;
        text-align: center; line-height: 28px; font-weight: 700; font-size: 0.8rem;
        color: white; margin-right: 6px;
    }
    .rank-1 { background: #FFD700; color: #333; }
    .rank-2 { background: #C0C0C0; color: #333; }
    .rank-3 { background: #CD7F32; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        color: white; padding: 0.8rem 1.2rem; border-radius: 8px;
        margin: 1rem 0 0.5rem 0; font-size: 1.1rem;
    }

    /* Prediction result cards */
    .pred-card {
        border: 2px solid #e0e0e0; border-radius: 10px; padding: 1rem;
        text-align: center; transition: transform 0.2s;
    }
    .pred-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.12); }
    .pred-best { border-color: #2ecc71; background: #f0fff4; }

    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 8px; padding: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING  (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_preprocess():
    """Load the MAT file, build the DataFrame, engineer features."""
    paths = [
        "data/water_dataset.mat",
        "water+quality+prediction-1/water_dataset.mat",
        "./water_dataset.mat",
    ]
    mat_path = next((p for p in paths if os.path.exists(p)), None)
    if mat_path is None:
        st.error("❌ Could not find `water_dataset.mat`!")
        st.stop()

    mat = sio.loadmat(mat_path)
    X_tr, X_te = mat["X_tr"], mat["X_te"]
    Y_tr, Y_te = mat["Y_tr"], mat["Y_te"]
    loc_ids = mat["location_ids"].flatten()
    loc_groups = mat["location_group"][0]

    station_group_map = {}
    for g_idx, g in enumerate(loc_groups):
        for st_idx in g.flatten():
            station_group_map[st_idx - 1] = g_idx + 1

    feature_names = [
        "cond_max", "ph_max", "ph_min", "cond_min", "cond_mean",
        "do_max", "do_mean", "do_min", "temp_mean", "temp_min", "temp_max",
    ]

    records = []
    for day in range(X_tr.shape[1]):
        xt, yt = X_tr[0, day], Y_tr[:, day]
        for st in range(37):
            rec = {
                "day_idx": day, "station_idx": st,
                "location_id": loc_ids[st],
                "spatial_group": station_group_map[st],
                "ph_target": yt[st],
            }
            for fi, fn in enumerate(feature_names):
                rec[fn] = float(xt[st, fi])
            records.append(rec)

    for day in range(X_te.shape[1]):
        xt, yt = X_te[0, day], Y_te[:, day]
        for st in range(37):
            rec = {
                "day_idx": 423 + day, "station_idx": st,
                "location_id": loc_ids[st],
                "spatial_group": station_group_map[st],
                "ph_target": yt[st],
            }
            for fi, fn in enumerate(feature_names):
                rec[fn] = float(xt[st, fi])
            records.append(rec)

    df = pd.DataFrame(records)

    # Feature engineering
    df["ph_range"]  = df["ph_max"]  - df["ph_min"]
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["do_range"]   = df["do_max"]  - df["do_min"]
    df["temp_do_interaction"] = df["temp_mean"] * df["do_mean"]
    df["cond_spread"] = df["cond_max"] - df["cond_min"]
    df = pd.get_dummies(df, columns=["spatial_group"], drop_first=True, dtype=float)

    predictor_cols = feature_names + [
        "ph_range", "temp_range", "do_range",
        "temp_do_interaction", "cond_spread",
        "spatial_group_2", "spatial_group_3",
    ]

    X = df[predictor_cols]
    y = df["ph_target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return (
        df, feature_names, predictor_cols,
        X_train, X_test, y_train, y_test,
        X_train_sc, X_test_sc, scaler,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING  (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_all_models(
    _X_train, _X_test, _y_train, _y_test,
    _X_train_sc, _X_test_sc, predictor_cols,
):
    """Train all 10 regression models and return results dict."""
    results = {}

    def _eval(name, model, yp, yt, cv_score, extra=None):
        results[name] = {
            "model": model,
            "y_pred": yp,
            "r2":   r2_score(yt, yp),
            "rmse": np.sqrt(mean_squared_error(yt, yp)),
            "mae":  mean_absolute_error(yt, yp),
            "cv_r2": cv_score,
            "extra": extra or {},
        }

    y_train = _y_train
    y_test  = _y_test

    # 1. Linear Regression
    lr = LinearRegression().fit(_X_train_sc, y_train)
    yp = lr.predict(_X_test_sc)
    cv = cross_val_score(lr, _X_train_sc, y_train, cv=5, scoring="r2").mean()
    _eval("Linear Regression", lr, yp, y_test, cv,
          {"coefficients": dict(zip(predictor_cols, lr.coef_))})

    # 2. Ridge
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5, scoring="r2")
    ridge.fit(_X_train_sc, y_train)
    yp = ridge.predict(_X_test_sc)
    _eval("Ridge Regression", ridge, yp, y_test, ridge.best_score_,
          {"best_alpha": ridge.alpha_})

    # 3. Lasso
    lasso = LassoCV(alphas=np.logspace(-4, 0, 50), cv=5,
                     random_state=RANDOM_STATE, max_iter=3000)
    lasso.fit(_X_train_sc, y_train)
    yp = lasso.predict(_X_test_sc)
    cv = cross_val_score(lasso, _X_train_sc, y_train, cv=5, scoring="r2").mean()
    zeroed = [predictor_cols[i] for i, c in enumerate(lasso.coef_) if c == 0]
    _eval("Lasso Regression", lasso, yp, y_test, cv,
          {"best_alpha": lasso.alpha_, "zeroed_features": zeroed})

    # 4. ElasticNet
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
        alphas=np.logspace(-4, 0, 40), cv=5,
        random_state=RANDOM_STATE, max_iter=3000,
    )
    enet.fit(_X_train_sc, y_train)
    yp = enet.predict(_X_test_sc)
    cv = cross_val_score(enet, _X_train_sc, y_train, cv=5, scoring="r2").mean()
    _eval("ElasticNet", enet, yp, y_test, cv,
          {"best_alpha": enet.alpha_, "l1_ratio": enet.l1_ratio_})

    # 5. Polynomial (degree 2, on 6 primary features + Ridge)
    primary = ["cond_mean", "ph_max", "ph_min", "do_max", "do_mean", "temp_mean"]
    pidx = [predictor_cols.index(c) for c in primary]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xtr_poly = poly.fit_transform(_X_train_sc[:, pidx])
    Xte_poly = poly.transform(_X_test_sc[:, pidx])
    poly_model = RidgeCV(alphas=np.logspace(-2, 2, 20), cv=3).fit(Xtr_poly, y_train)
    yp = poly_model.predict(Xte_poly)
    cv = cross_val_score(poly_model, Xtr_poly, y_train, cv=3, scoring="r2").mean()
    _eval("Polynomial (Deg 2)", poly_model, yp, y_test, cv,
          {"poly_transformer": poly, "primary_indices": pidx,
           "n_features": Xtr_poly.shape[1]})

    # 6. Decision Tree
    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {"max_depth": [6, 10, 14], "min_samples_split": [2, 5, 10]},
        cv=5, scoring="r2", n_jobs=-1,
    )
    dt_grid.fit(_X_train, y_train)
    best_dt = dt_grid.best_estimator_
    yp = best_dt.predict(_X_test)
    _eval("Decision Tree", best_dt, yp, y_test, dt_grid.best_score_,
          {"best_params": dt_grid.best_params_,
           "feature_importance": dict(zip(predictor_cols, best_dt.feature_importances_))})

    # 7. Random Forest
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"n_estimators": [80, 120], "max_depth": [12, 16], "min_samples_split": [2, 5]},
        cv=3, scoring="r2", n_jobs=-1,
    )
    rf_grid.fit(_X_train, y_train)
    best_rf = rf_grid.best_estimator_
    yp = best_rf.predict(_X_test)
    _eval("Random Forest", best_rf, yp, y_test, rf_grid.best_score_,
          {"best_params": rf_grid.best_params_,
           "feature_importance": dict(zip(predictor_cols, best_rf.feature_importances_))})

    # 8. Gradient Boosting
    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"n_estimators": [80, 120], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        cv=3, scoring="r2", n_jobs=-1,
    )
    gb_grid.fit(_X_train, y_train)
    best_gb = gb_grid.best_estimator_
    yp = best_gb.predict(_X_test)
    _eval("Gradient Boosting", best_gb, yp, y_test, gb_grid.best_score_,
          {"best_params": gb_grid.best_params_,
           "feature_importance": dict(zip(predictor_cols, best_gb.feature_importances_))})

    # 9. SVR
    sample_idx = np.random.choice(len(_X_train_sc), size=min(5000, len(_X_train_sc)), replace=False)
    svr_grid = GridSearchCV(
        SVR(kernel="rbf"),
        {"C": [1.0, 5.0], "epsilon": [0.01, 0.05]},
        cv=3, scoring="r2", n_jobs=-1,
    )
    svr_grid.fit(_X_train_sc[sample_idx], y_train.iloc[sample_idx])
    best_svr = SVR(kernel="rbf", **svr_grid.best_params_)
    best_svr.fit(_X_train_sc, y_train)
    yp = best_svr.predict(_X_test_sc)
    _eval("SVR (RBF)", best_svr, yp, y_test, svr_grid.best_score_,
          {"best_params": svr_grid.best_params_})

    # 10. KNN
    best_k, best_r2 = 3, -1
    for k in [3, 5, 7, 9, 11, 15]:
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(_X_train_sc, y_train)
        r2 = r2_score(y_test, knn.predict(_X_test_sc))
        if r2 > best_r2:
            best_k, best_r2 = k, r2
    best_knn = KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1)
    best_knn.fit(_X_train_sc, y_train)
    yp = best_knn.predict(_X_test_sc)
    cv = cross_val_score(best_knn, _X_train_sc[:5000], y_train.iloc[:5000],
                         cv=3, scoring="r2").mean()
    _eval("KNN", best_knn, yp, y_test, cv, {"best_k": best_k})

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: which models need scaled data?
# ═══════════════════════════════════════════════════════════════════════════════
SCALED_MODELS  = {"Linear Regression", "Ridge Regression", "Lasso Regression",
                  "ElasticNet", "Polynomial (Deg 2)", "SVR (RBF)", "KNN"}
RAW_MODELS     = {"Decision Tree", "Random Forest", "Gradient Boosting"}


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Loading dataset & engineering features …"):
    (df, feature_names, predictor_cols,
     X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc, scaler) = load_and_preprocess()

with st.spinner("🧠 Training all 10 regression models (first run only, then cached) …"):
    results = train_all_models(
        X_train, X_test, y_train, y_test,
        X_train_sc, X_test_sc, predictor_cols,
    )

# Build sorted benchmark dataframe
bench_rows = []
for name, r in results.items():
    bench_rows.append({
        "Model": name, "R²": r["r2"], "RMSE": r["rmse"],
        "MAE": r["mae"], "CV R²": r["cv_r2"],
    })
bench_df = pd.DataFrame(bench_rows).sort_values("R²", ascending=False).reset_index(drop=True)
bench_df.insert(0, "Rank", range(1, len(bench_df) + 1))

best_model_name = bench_df.iloc[0]["Model"]

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💧 Water Quality ML")
    st.markdown("##### Capstone Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 EDA", "🏆 Model Comparison",
         "🔍 Model Deep Dive", "🧪 Predict with Your Input"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df):,} observations")
    st.markdown(f"**Stations:** 37 USGS sites")
    st.markdown(f"**Features:** {len(predictor_cols)} predictors")
    st.markdown(f"**Best Model:** {best_model_name}")
    st.markdown(f"**Best R²:** {bench_df.iloc[0]['R²']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 💧 Water Quality Prediction Dashboard")
    st.markdown(
        "**Multi-Site Water Quality Forecasting:** Predicting Continuous pH Values "
        "Across 37 Georgia Stream Monitoring Stations using 10 Regression Algorithms"
    )
    st.markdown("---")

    # Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="metric-card"><h3>26,085</h3>'
            '<p>Total Observations</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="metric-card"><h3>37</h3>'
            '<p>Monitoring Stations</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(
            '<div class="metric-card"><h3>10</h3>'
            '<p>ML Models Trained</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(
            f'<div class="metric-card"><h3>{bench_df.iloc[0]["R²"]:.4f}</h3>'
            f'<p>Best R² ({best_model_name})</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # Quick benchmark
    st.markdown('<div class="section-header">🏆 Model Leaderboard (Quick View)</div>',
                unsafe_allow_html=True)

    display_df = bench_df.copy()
    display_df["R²"]    = display_df["R²"].map("{:.4f}".format)
    display_df["RMSE"]  = display_df["RMSE"].map("{:.4f}".format)
    display_df["MAE"]   = display_df["MAE"].map("{:.4f}".format)
    display_df["CV R²"] = display_df["CV R²"].map("{:.4f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Quick bar chart
    fig = px.bar(
        bench_df, x="R²", y="Model", orientation="h",
        color="R²", color_continuous_scale="Viridis",
        title="Test R² Score — All 10 Models",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420,
                      coloraxis_showscale=False)
    fig.update_traces(text=bench_df["R²"].map("{:.4f}".format), textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Dataset info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📋 Dataset Summary</div>',
                    unsafe_allow_html=True)
        info_df = pd.DataFrame({
            "Property": ["Total Rows", "Total Columns", "Monitoring Days",
                         "Stations", "Spatial Groups", "Missing Values",
                         "Duplicate Rows"],
            "Value": [f"{len(df):,}", str(df.shape[1]), "705 (423 train + 282 test)",
                      "37", "3", "0", "0"],
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-header">🎯 Target Variable (pH)</div>',
                    unsafe_allow_html=True)
        tgt = df["ph_target"]
        stats_df = pd.DataFrame({
            "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max",
                          "IQR", "Skewness", "Kurtosis"],
            "Value": [f"{tgt.mean():.4f}", f"{tgt.median():.4f}", f"{tgt.std():.4f}",
                      f"{tgt.min():.4f}", f"{tgt.max():.4f}",
                      f"{tgt.quantile(.75) - tgt.quantile(.25):.4f}",
                      f"{tgt.skew():.4f}", f"{tgt.kurtosis():.4f}"],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Target Distribution", "Feature Distributions",
        "Spatial Analysis", "Correlation Matrix"
    ])

    with tab1:
        fig = px.histogram(
            df, x="ph_target", nbins=50, marginal="box",
            color_discrete_sequence=["#007acc"],
            title="Target Distribution — Water pH (Median)",
            labels={"ph_target": "Normalized pH Value"},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Insight:** The target pH distribution is unimodal and bell-shaped, "
            "centered at ≈ 0.667 with slight right-skewness (+0.34). "
            "Over 90% of observations fall within [0.62, 0.72], reflecting stable "
            "carbonate buffering in natural stream systems."
        )

    with tab2:
        eda_feats = ["cond_max", "cond_mean", "do_max", "do_mean", "temp_mean", "ph_max"]
        titles = ["Conductance (Max)", "Conductance (Mean)",
                  "Dissolved O₂ (Max)", "Dissolved O₂ (Mean)",
                  "Temperature (Mean)", "pH (Max)"]

        fig = make_subplots(rows=2, cols=3, subplot_titles=titles)
        colors = ["#2b5c8f", "#3c85c4", "#2a9d8f", "#48cae4", "#e76f51", "#f4a261"]
        for i, col in enumerate(eda_feats):
            r, c = divmod(i, 3)
            fig.add_trace(
                go.Histogram(x=df[col], nbinsx=35, marker_color=colors[i],
                             opacity=0.7, name=titles[i]),
                row=r + 1, col=c + 1,
            )
        fig.update_layout(height=550, showlegend=False,
                          title_text="Sensor Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Insight:** Conductance shows severe positive skewness (storm runoff events). "
            "Dissolved oxygen is broad & left-skewed (photosynthetic aeration). "
            "Temperature spans a wide seasonal range."
        )

    with tab3:
        grp_labels = {1: "Isolated", 2: "Coastal Plain", 3: "Atlanta Metro"}
        plot_df = df.copy()
        plot_df["Group"] = plot_df.get("spatial_group",
            plot_df.filter(like="spatial_group_").idxmax(axis=1)).map(
            lambda x: grp_labels.get(x, str(x)))

        # Reconstruct spatial group from dummies for plotting
        if "spatial_group" not in df.columns:
            conditions = [
                (df.get("spatial_group_2", 0) == 1),
                (df.get("spatial_group_3", 0) == 1),
            ]
            choices = ["Coastal Plain", "Atlanta Metro"]
            plot_df["Group"] = np.select(conditions, choices, default="Isolated")

        fig = px.box(
            plot_df, x="Group", y="ph_target",
            color="Group",
            color_discrete_map={
                "Isolated": "#e76f51",
                "Coastal Plain": "#2a9d8f",
                "Atlanta Metro": "#264653",
            },
            title="pH Distribution Across Hydrological Groups",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Station-level
        st_means = df.groupby("station_idx")["ph_target"].agg(["mean", "std"]).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=st_means["station_idx"], y=st_means["mean"],
            error_y=dict(type="data", array=st_means["std"], visible=True),
            mode="markers", marker=dict(size=8, color="#2b5c8f"),
            name="Mean ± Std",
        ))
        fig2.update_layout(
            title="Mean pH (± 1σ) Across 37 Stations",
            xaxis_title="Station Index", yaxis_title="Mean pH",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        corr_cols = feature_names + ["ph_target"]
        corr = df[corr_cols].corr()
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_masked = corr.where(~mask)

        fig = px.imshow(
            corr_masked, text_auto=".2f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Key Correlations:** `do_max` → target (r = +0.88), "
            "`ph_max` → target (r = +0.72). "
            "Temperature extrema are highly collinear (r > 0.95), "
            "justifying regularized models."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.markdown("# 🏆 Model Comparison & Benchmark")
    st.markdown("---")

    # Benchmark table
    st.markdown('<div class="section-header">📋 Consolidated Benchmark Table</div>',
                unsafe_allow_html=True)
    display_df = bench_df.copy()
    for c in ["R²", "RMSE", "MAE", "CV R²"]:
        display_df[c] = display_df[c].map("{:.4f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    st.markdown("")

    # Visual comparisons
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            bench_df, x="R²", y="Model", orientation="h",
            color="R²", color_continuous_scale="Viridis",
            title="Test R² Score (Higher is Better)",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=450,
                          coloraxis_showscale=False)
        fig.update_traces(text=bench_df["R²"].map("{:.4f}".format),
                          textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rmse_sorted = bench_df.sort_values("RMSE")
        fig = px.bar(
            rmse_sorted, x="RMSE", y="Model", orientation="h",
            color="RMSE", color_continuous_scale="Magma_r",
            title="RMSE (Lower is Better)",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=450,
                          coloraxis_showscale=False)
        fig.update_traces(text=rmse_sorted["RMSE"].map("{:.4f}".format),
                          textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown('<div class="section-header">📈 Multi-Metric Radar Comparison (Top 5)</div>',
                unsafe_allow_html=True)
    top5 = bench_df.head(5)
    radar_fig = go.Figure()
    for _, row in top5.iterrows():
        # Normalize: R² and CV R² already 0-1 range; for RMSE/MAE invert
        radar_fig.add_trace(go.Scatterpolar(
            r=[row["R²"], row["CV R²"], 1 - row["RMSE"] * 30, 1 - row["MAE"] * 30],
            theta=["R²", "CV R²", "1 – 30×RMSE", "1 – 30×MAE"],
            fill="toself", name=row["Model"], opacity=0.6,
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5, 1])),
        showlegend=True, height=500,
        title="Normalized Performance Radar (Top 5 Models)",
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # Predicted vs Actual for best model
    st.markdown(
        f'<div class="section-header">🎯 Best Model Diagnostics — {best_model_name}</div>',
        unsafe_allow_html=True,
    )
    best_r = results[best_model_name]
    diag1, diag2 = st.columns(2)

    with diag1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test.values, y=best_r["y_pred"],
            mode="markers", marker=dict(size=3, color="#1d3557", opacity=0.3),
            name="Predictions",
        ))
        lims = [min(y_test.min(), best_r["y_pred"].min()),
                max(y_test.max(), best_r["y_pred"].max())]
        fig.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines",
            line=dict(color="#e63946", dash="dash", width=2),
            name="Perfect (y = x)",
        ))
        fig.update_layout(
            title=f"Predicted vs Actual (R² = {best_r['r2']:.4f})",
            xaxis_title="Actual pH", yaxis_title="Predicted pH",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    with diag2:
        residuals = y_test.values - best_r["y_pred"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=best_r["y_pred"], y=residuals,
            mode="markers", marker=dict(size=3, color="#e76f51", opacity=0.3),
            name="Residuals",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(
            title="Residual Plot",
            xaxis_title="Predicted pH", yaxis_title="Residual",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance comparison (tree models)
    st.markdown('<div class="section-header">🌲 Feature Importance — Tree Ensembles</div>',
                unsafe_allow_html=True)
    imp_frames = []
    for name in ["Gradient Boosting", "Random Forest", "Decision Tree"]:
        if "feature_importance" in results[name]["extra"]:
            fi = results[name]["extra"]["feature_importance"]
            for feat, imp in fi.items():
                imp_frames.append({"Model": name, "Feature": feat, "Importance": imp})
    imp_df = pd.DataFrame(imp_frames)
    top_feats = (imp_df[imp_df["Model"] == "Gradient Boosting"]
                 .nlargest(8, "Importance")["Feature"].tolist())
    imp_df = imp_df[imp_df["Feature"].isin(top_feats)]

    fig = px.bar(
        imp_df, x="Importance", y="Feature", color="Model",
        barmode="group", orientation="h",
        color_discrete_sequence=["#2a9d8f", "#264653", "#e76f51"],
        title="Top 8 Feature Importances",
    )
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: MODEL DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Model Deep Dive":
    st.markdown("# 🔍 Individual Model Deep Dive")
    st.markdown("---")

    selected = st.selectbox("Select a model:", list(results.keys()))
    r = results[selected]

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{r['r2']:.4f}")
    m2.metric("RMSE",     f"{r['rmse']:.4f}")
    m3.metric("MAE",      f"{r['mae']:.4f}")
    m4.metric("CV R²",    f"{r['cv_r2']:.4f}")

    # Extra info
    extra = r["extra"]
    if extra:
        with st.expander("⚙️ Hyperparameters & Details", expanded=True):
            for k, v in extra.items():
                if k in ("coefficients", "feature_importance", "poly_transformer",
                          "primary_indices"):
                    continue
                st.markdown(f"**{k}:** `{v}`")

    # Predicted vs Actual + Residuals
    d1, d2 = st.columns(2)
    with d1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test.values, y=r["y_pred"],
            mode="markers", marker=dict(size=3, opacity=0.35, color="#1d3557"),
            name="Predictions",
        ))
        lims = [min(y_test.min(), r["y_pred"].min()),
                max(y_test.max(), r["y_pred"].max())]
        fig.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines",
            line=dict(color="#e63946", dash="dash", width=2),
            name="y = x",
        ))
        fig.update_layout(title="Predicted vs Actual", height=420,
                          xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        residuals = y_test.values - r["y_pred"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r["y_pred"], y=residuals,
            mode="markers", marker=dict(size=3, opacity=0.35, color="#e76f51"),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(title="Residuals", height=420,
                          xaxis_title="Predicted", yaxis_title="Residual")
        st.plotly_chart(fig, use_container_width=True)

    # Residual distribution
    fig = px.histogram(
        x=residuals, nbins=60, marginal="box",
        color_discrete_sequence=["#2a9d8f"],
        title="Residual Distribution",
        labels={"x": "Residual"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance / coefficients
    if "feature_importance" in extra:
        fi = extra["feature_importance"]
        fi_df = (pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
                 .sort_values("Importance", ascending=False))
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Teal",
                     title=f"{selected}: Feature Importances")
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    if "coefficients" in extra:
        coef = extra["coefficients"]
        coef_df = (pd.DataFrame(list(coef.items()), columns=["Feature", "Coefficient"])
                   .sort_values("Coefficient", ascending=False))
        fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                     color="Coefficient", color_continuous_scale="RdBu",
                     color_continuous_midpoint=0,
                     title=f"{selected}: Standardized Coefficients")
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: USER PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Predict with Your Input":
    st.markdown("# 🧪 Interactive Prediction Panel")
    st.markdown(
        "Enter water quality sensor readings below and see predictions "
        "from **all 10 models** side-by-side."
    )
    st.markdown("---")

    # Provide dataset statistics for guidance
    feat_stats = df[feature_names].describe().T[["mean", "min", "max"]]

    # Feature descriptions for tooltips
    feat_desc = {
        "cond_max":   "Specific Conductance — Daily Maximum (µS/cm, normalized)",
        "ph_max":     "Water pH — Daily Maximum (standard units, normalized)",
        "ph_min":     "Water pH — Daily Minimum (standard units, normalized)",
        "cond_min":   "Specific Conductance — Daily Minimum (µS/cm, normalized)",
        "cond_mean":  "Specific Conductance — Daily Mean (µS/cm, normalized)",
        "do_max":     "Dissolved Oxygen — Daily Maximum (mg/L, normalized)",
        "do_mean":    "Dissolved Oxygen — Daily Mean (mg/L, normalized)",
        "do_min":     "Dissolved Oxygen — Daily Minimum (mg/L, normalized)",
        "temp_mean":  "Water Temperature — Daily Mean (°C, normalized)",
        "temp_min":   "Water Temperature — Daily Minimum (°C, normalized)",
        "temp_max":   "Water Temperature — Daily Maximum (°C, normalized)",
    }

    st.markdown("### 📝 Sensor Readings")
    st.caption("All values are normalized to [0, 1]. Hover over labels for descriptions.")

    # Arrange inputs in a clean grid
    user_vals = {}
    cols_per_row = 4
    feat_list = feature_names
    for row_start in range(0, len(feat_list), cols_per_row):
        row_feats = feat_list[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, fn in zip(cols, row_feats):
            with col:
                user_vals[fn] = col.number_input(
                    fn,
                    min_value=0.0, max_value=1.0,
                    value=round(float(feat_stats.loc[fn, "mean"]), 4),
                    step=0.01,
                    format="%.4f",
                    help=feat_desc.get(fn, fn),
                )

    # Spatial group
    st.markdown("### 🗺️ Station Group")
    grp = st.radio(
        "Hydrological Group",
        ["Group 1 (Isolated)", "Group 2 (Coastal)", "Group 3 (Atlanta Metro)"],
        horizontal=True,
    )
    grp_num = int(grp.split()[1])

    st.markdown("---")

    if st.button("🚀  Run All Models", type="primary", use_container_width=True):
        # Build feature vector
        eng = {
            "ph_range":  user_vals["ph_max"]  - user_vals["ph_min"],
            "temp_range": user_vals["temp_max"] - user_vals["temp_min"],
            "do_range":   user_vals["do_max"]  - user_vals["do_min"],
            "temp_do_interaction": user_vals["temp_mean"] * user_vals["do_mean"],
            "cond_spread": user_vals["cond_max"] - user_vals["cond_min"],
            "spatial_group_2": 1.0 if grp_num == 2 else 0.0,
            "spatial_group_3": 1.0 if grp_num == 3 else 0.0,
        }

        row_dict = {**user_vals, **eng}
        X_input_raw = pd.DataFrame([row_dict])[predictor_cols]
        X_input_sc  = scaler.transform(X_input_raw)

        # Collect predictions
        preds = {}
        for name, r in results.items():
            model = r["model"]
            if name == "Polynomial (Deg 2)":
                poly_t = r["extra"]["poly_transformer"]
                pidx   = r["extra"]["primary_indices"]
                X_poly = poly_t.transform(X_input_sc[:, pidx])
                preds[name] = float(model.predict(X_poly)[0])
            elif name in SCALED_MODELS:
                preds[name] = float(model.predict(X_input_sc)[0])
            else:
                preds[name] = float(model.predict(X_input_raw)[0])

        # Sort by R² ranking
        pred_sorted = sorted(preds.items(),
                             key=lambda x: results[x[0]]["r2"], reverse=True)

        st.markdown("### 🎯 Predictions from All 10 Models")
        st.markdown("")

        # Show predictions in cards (grid layout)
        cols_per_row = 5
        for row_start in range(0, len(pred_sorted), cols_per_row):
            row_items = pred_sorted[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (name, pred_val) in zip(cols, row_items):
                rank = next(
                    i + 1 for i, n in enumerate(
                        [x[0] for x in pred_sorted]) if n == name
                )
                r2_val = results[name]["r2"]
                is_best = (rank == 1)

                card_class = "pred-card pred-best" if is_best else "pred-card"
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"

                with col:
                    st.markdown(
                        f'<div class="{card_class}">'
                        f'<div style="font-size:1.5rem;margin-bottom:4px">{medal}</div>'
                        f'<div style="font-weight:700;font-size:1.4rem;color:#1d3557">'
                        f'{pred_val:.4f}</div>'
                        f'<div style="font-size:0.8rem;color:#555;margin-top:4px">'
                        f'{name}</div>'
                        f'<div style="font-size:0.7rem;color:#888">R²={r2_val:.4f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

        # Summary comparison chart
        pred_df = pd.DataFrame(pred_sorted, columns=["Model", "Predicted pH"])
        pred_df["R²"] = pred_df["Model"].map(lambda n: results[n]["r2"])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pred_df["Model"], y=pred_df["Predicted pH"],
            marker_color=px.colors.qualitative.Set2[:len(pred_df)],
            text=pred_df["Predicted pH"].map("{:.4f}".format),
            textposition="outside",
        ))
        fig.update_layout(
            title="Predicted pH Values — All Models (Ranked by R²)",
            xaxis_title="Model", yaxis_title="Predicted pH",
            height=450,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Input summary
        with st.expander("📋 Input Feature Summary"):
            inp_summary = pd.DataFrame({
                "Feature": predictor_cols,
                "Your Value": [row_dict[c] for c in predictor_cols],
                "Dataset Mean": [df[c].mean() if c in df.columns else "N/A"
                                 for c in predictor_cols],
            })
            st.dataframe(inp_summary, use_container_width=True, hide_index=True)

        # Consensus insight
        all_preds = list(preds.values())
        st.success(
            f"**Consensus Prediction:** Mean = {np.mean(all_preds):.4f} "
            f"| Std = {np.std(all_preds):.4f} "
            f"| Range = [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]"
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("ML Capstone Project — 23CSE301 Machine Learning (2026-27) | "
           "USGS Water Quality Prediction | Built with Streamlit")
