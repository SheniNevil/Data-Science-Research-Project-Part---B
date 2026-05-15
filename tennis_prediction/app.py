"""
Streamlit dashboard for Tennis Match Winner Prediction.
Run with: streamlit run app.py
Use the project's .venv: .venv/bin/streamlit run app.py
"""
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"
DATA_RAW = ROOT / "data" / "raw"

# Defaults for features not exposed as primary sliders (match-level Elo, synthetic player ids)
_LGB_EXTRA_DEFAULTS = {
    "elo_a": 1500.0,
    "elo_b": 1500.0,
    "elo_surf_a": 1500.0,
    "elo_surf_b": 1500.0,
    "player_a_id": 1_000_000_001,
    "player_b_id": 1_000_000_002,
}


@st.cache_data
def load_models():
    """Load scaler and LightGBM classifier from models/."""
    scaler_path = MODEL_DIR / "scaler.joblib"
    lgb_path = MODEL_DIR / "lgb_model.txt"
    rank_features_path = MODEL_DIR / "rank_features.joblib"
    if not scaler_path.exists() or not lgb_path.exists():
        return None, None, None
    scaler = joblib.load(scaler_path)
    clf = lgb.Booster(model_file=str(lgb_path))
    rank_features = joblib.load(rank_features_path) if rank_features_path.exists() else []
    return scaler, clf, rank_features


@st.cache_data
def load_feature_names():
    """Infer feature names from saved model or default."""
    rank_path = MODEL_DIR / "rank_features.joblib"
    if rank_path.exists():
        # Difference-feature model uses different set; we need to match what the classifier was trained on
        pass
    return [
        "elo_diff", "elo_surf_diff", "rank_diff", "form_diff", "surf_form_diff",
        "days_since_diff", "is_hard", "is_clay", "is_grass"
    ]


def main():
    st.set_page_config(page_title="Tennis Match Prediction", layout="wide")
    st.title("Tennis Match Winner Prediction")
    st.markdown("Predict match outcome using Elo, form, and optional head-to-head features. "
                "Models are trained on ATP main-tour data (Jeff Sackmann).")

    scaler, clf, rank_features = load_models()
    if scaler is None or clf is None:
        st.warning("Saved models not found. Run the notebook `tennis-prediction.ipynb` to train and save models to `models/`.")
        st.stop()

    try:
        feature_names = list(clf.feature_name()) or []
    except Exception:
        feature_names = []
    if not feature_names:
        feature_names = ["elo_diff", "elo_surf_diff", "rank_diff", "form_diff", "surf_form_diff",
                         "days_since_diff", "is_hard", "is_clay", "is_grass"]

    tab_overview, tab_predict, tab_explore = st.tabs(["Overview", "Predict", "Explore"])

    with tab_overview:
        st.subheader("Project overview")
        st.markdown("""
        - **Data:** ATP main-tour singles (Jeff Sackmann tennis_atp dataset).
        - **Features:** Elo (global + surface), match-level pre-match Elo levels, rolling form, surface form, rest days, rank difference, optional H2H, ATP/synthetic **player IDs** as tree categoricals (approximation to random effects).
        - **Models:** Logistic Regression (baseline, numeric features only), LightGBM classifier, LambdaRank (learning-to-rank).
        - **Validation:** Time-based 70/15/15 train/val/test split; rows are **not** iid (repeated players).
        """)
        st.markdown("See `tennis-prediction.ipynb` for full pipeline and `README.md` for setup.")

    with tab_predict:
        st.subheader("Predict match outcome")
        st.markdown("Enter relative strengths (e.g. from past stats). Difference = Player A value − Player B value. "
                    "Positive Elo/form diff favors Player A.")
        col1, col2 = st.columns(2)
        with col1:
            elo_diff = st.number_input("Elo difference (A − B)", value=0.0, step=10.0)
            elo_surf_diff = st.number_input("Surface Elo difference (A − B)", value=0.0, step=10.0)
            rank_diff = st.number_input("Rank difference (B rank − A rank; higher = A better)", value=0.0, step=1.0)
        with col2:
            form_diff = st.slider("Form diff (A − B win rate last K)", -0.5, 0.5, 0.0, 0.05)
            surf_form_diff = st.slider("Surface form diff (A − B)", -0.5, 0.5, 0.0, 0.05)
            days_since_diff = st.number_input("Days since last match (A − B; negative = A more rested)", value=0.0, step=1.0)
        surface = st.selectbox("Surface", ["hard", "clay", "grass"])
        is_hard = 1 if surface == "hard" else 0
        is_clay = 1 if surface == "clay" else 0
        is_grass = 1 if surface == "grass" else 0

        h2h_diff = 0.0
        if "h2h_diff" in feature_names:
            h2h_diff = st.slider("H2H diff (A win rate vs B − B vs A)", -0.5, 0.5, 0.0, 0.05)

        extra_needed = [c for c in _LGB_EXTRA_DEFAULTS if c in feature_names]
        with st.expander("Advanced: match-level Elo & player IDs", expanded=bool(extra_needed)):
            st.caption("Shown when the saved model includes these features (defaults: equal 1500 Elo; placeholder ids).")
            adv_elo_a = st.number_input("Player A pre-match Elo (global)", value=_LGB_EXTRA_DEFAULTS["elo_a"], step=10.0)
            adv_elo_b = st.number_input("Player B pre-match Elo (global)", value=_LGB_EXTRA_DEFAULTS["elo_b"], step=10.0)
            adv_elo_surf_a = st.number_input("Player A surface Elo (pre-match)", value=_LGB_EXTRA_DEFAULTS["elo_surf_a"], step=10.0)
            adv_elo_surf_b = st.number_input("Player B surface Elo (pre-match)", value=_LGB_EXTRA_DEFAULTS["elo_surf_b"], step=10.0)
            adv_pid_a = st.number_input("Player A id (ATP or synthetic)", value=_LGB_EXTRA_DEFAULTS["player_a_id"], step=1, format="%d")
            adv_pid_b = st.number_input("Player B id (ATP or synthetic)", value=_LGB_EXTRA_DEFAULTS["player_b_id"], step=1, format="%d")

        row = {
            "elo_diff": elo_diff, "elo_surf_diff": elo_surf_diff, "rank_diff": rank_diff,
            "form_diff": form_diff, "surf_form_diff": surf_form_diff, "days_since_diff": days_since_diff,
            "is_hard": is_hard, "is_clay": is_clay, "is_grass": is_grass,
        }
        if "h2h_diff" in feature_names:
            row["h2h_diff"] = h2h_diff
        row["elo_a"] = adv_elo_a
        row["elo_b"] = adv_elo_b
        row["elo_surf_a"] = adv_elo_surf_a
        row["elo_surf_b"] = adv_elo_surf_b
        row["player_a_id"] = int(adv_pid_a)
        row["player_b_id"] = int(adv_pid_b)

        X = pd.DataFrame([row])
        for c in feature_names:
            if c not in X.columns:
                X[c] = _LGB_EXTRA_DEFAULTS.get(c, 0.0)
        X = X[feature_names]
        for c in ("player_a_id", "player_b_id"):
            if c in X.columns:
                X[c] = X[c].astype(int)
        prob = clf.predict(X)[0]
        if hasattr(prob, "__len__"):
            prob = prob[0] if len(prob) > 0 else 0.5
        prob = float(prob)
        st.metric("P(Player A wins)", f"{prob:.1%}")
        st.caption("Based on LightGBM classifier. Train the notebook with your data for best results.")

    with tab_explore:
        st.subheader("Explore")
        st.markdown("If you have processed data or match CSVs in `data/raw/`, you can load and filter here. "
                    "Otherwise run the notebook first to generate models and optional processed data.")
        if (ROOT / "data" / "raw").exists():
            csvs = list(Path(ROOT / "data" / "raw").rglob("atp_matches_*.csv"))
            if csvs:
                choice = st.selectbox("Pick a file", [str(p.relative_to(ROOT)) for p in csvs[:20]])
                if choice:
                    path = ROOT / choice
                    try:
                        df = pd.read_csv(path, nrows=1000, low_memory=False)
                        st.dataframe(df.head(100), use_container_width=True)
                    except Exception as e:
                        st.error(str(e))
            else:
                st.info("No atp_matches_*.csv files found in data/raw/.")
        else:
            st.info("No data/raw/ directory found.")


if __name__ == "__main__":
    main()
