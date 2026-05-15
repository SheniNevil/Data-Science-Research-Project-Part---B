# 🎾 Tennis Match Winner Prediction — End-to-End Machine Learning Pipeline

## 📘 Project Overview
This project implements a **complete machine learning pipeline** to predict outcomes of professional tennis matches using historical data from the **Jeff Sackmann ATP dataset**. It integrates feature engineering, dynamic player strength modeling (Elo), and both interpretable and advanced predictive models such as **Logistic Regression** and **LightGBM**.

The workflow follows best practices in **sports analytics**, including **time-aware validation**, **rolling form metrics**, and **probability calibration**. The goal is to produce a reproducible and extensible framework for tennis match prediction, suitable for research, portfolio projects, or deployment as an analytics dashboard.

---

## ⚙️ Key Features

| Module | Description |
|--------|-------------|
| **Data Loading** | Reads ATP match data from Jeff Sackmann’s open dataset (local CSV or GitHub). |
| **Data Cleaning** | Normalizes column names, formats dates, fixes surfaces, coerces `winner_id`/`loser_id`, and drops invalid rows. Missing ATP ids use a **deterministic name-based surrogate** (see `src.preprocessing.synthetic_player_id`). |
| **Elo Rating System** | Online rating updates per match for both global and surface-specific player strength. |
| **Rolling Features** | Computes recent form (win rate over last *K* matches), surface winrate, and rest time. |
| **Feature Engineering** | Builds mirrored rows with difference features, **match-level pre-match Elo** (`elo_a`/`elo_b`, surface analogs), and optional **player id** columns for tree models. **`elo_spec_id`** tags the Elo implementation (`ELO_SPEC_ID` in `src/features.py`). |
| **Baseline Models** | Implements Win% and Logistic Regression baselines with standardized features. |
| **Advanced Models** | LightGBM classifier + LambdaRank learning-to-rank booster with early stopping. |
| **Time-Aware Validation** | Splits data chronologically (train/val/test) to avoid lookahead bias. |
| **Evaluation** | Calculates Accuracy, AUC, Brier score, and produces calibration (reliability) plots. |
| **Model Persistence** | Saves trained models (`.joblib`, `.txt`) for reuse or deployment. |

---

## 🧠 Data Source
**Jeff Sackmann’s Tennis ATP Dataset** — freely available on GitHub:

```
https://github.com/JeffSackmann/tennis_atp
```

Each yearly file (e.g., `atp_matches_2019.csv`) contains match-level statistics such as player names, ranks, surface, scores, and performance metrics.

---

## 🏗️ Project Structure

```
📂 tennis_prediction/
├── data/
│   └── raw/
│       └── tennis_atp-master/   # Main-tour atp_matches_YYYY.csv (1968–2024)
├── models/                     # Saved artifacts (created by notebook)
├── src/
│   ├── data_loader.py          # Load & subset ATP data (year range, max_rows)
│   ├── preprocessing.py       # Canonicalize columns, clean dates/surface/ranks
│   ├── features.py            # Elo, rolling form, H2H (Bayesian), build_diff_dataset
│   └── models/
│       └── __init__.py
├── tennis-prediction.ipynb     # Full pipeline: load → features → train → save
├── app.py                      # Streamlit dashboard (predict & explore)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Create virtual environment (Python 3.12)
```bash
cd tennis_prediction
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Prepare data
Place the full [Jeff Sackmann tennis_atp](https://github.com/JeffSackmann/tennis_atp) repo (or main-tour `atp_matches_YYYY.csv` files) in:
```
data/raw/tennis_atp-master/
```
The notebook and `src.data_loader` use **main-tour singles** only (`atp_matches_YYYY.csv`). You can subset by `YEAR_MIN`, `YEAR_MAX`, or `MAX_ROWS` in the notebook to speed up runs. If no local data is found, the notebook falls back to downloading a sample CSV.

### 3️⃣ Run the notebook
Open `tennis-prediction.ipynb`, select the **Python 3 (.venv)** kernel, and run all cells. This will:
- Load and clean data (with optional subsetting)
- Compute Elo, rolling form, and head-to-head features
- Build the difference-feature dataset and train/val/test split
- Train Logistic Regression, LightGBM classifier, and LambdaRank
- Optionally run Optuna tuning and SHAP explainability
- Save models and scaler to `models/`

### 4️⃣ Run the Streamlit dashboard
After training and saving models:
```bash
.venv/bin/streamlit run app.py
```
Use the **Predict** tab to enter feature differences and get P(Player A wins). The **Explore** tab can load sample CSVs from `data/raw/`.

---

## 📊 Model Performance (Example)
| Model | Accuracy | AUC | Brier Score |
|--------|-----------|-----------|-------------|
| Logistic Regression | ~0.63 | ~0.69 | ~0.23 |
| LightGBM | ~0.67 | ~0.73 | ~0.21 |

> *Actual values depend on dataset year, feature configuration, and random seed.*

---

## 🧩 Feature Highlights
- **Elo Ratings:** dynamic measure of player strength, updated after every match.
- **Surface-Specific Elo:** tracks player strength separately for hard, clay, and grass.
- **Form Metrics:** win rate over last *K* matches (default *K=8*).
- **Surface Winrate:** win rate over last *K* matches on a specific surface.
- **Rest/Fatigue:** days since last match.
- **Rank Difference:** adjusted difference in ATP rankings.
- **Head-to-head:** Bayesian-smoothed H2H win rate (prior weight 3, prior 0.5) as of before each match.
- **Player IDs:** `winner_id` / `loser_id` from Sackmann when present; otherwise a stable synthetic id from the player name so entities stay mergeable.
- **Match-level Elo columns:** Pre-match global and surface Elo for each side (`elo_a`/`elo_b`, `elo_surf_a`/`elo_surf_b`) alongside differences, for nonlinear favorite/underdog effects in tree models.

---

## 🧪 Validation Strategy
The project avoids data leakage via **time-aware splits**:
- Train: first 70% of matches chronologically
- Validation: next 15%
- Test: final 15%

This ensures the model only sees *past* matches when predicting *future* outcomes, reflecting realistic prediction conditions.

### Repeated measures and “mixed effects”
Training rows are **not independent**: the same player appears in many matches, and the mirrored two-row-per-match layout induces correlation. **LightGBM** uses `player_a_id` and `player_b_id` as **categorical** features—this is a practical, tree-based way to capture **player-specific deviations** from the Elo-based signal; it is **not** the same as estimating random-effect variance in a **GLMM** (see the optional appendix in `tennis-prediction.ipynb` and tools such as R `lme4::glmer`, **pymer4**, or **Bambi** for a formal hierarchical baseline). The notebook prints a small **per-player Brier** summary on the test split as a repeated-measures diagnostic.

---

## 🔧 Model Details
### Logistic Regression
A linear interpretable baseline trained on standardized **numeric** features only (difference features and match-level Elo levels); **player id** columns are excluded so the baseline stays low-dimensional.
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)
```

### LightGBM Classifier
Tree-based gradient boosting model for non-linear relationships. When player ids are present, they are declared as **categorical** with moderate `min_data_in_leaf` / `max_depth` to limit overfitting on rare players.
```python
import lightgbm as lgb
params = {
    'objective': 'binary', 'metric': 'auc',
    'learning_rate': 0.05, 'num_leaves': 31
}
model = lgb.train(params, train_data, valid_sets=[val_data])
```

### LambdaRank (Learning-to-Rank)
Implemented LightGBM LambdaRank head that ingests one row per player per match (winner + loser) with Elo, rolling form, rest, rank, and surface indicators. Uses group size of 2, optimises `ndcg@2`, and reports match-level accuracy, pairwise AUC, and Brier scores for train/val/test splits.

---

## 📈 Evaluation Metrics
- **Accuracy** — correct match winner predictions
- **ROC-AUC** — ranking quality of predicted probabilities
- **Brier Score** — calibration of probabilities
- **Reliability Diagram** — visual calibration check

Example calibration plot:
```
Predicted probability vs. observed win frequency (validation set)
```

---

## 🧩 Saved Artifacts
| File | Description |
|------|--------------|
| `models/scaler.joblib` | StandardScaler used for feature normalization |
| `models/logistic.joblib` | Trained Logistic Regression model |
| `models/logistic_features.joblib` | Column names used for the logistic baseline (numeric only) |
| `models/lgb_classifier_features.joblib` | Full feature list for the LightGBM classifier |
| `models/lgb_model.txt` | Trained LightGBM classifier (best iteration) |
| `models/lgb_lambdarank.txt` | Trained LightGBM LambdaRank booster |
| `models/rank_features.joblib` | Feature list used when building ranking inputs |

These files can be loaded directly for inference or API deployment.

---

## 🔍 Done & Next Steps
- [x] Add **head-to-head features** with Bayesian smoothing.
- [x] Implement **Optuna hyperparameter tuning** for LightGBM (notebook section 7b).
- [x] Extend to **LambdaRank** for match ranking optimization.
- [x] Add **SHAP explainability** (notebook section 8b).
- [x] Integrate with **Streamlit dashboard** (`app.py`).

---

## ⚖️ Ethical Use
This model is intended for **research and educational purposes only**. If used for prediction or betting-related applications, ensure:
- Transparency of model uncertainty
- No facilitation of gambling or exploitation

---

## 🧑‍💻 Author & Credits
**Developed by:** [Sheni]  
**Data Source:** Jeff Sackmann, *Tennis ATP Dataset*  
**License:** MIT (open for research and educational use)

---

## 📚 References
1. Sackmann, Jeff — *Tennis ATP Match Data* ([GitHub Repository](https://github.com/JeffSackmann/tennis_atp))  
2. Glickman, Mark — *Elo rating system and extensions*  
3. LightGBM Documentation — *Microsoft Research*  

---

> ⚡ *"Thanks"*

