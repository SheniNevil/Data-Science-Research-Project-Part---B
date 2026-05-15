## Tennis Match Prediction: End-to-End Machine Learning Pipeline





#### Project Overview

This project implements a complete machine learning pipeline to predict outcomes of professional tennis matches using historical data from the Jeff Sackmann ATP dataset. It integrates feature engineering, dynamic player strength modeling (Elo), and both interpretable and advanced predictive models such as Logistic Regression and LightGBM.

The workflow follows best practices in sports analytics, including time-aware validation, rolling form metrics and probability calibration. The goal is to produce a reproducible and extensible framework for tennis match prediction suitable for research, portfolio projects or deployment as an analytics dashboard.





#### Key Features

|Module|Description|
|-|-|
|Data Loading|Reads ATP match data from Jeff Sackmann’s open dataset (local CSV or GitHub).|
|Data Cleaning|Normalizes column names, formats dates, fixes surfaces and drops invalid rows. Missing ATP ids use a deterministic name-based surrogate.|
|Elo Rating System|Online rating updates per match for both global and surface-specific player strength.|
|Rolling Features|Computes recent form (win rate over last k matches), surface win-rate, and rest time.|
|Feature Engineering|Builds mirrored rows with difference features, match-level pre-match Elo and optional player id columns for tree models.|
|Baseline Models|Implements Win% and Logistic Regression baselines with standardized features.|
|Advanced Models|LightGBM classifier + LambdaRank learning-to-rank booster with early stopping.|
|Time-Aware Validation|Splits data chronologically (train/val/test) to avoid lookahead bias.|
|Evaluation|Calculates Accuracy, AUC, Brier score, and produces calibration (reliability) plots.|
|Model Persistence|Saves trained models (`.joblib`, `.txt`) for reuse or deployment.|





#### Data Source

Jeff Sackmann’s Tennis ATP Dataset (freely available on GitHub): https://github.com/JeffSackmann/tennis\_atp



Each yearly file contains match-level statistics such as player names, ranks, surface, scores, and performance metrics.





#### Project Structure

```
tennis\_prediction
├── data
│   └── raw
│       └── tennis\_atp-master   # Main-tour atp\_matches\_YYYY.csv (1968–2024)
├── models                     # Saved artifacts (created by notebook)
├── src
│   ├── data\_loader.py          # Load \& subset ATP data (year range, max\_rows)
│   ├── preprocessing.py       # Canonicalize columns,clean dates/surface/ranks
│   ├── features.py            # Elo,rolling form,H2H(Bayesian),build\_diff\_dataset
│   └── models
│       └── \_\_init\_\_.py
├── tennis-prediction.ipynb     # Full pipeline: load → features → train → save
├── app.py                      # Streamlit dashboard (predict \& explore)
├── requirements.txt
└── README.md
```



#### How to Run:

1. ###### Create virtual environment (Python 3.12)

```
bash
cd tennis\_prediction
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

###### 2\. Prepare data

Place the full [Jeff Sackmann tennis\_atp](https://github.com/JeffSackmann/tennis_atp) repo (or main-tour files) in: data/raw/tennis\_atp-master/

The notebook and `src.data\_loader` use main-tour singles only (`atp\_matches\_YYYY.csv`). We can subset by `YEAR\_MIN`, `YEAR\_MAX` or `MAX\_ROWS` in the notebook to speed up runs. If no local data is found, the notebook falls back to downloading a sample CSV.



###### 3\. Run the notebook

Open `tennis-prediction.ipynb`, select the Python 3 kernel and run all cells. This will:

* Load and clean data
* Compute Elo, rolling form, and head-to-head features
* Build the difference-feature dataset and train/val/test split
* Train Logistic Regression, LightGBM classifier, and LambdaRank
* Optionally run Optuna tuning and SHAP explainability
* Save models and scaler to `models`
* 

###### 4\. Run the Streamlit dashboard

After training and saving models:

```
bash
.venv/bin/streamlit run app.py
```

Use the Predict tab to enter feature differences and get P(Player A wins). The Explore tab can load sample CSVs from `data/raw/`.





#### Feature Highlights

* Elo Ratings: dynamic measure of player strength, updated after every match.
* Surface-Specific Elo: tracks player strength separately for hard, clay, and grass.
* Form Metrics: win rate over last *K* matches (default *K=8*).
* Surface Winrate: win rate over last *K* matches on a specific surface.
* Rest/Fatigue: days since last match.
* Rank Difference: adjusted difference in ATP rankings.
* Head-to-head: Bayesian-smoothed H2H win rate (prior weight 3, prior 0.5) as of before each match.
* Player IDs: `winner\_id` / `loser\_id` from Sackmann when present; otherwise a stable synthetic id from the player name so entities stay mergeable.
* Match-level Elo columns: Pre-match global and surface Elo for each side (`elo\_a`/`elo\_b`, `elo\_surf\_a`/`elo\_surf\_b`) alongside differences, for nonlinear favorite/underdog effects in tree models.





#### Validation Strategy

The project avoids data leakage via time-aware splits:

* Train: first 70% of matches chronologically
* Validation: next 15%
* Test: final 15%

This ensures the model only sees past matches when predicting future outcomes, reflecting realistic prediction conditions.

### 

#### Repeated measures and mixed effects

Training rows are not independent: the same player appears in many matches and the mirrored two-row-per-match layout induces correlation.

LightGBM uses `player\_a\_id` and `player\_b\_id` as categorical features—this is a practical, tree-based way to capture player-specific deviations from the Elo-based signal. The notebook prints a small per-player Brier summary on the test split as a repeated-measures diagnostic.





#### Model Details:

1. ###### Logistic Regression

A linear interpretable baseline trained on standardized numeric features only (difference features and match-level Elo levels), player id columns are excluded so the baseline stays low-dimensional.

```
python
from sklearn.linear\_model import LogisticRegression
clf = LogisticRegression(max\_iter=1000)
clf.fit(X\_train\_scaled, y\_train)
```

###### 2\. LightGBM Classifier

Tree-based gradient boosting model for non-linear relationships. When player ids are present, they are declared as categorical with moderate `min\_data\_in\_leaf` / `max\_depth` to limit overfitting on rare players.

```
python
import lightgbm as lgb
params = {
'objective': 'binary', 'metric': 'auc',
'learning\_rate': 0.05, 'num\_leaves': 31
}
model = lgb.train(params, train\_data, valid\_sets=\[val\_data])
```

###### 3\. LambdaRank (Learning-to-Rank)

Implemented LightGBM LambdaRank head that ingests one row per player per match (winner + loser) with Elo, rolling form, rest, rank, and surface indicators. Uses group size of 2, optimises and reports match-level accuracy, pairwise AUC, and Brier scores for train/val/test splits.





#### Evaluation Metrics

* Accuracy: correct match winner predictions
* ROC-AUC: ranking quality of predicted probabilities
* Brier Score: calibration of probabilities
* Reliability Diagram: visual calibration check
* Calibration plot: Predicted probability vs. observed win frequency (validation set)





#### Saved Artifacts

|File|Description|
|-|-|
|`models/scaler.joblib`|StandardScaler used for feature normalization|
|`models/logistic.joblib`|Trained Logistic Regression model|
|`models/logistic\_features.joblib`|Column names used for the logistic baseline (numeric only)|
|`models/lgb\_classifier\_features.joblib`|Full feature list for the LightGBM classifier|
|`models/lgb\_model.txt`|Trained LightGBM classifier (best iteration)|
|`models/lgb\_lambdarank.txt`|Trained LightGBM LambdaRank booster|
|`models/rank\_features.joblib`|Feature list used when building ranking inputs|

These files can be loaded directly for inference or API deployment.




**NOTE: THE FOLDERS HAVE BEEN ZIPPED AND UPLOADED IN GITHUB DUE TO EXCEEDING FOLDER SIZE.**

