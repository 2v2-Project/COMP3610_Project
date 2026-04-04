# Clash Royale Analytics Engine

> COMP 3610 — Big Data Analytics Project

A data-driven esports analytics platform that analyses 12.4 M+ Clash Royale
ladder matches, predicts match outcomes with XGBoost, and surfaces strategic
insights through an interactive Streamlit dashboard.

## Project Structure

```
├── data/
│   ├── raw/                 # Daily match CSVs (Oct 2–11, 2023)
│   ├── processed/           # Cleaned parquets, feature matrices, metadata
│   └── outputs/             # Exploratory analysis plots
├── scr/                     # Data pipeline & model training scripts
│   ├── 01_load_data.py            # Load & benchmark raw CSVs
│   ├── 02_preprocess_clash_royale_data.py  # Clean, deduplicate, create target
│   ├── 03_build_deck_feature_matrices.py   # Card one-hot & elixir features
│   ├── 04_analyze_common_cards.py          # (exploratory) card frequency plots
│   ├── 05_analyze_win_rates.py             # (exploratory) win-rate analysis
│   ├── 06_archetype_synergy_features.py    # Archetype & synergy features
│   ├── 07_matchup_features.py              # Cross-deck matchup features
│   ├── 08_assemble_final_ml_dataset.py     # Merge all features → final dataset
│   ├── 09_train_logistic_regression.py     # (experiment) Logistic Regression
│   ├── 10_train_random_forest.py           # (experiment) Random Forest
│   ├── 11_tune_random_forest.py            # (experiment) Tuned Random Forest
│   ├── 12_train_xgboost.py                # (experiment) XGBoost with CV tuning
│   ├── 13_train_final.py                  # Final XGBoost training → deployed model
│   └── utils/
│       └── metadata_utils.py              # RoyaleAPI card metadata fetcher
├── models/
│   ├── xgboost_model.joblib   # Deployed XGBoost model
│   ├── columns.json           # Feature schema (column order)
│   └── xgboost_metrics.json   # Evaluation metrics
├── webapp/
│   ├── app.py                 # Streamlit home page
│   ├── pages/
│   │   ├── 01_overview.py         # Dataset dashboard
│   │   ├── 02_popular_decks.py    # Top deck browser
│   │   ├── 03_win_predictor.py    # ML win probability predictor
│   │   ├── 04_matchup.py          # Deck vs deck matchup analysis
│   │   ├── 05_trends.py           # Meta trends & card usage over time
│   │   ├── 06_archetype_insights.py # Archetype heatmaps & SHAP
│   │   ├── 07_game_theory.py      # Payoff matrices & Nash equilibrium
│   │   └── 08_recommendations.py  # Deck & card-swap suggestions
│   ├── static/                # Banner images
│   └── utils/                 # Shared webapp utilities
│       ├── metadata.py            # RoyaleAPI integration & card metadata
│       ├── model_loader.py        # Model & schema loading
│       ├── preprocess.py          # Feature vector construction
│       ├── deck_helpers.py        # Deck key, archetype, elixir helpers
│       ├── explanation_engine.py  # SHAP + rule-based prediction explanations
│       ├── shap_utils.py          # SHAP explainer wrappers
│       ├── uncertainty.py         # Confidence / uncertainty estimation
│       ├── recommendation.py      # Card-swap & deck ranking
│       ├── prediction.py          # Prediction wrappers
│       ├── data_loader.py         # Cached data loading utilities
│       └── ui_helpers.py          # CSS & UI component helpers
├── requirements.txt
└── README.md
```

## Dataset

This project uses the [Clash Royale Games](https://www.kaggle.com/datasets/s1m0n38/clash-royale-games) dataset from Kaggle.

- **Time Period:** October 2–11, 2023
- **Match Type:** Ladder matches (4,000 + trophies)
- **Scale:** ~12.4 million matches, 107 unique cards, 3.6 M+ unique players

Card metadata (names, elixir costs, types, icons) is enriched via the
[RoyaleAPI](https://royaleapi.github.io/cr-api-data/json/cards.json) data endpoint.

To reproduce:

1. Download the dataset from Kaggle.
2. Extract the archive.
3. Place the daily CSV files (`20231002.csv` – `20231011.csv`) into `data/raw/`.

## Setup

```powershell
# 1) Create virtual environment (once)
python -m venv .venv

# 2) Activate
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Pipeline

Run the core pipeline in order:

```powershell
python scr\01_load_data.py
python scr\02_preprocess_clash_royale_data.py
python scr\03_build_deck_feature_matrices.py
python scr\06_archetype_synergy_features.py
python scr\07_matchup_features.py
python scr\08_assemble_final_ml_dataset.py
```

Optional exploratory analysis (generates plots in `data/outputs/`):

```powershell
python scr\04_analyze_common_cards.py
python scr\05_analyze_win_rates.py
```

### Expected Outputs (`data/processed/`)

| File | Description |
|------|-------------|
| `clash_royale_clean.csv` / `.parquet` | Cleaned match data |
| `load_benchmark.csv` | Load-time benchmarks |
| `card_list.csv` | Unique card IDs |
| `card_metadata.csv` / `card_metadata_raw.json` | API-sourced card metadata |
| `player_card_feature_matrix.parquet` | Player 1 card one-hot features |
| `opponent_card_feature_matrix.parquet` | Player 2 card one-hot features |
| `deck_elixir_features.parquet` | Elixir cost features per deck |
| `archetype_features.parquet` | Detected archetype labels |
| `synergy_features.parquet` | Card synergy scores |
| `archetype_synergy_features.parquet` | Combined archetype + synergy |
| `matchup_features.parquet` | Cross-deck matchup features |
| `opponent_elixir_features.parquet` | Opponent elixir features |
| `matchup_deck_diff_features.parquet` | Pairwise deck difference features |
| `final_ml_dataset.parquet` | Merged ML-ready dataset (305 columns) |
| `clean_training_dataset.parquet` | Quality-checked training set |
| `final_dataset_quality_report.json` | Dataset quality report |

## Model Training and Selection

Multiple models were trained and compared on a 500,000-row sample:

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.547 | 0.590 | 0.574 |
| Random Forest | 0.564 | 0.563 | 0.591 |
| Tuned Random Forest | 0.568 | 0.514 | 0.596 |
| **XGBoost** | **0.590** | **0.564** | **0.626** |

XGBoost was selected as the deployed model based on the best ROC-AUC
(+3.4 pp over Random Forest), which is the primary metric for
probabilistic ranking quality.

Training scripts `09`–`12` document the model selection experiments.
The final deployed model is produced by `13_train_final.py`.

### Final Model (XGBoost)

Trained on a 1,000,000-row stratified sample with fixed hyper-parameters
selected from prior RandomizedSearchCV tuning:

```powershell
python scr\13_train_final.py
```

| Metric   | Value |
|----------|-------|
| Accuracy | 0.587 |
| F1 Score | 0.559 |
| ROC-AUC  | 0.621 |

**Note:** The moderate ROC-AUC reflects that deck composition alone is a
limited predictor of match outcomes — player skill, card levels, and
in-match decisions are not captured in this dataset.

### Saved Artifacts (`models/`)

| File | Description |
|------|-------------|
| `xgboost_model.joblib` | Deployed XGBoost model |
| `columns.json` | Feature schema (column order for inference) |
| `xgboost_metrics.json` | Evaluation metrics |

## Web Application

An 8-page Streamlit dashboard for interactive exploration:

| Page | Description |
|------|-------------|
| **Overview** | Key dataset statistics, trophy distributions, win/loss breakdown |
| **Popular Decks** | Most-played decks with archetype, confidence, and elixir filters |
| **Win Predictor** | Build a deck → ML win probability with SHAP explanations |
| **Matchup Analysis** | Deck vs deck prediction with feature-level breakdown |
| **Trends** | Card usage trends, archetype distribution, meta evolution |
| **Archetype Insights** | Archetype vs archetype win-rate heatmaps, SHAP importance |
| **Game Theory** | Payoff matrices, Nash equilibrium, dominant strategy analysis |
| **Recommendations** | Card-swap suggestions and top historical deck rankings |

### Run the App

```powershell
cd webapp
streamlit run app.py
```

### Key Features

- **XGBoost predictions** with uncertainty estimation and confidence labels
- **SHAP explanations** for individual predictions (local feature importance)
- **Game theory analysis** — archetypes as strategies, Nash equilibrium computation
- **RoyaleAPI integration** for card metadata, icons, and enrichment
- **DuckDB** for fast analytical queries over 12 M+ row parquet files
- **Deck recommendation engine** with model-scored card swaps