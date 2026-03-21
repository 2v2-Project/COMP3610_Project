# COMP3610_Project

## Dataset

This project uses the Clash Royale Games dataset from Kaggle:

https://www.kaggle.com/datasets/s1m0n38/clash-royale-games

To reproduce results:

1. Download the dataset from Kaggle.
2. Extract the archive.
3. Place selected daily CSV files in `data/raw/`.

## Setup

Run from project root:

```powershell
# 1) Create virtual environment (once)
python -m venv .venv

# 2) Activate
.\.venv\Scripts\Activate.ps1

# 3) Upgrade pip
python -m pip install --upgrade pip

# 4) Install dependencies
python -m pip install -r requirements.txt
```

## Quick Sanity Checks

```powershell
python -c "import pandas, polars, duckdb, pyarrow; print('deps ok')"
python -m py_compile scr\01_load_data.py
python -m py_compile scr\02_preprocess_clash_royale_data.py
python -m py_compile scr\03_build_deck_feature_matrices.py
python -m py_compile scr\06_archetype_synergy_features.py
python -m py_compile scr\07_matchup_features.py
python -m py_compile scr\08_assemble_final_ml_dataset.py
```

## End-to-End Run (All Core Phases)

Run the pipeline in this order:

```powershell
python scr\01_load_data.py
python scr\02_preprocess_clash_royale_data.py
python scr\03_build_deck_feature_matrices.py
python scr\06_archetype_synergy_features.py
python scr\07_matchup_features.py
python scr\08_assemble_final_ml_dataset.py

python scr/11_tune_random_forest.py --sample-size 200000 --cv 2 --n-iter 6
```

Optional analysis scripts (not required for final dataset assembly):

```powershell
python scr\04_analyze_common_cards.py
python scr\05_analyze_win_rates.py
```

## Expected Outputs (data/processed)

Core preprocessing and feature outputs:

- `clash_royale_clean.csv`
- `clash_royale_clean.parquet`
- `load_benchmark.csv`
- `card_list.csv`
- `card_metadata.csv`
- `player_card_feature_matrix.parquet`
- `opponent_card_feature_matrix.parquet`
- `deck_elixir_features.parquet`
- `archetype_features.parquet`
- `synergy_features.parquet`
- `archetype_synergy_features.parquet`
- `matchup_features.parquet`
- `opponent_elixir_features.parquet`
- `matchup_deck_diff_features.parquet`
- `final_ml_dataset.parquet`
- `clean_training_dataset.parquet`
- `final_dataset_quality_report.json`

Verify outputs:

```powershell
Get-ChildItem -Path data/processed | Select-Object Name,Length
```
## Model Evaluation 
## Model Training and Selection

This project evaluated multiple machine learning models to predict match outcomes based on gameplay features derived from a large dataset (~13.5 million rows).

## Models Evaluated

The following models were trained and compared on a consistent 500,000-row sample:

Logistic Regression
Random Forest (manually configured)
Tuned Random Forest (RandomizedSearchCV)

Results Summary
Model	Accuracy	F1 Score	ROC-AUC
Logistic Regression	0.547	0.590	0.574
Random Forest	0.564	0.563	0.591
Tuned Random Forest	0.568	0.514	0.596

## Model Selection

Although Logistic Regression achieved the highest F1 score and the tuned Random Forest achieved the highest ROC-AUC, the baseline Random Forest model was selected for deployment.

This decision was based on:

strong overall performance across all metrics
balanced classification capability (precision vs recall)
stability compared to the tuned model
ability to capture nonlinear relationships in the data

The tuned model improved ROC-AUC slightly but significantly reduced F1 score, indicating poorer classification balance.

## Final Model Training

The selected Random Forest model was retrained using a 1,000,000-row sample, which represented the largest dataset size that could be processed reliably within local hardware constraints.

Final Model Configuration
n_estimators = 200
max_depth = 15
min_samples_split = 5
min_samples_leaf = 2
class_weight = "balanced"
Final Performance
Accuracy: 0.5642
F1 Score: 0.5626
ROC-AUC: 0.5915

These results were consistent with earlier experiments, indicating stable model behavior.

## Saved Artifacts

The final trained model and required metadata were saved for deployment:

models/
├── random_forest.pkl           # serialized model
├── columns.json                # feature schema (column order)
└── random_forest_metrics.json  # evaluation metrics
Why the Feature Schema is Saved

The model expects input features in a specific order.
The columns.json file ensures that:

input data is aligned correctly during inference
missing features can be handled safely
predictions remain consistent between training and deployment