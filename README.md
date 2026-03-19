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
