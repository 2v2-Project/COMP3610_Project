# COMP3610_Project

## Just some checks for everybody
# 1) Create venv (only once)
python -m venv .venv

# 2) Activate it
.\.venv\Scripts\Activate.ps1

# 3) Upgrade pip
python -m pip install --upgrade pip

# 4) Install dependencies
python -m pip install -r requirements.txt

# 5) Quick sanity checks
python -c "import pandas, polars, duckdb; print('deps ok')"
python -m py_compile scr\load_data.py
python -m py_compile scr\preprocess_clash_royale_data.py

# 6) Run preprocessing pipeline
# Run load script
python scr\load_data.py

# Run preprocess script
python scr\preprocess_clash_royale_data.py

## Dataset

This project uses the Clash Royale Games dataset from Kaggle:

https://www.kaggle.com/datasets/s1m0n38/clash-royale-games

Due to the large size of the dataset (~100GB extracted), only a subset of CSV files was used.

To reproduce the experiments:

1. Download the dataset from Kaggle
2. Extract the archive
3. Place a few CSV files (e.g., `20231002.csv`, `20231003.csv`) in:

data/raw/

## Run the Pipeline by Phase

### Phase 1 — Loading + Preprocessing

Run from the project root:

```powershell
python scr\load_data.py
python scr\preprocess_clash_royale_data.py
```

Expected Phase 1 outputs in `data/processed/`:

1. `clash_royale_clean.csv`
2. `clash_royale_clean.parquet`
3. `load_benchmark.csv`

---

Core Feature Engineering

Run from the project root:

```powershell
python scr\phase2_core_feature_engineering.py
```

Optional quick smoke test:

```powershell
python scr\phase2_core_feature_engineering.py --limit 5000
```

Expected Phase 2 outputs in `data/processed/`:

1. `card_list.csv`
2. `card_metadata.csv` (optional metadata scaffold)
3. `player_card_feature_matrix.parquet`
4. `opponent_card_feature_matrix.parquet`

---

## Full Run (End-to-End)

If you want to run everything in order:

```powershell
python scr\load_data.py
python scr\preprocess_clash_royale_data.py
python scr\phase2_core_feature_engineering.py
```

You can verify outputs with:

```powershell
Get-ChildItem -Path data/processed | Select-Object Name,Length
```
