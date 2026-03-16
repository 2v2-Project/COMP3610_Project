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