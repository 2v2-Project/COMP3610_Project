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