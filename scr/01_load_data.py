import glob

import pandas as pd
import polars as pl

RAW_GLOB = "data/raw/*.csv"

COLS = [
    "datetime", "gamemode",
    "player1.tag", "player1.trophies", "player1.crowns",
    "player1.card1", "player1.card2", "player1.card3", "player1.card4",
    "player1.card5", "player1.card6", "player1.card7", "player1.card8",
    "player2.tag", "player2.trophies", "player2.crowns",
    "player2.card1", "player2.card2", "player2.card3", "player2.card4",
    "player2.card5", "player2.card6", "player2.card7", "player2.card8",
]


def load_pandas() -> pd.DataFrame:
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {RAW_GLOB}")
    return pd.concat(
        [pd.read_csv(f, header=None, names=COLS) for f in files],
        ignore_index=True,
    )


def load_polars() -> pl.DataFrame:
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {RAW_GLOB}")
    return pl.concat(
        [pl.read_csv(f, has_header=False, new_columns=COLS) for f in files],
        how="vertical",
    )


if __name__ == "__main__":
    df = load_pandas()
    print("Data loaded successfully!")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nShape: {df.shape}")
    print(df.head())