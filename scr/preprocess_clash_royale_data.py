import glob
import time
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl

RAW_GLOB = "data/raw/*.csv"
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLS = [
    "datetime",
    "gamemode",
    "player1.tag",
    "player1.trophies",
    "player1.crowns",
    "player1.card1",
    "player1.card2",
    "player1.card3",
    "player1.card4",
    "player1.card5",
    "player1.card6",
    "player1.card7",
    "player1.card8",
    "player2.tag",
    "player2.trophies",
    "player2.crowns",
    "player2.card1",
    "player2.card2",
    "player2.card3",
    "player2.card4",
    "player2.card5",
    "player2.card6",
    "player2.card7",
    "player2.card8",
]

CARD1 = [f"player1.card{i}" for i in range(1, 9)]
CARD2 = [f"player2.card{i}" for i in range(1, 9)]
STRING_COLS = ["datetime", "player1.tag", "player2.tag"]
NULL_LIKE = {"", "null", "none", "nan", "na"}


def measure_time(fn):
    start = time.time()
    result = fn()
    end = time.time()
    return result, end - start


def main():
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {RAW_GLOB}")

    # Task 1: Load and benchmark Pandas vs Polars
    df_pd, pandas_time = measure_time(
        lambda: pd.concat(
            [pd.read_csv(f, header=None, names=COLS) for f in files],
            ignore_index=True,
        )
    )

    df_pl, polars_time = measure_time(
        lambda: pl.concat(
            [pl.read_csv(f, has_header=False, new_columns=COLS) for f in files],
            how="vertical",
        )
    )

    print(f"Pandas load time: {pandas_time:.2f}s")
    print(f"Polars load time: {polars_time:.2f}s")
    print(f"Rows: {df_pl.height:,}  Cols: {df_pl.width}")

    # Task 2: Cleaning + feature engineering in Polars
    cleaned = (
        df_pl
        .with_columns([pl.col(c).cast(pl.Utf8).str.strip_chars() for c in STRING_COLS])
        .with_columns([
            pl.when(pl.col(c).str.to_lowercase().is_in(NULL_LIKE))
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in STRING_COLS
        ])
        .with_columns([
            pl.col("datetime").str.strptime(
                pl.Datetime, format="%Y%m%dT%H%M%S%.3fZ", strict=False
            ),
            pl.col("gamemode").cast(pl.Int64, strict=False),
            pl.col("player1.trophies").cast(pl.Int64, strict=False),
            pl.col("player2.trophies").cast(pl.Int64, strict=False),
            pl.col("player1.crowns").cast(pl.Int64, strict=False),
            pl.col("player2.crowns").cast(pl.Int64, strict=False),
            *[pl.col(c).cast(pl.Int64, strict=False) for c in CARD1 + CARD2],
        ])
        .drop_nulls()
        .filter(pl.col("player1.crowns").is_between(0, 3))
        .filter(pl.col("player2.crowns").is_between(0, 3))
        .filter(pl.col("player1.trophies") >= 0)
        .filter(pl.col("player2.trophies") >= 0)
        .filter(pl.col("player1.tag") != pl.col("player2.tag"))
        .with_columns([
            pl.concat_list(CARD1).list.n_unique().alias("p1_unique_cards"),
            pl.concat_list(CARD2).list.n_unique().alias("p2_unique_cards"),
        ])
        .filter(pl.col("p1_unique_cards") == 8)
        .filter(pl.col("p2_unique_cards") == 8)
        .unique()
        .unique(subset=["datetime", "player1.tag", "player2.tag", "gamemode"], keep="first")
        .with_columns([
            (pl.col("player1.crowns") > pl.col("player2.crowns"))
            .cast(pl.Int8)
            .alias("win_label"),
            (pl.col("player1.crowns") - pl.col("player2.crowns")).alias("crown_diff"),
            (pl.col("player1.trophies") - pl.col("player2.trophies")).alias("trophy_diff"),
        ])
        .drop(["p1_unique_cards", "p2_unique_cards"])
    )

    print(f"Cleaned rows: {cleaned.height:,}")

    # Task 3: DuckDB validation
    con = duckdb.connect()
    con.register("clean_df", cleaned.to_pandas())

    check = con.execute(
        """
        SELECT
          COUNT(*) AS rows_after_cleaning,
          AVG(win_label) AS player1_win_rate,
          AVG(trophy_diff) AS avg_trophy_diff
        FROM clean_df
        """
    ).fetchdf()

    print(check)

    # Task 4: Export outputs
    cleaned.write_csv(OUT_DIR / "clash_royale_clean.csv")
    cleaned.write_parquet(OUT_DIR / "clash_royale_clean.parquet")

    # Optional: keep a simple benchmark summary for your report
    benchmark = pd.DataFrame(
        {
            "loader": ["pandas", "polars"],
            "seconds": [pandas_time, polars_time],
            "rows": [len(df_pd), cleaned.height],
        }
    )
    benchmark.to_csv(OUT_DIR / "load_benchmark.csv", index=False)

    print("Saved cleaned files to data/processed/")


if __name__ == "__main__":
    main()
