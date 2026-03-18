from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

INPUT_DEFAULT = Path("data/processed/clash_royale_clean.parquet")
OUTPUT_DIR_DEFAULT = Path("data/processed")

PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
OPPONENT_CARD_COLS = [f"player2.card{i}" for i in range(1, 9)]


def load_clean_data(input_path: Path, row_limit: int | None) -> pl.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Run preprocess_clash_royale_data.py first."
        )

    data = pl.read_parquet(input_path)

    missing_columns = [
        column
        for column in PLAYER_CARD_COLS + OPPONENT_CARD_COLS
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Input dataset is missing expected deck columns: "
            + ", ".join(missing_columns)
        )

    if row_limit is not None and row_limit > 0:
        data = data.head(row_limit)

    return data


def build_card_list(data: pl.DataFrame) -> pl.DataFrame:
    all_card_columns = PLAYER_CARD_COLS + OPPONENT_CARD_COLS

    card_ids = (
        data.select(all_card_columns)
        .unpivot(on=all_card_columns)
        .drop_nulls("value")
        .select(pl.col("value").cast(pl.Int64).alias("card_id"))
        .unique()
        .sort("card_id")
    )

    card_list = card_ids.with_columns(
        [
            pl.format("card_{}", pl.col("card_id")).alias("player_feature_column"),
            pl.format("opp_card_{}", pl.col("card_id")).alias("opponent_feature_column"),
        ]
    )

    return card_list


def build_player_matrix(data: pl.DataFrame, card_ids: list[int]) -> pl.DataFrame:
    player_features = [pl.int_range(0, pl.len()).alias("match_id")]

    for card_id in card_ids:
        player_features.append(
            pl.any_horizontal([pl.col(column) == card_id for column in PLAYER_CARD_COLS])
            .cast(pl.UInt8)
            .alias(f"card_{card_id}")
        )

    return data.select(player_features)


def build_opponent_matrix(data: pl.DataFrame, card_ids: list[int]) -> pl.DataFrame:
    opponent_features = [pl.int_range(0, pl.len()).alias("match_id")]

    for card_id in card_ids:
        opponent_features.append(
            pl.any_horizontal([pl.col(column) == card_id for column in OPPONENT_CARD_COLS])
            .cast(pl.UInt8)
            .alias(f"opp_card_{card_id}")
        )

    return data.select(opponent_features)


def build_optional_card_metadata(card_list: pl.DataFrame) -> pl.DataFrame:
    return card_list.select("card_id").with_columns(
        [
            pl.lit(None, dtype=pl.Utf8).alias("card_name"),
            pl.lit(None, dtype=pl.Int64).alias("elixir_cost"),
            pl.lit(None, dtype=pl.Utf8).alias("card_type"),
            pl.lit("metadata_not_available_in_source").alias("metadata_status"),
        ]
    )


def run_phase2(input_path: Path, output_dir: Path, row_limit: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_clean_data(input_path=input_path, row_limit=row_limit)

    card_list = build_card_list(data)
    card_ids = card_list.get_column("card_id").to_list()

    player_matrix = build_player_matrix(data=data, card_ids=card_ids)
    opponent_matrix = build_opponent_matrix(data=data, card_ids=card_ids)
    card_metadata = build_optional_card_metadata(card_list)

    card_list.write_csv(output_dir / "card_list.csv")
    player_matrix.write_parquet(output_dir / "player_card_feature_matrix.parquet")
    opponent_matrix.write_parquet(output_dir / "opponent_card_feature_matrix.parquet")
    card_metadata.write_csv(output_dir / "card_metadata.csv")

    print(f"Rows processed: {data.height:,}")
    print(f"Unique cards identified: {len(card_ids)}")
    print(f"card_list saved: {output_dir / 'card_list.csv'}")
    print(
        "player matrix saved: "
        f"{output_dir / 'player_card_feature_matrix.parquet'}"
    )
    print(
        "opponent matrix saved: "
        f"{output_dir / 'opponent_card_feature_matrix.parquet'}"
    )
    print(f"optional card_metadata saved: {output_dir / 'card_metadata.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 core feature engineering for Clash Royale deck cards."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_DEFAULT,
        help="Path to cleaned parquet produced by preprocess_clash_royale_data.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Directory where Phase 2 outputs will be written",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke testing",
    )

    arguments = parser.parse_args()
    run_phase2(
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        row_limit=arguments.limit,
    )
