from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import requests

INPUT_DEFAULT = Path("data/processed/clash_royale_clean.parquet")
OUTPUT_DIR_DEFAULT = Path("data/processed")
CARD_API_URL = "https://royaleapi.github.io/cr-api-data/json/cards.json"

PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
OPPONENT_CARD_COLS = [f"player2.card{i}" for i in range(1, 9)]


def load_clean_data(input_path: Path, row_limit: int | None) -> pl.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Run 02_preprocess_clash_royale_data.py first."
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


def fetch_card_metadata() -> list[dict]:
    """
    Fetch Clash Royale card metadata from the Royale API.
    Returns a list of card metadata dicts.
    If the API is unavailable, returns an empty list.
    """
    try:
        response = requests.get(CARD_API_URL, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        print(f"Could not fetch card metadata: {error}")
        return []


def fetch_card_elixir_costs() -> dict[int, int]:
    """
    Extract a mapping of card_id -> elixir_cost from API metadata.
    """
    cards = fetch_card_metadata()

    elixir_costs: dict[int, int] = {}
    for card in cards:
        card_id = card.get("id")
        elixir = card.get("elixir")

        if card_id is None or elixir is None:
            continue

        try:
            elixir_costs[int(card_id)] = int(elixir)
        except (TypeError, ValueError):
            continue

    return elixir_costs


def build_card_metadata_from_api(card_list: pl.DataFrame) -> pl.DataFrame:
    """
    Build card metadata table using API data when available.
    Falls back to placeholder metadata if the API is unavailable.
    """
    cards = fetch_card_metadata()
    if not cards:
        return build_optional_card_metadata(card_list)

    metadata_rows = []
    for card in cards:
        card_id = card.get("id")
        if card_id is None:
            continue

        try:
            metadata_rows.append(
                {
                    "card_id": int(card_id),
                    "card_name": card.get("name"),
                    "elixir_cost": (
                        int(card["elixir"]) if card.get("elixir") is not None else None
                    ),
                    "card_type": card.get("type"),
                    "metadata_status": "fetched_from_api",
                }
            )
        except (TypeError, ValueError):
            continue

    if not metadata_rows:
        return build_optional_card_metadata(card_list)

    api_metadata = pl.DataFrame(metadata_rows)

    return (
        card_list.select("card_id")
        .join(api_metadata, on="card_id", how="left")
        .with_columns(
            pl.when(pl.col("metadata_status").is_null())
            .then(pl.lit("missing_for_card_id"))
            .otherwise(pl.col("metadata_status"))
            .alias("metadata_status")
        )
    )


def compute_elixir_features(
    data: pl.DataFrame,
    card_cols: list[str],
    elixir_costs: dict[int, int],
    prefix: str,
) -> pl.DataFrame:
    """
    Compute deck-level elixir features for a given set of card columns.

    Features:
    - {prefix}_avg_elixir
    - {prefix}_low_cost_cards   (elixir <= 3)
    - {prefix}_medium_cost_cards (elixir == 4)
    - {prefix}_high_cost_cards  (elixir >= 5)
    """
    base = data.select(pl.int_range(0, pl.len()).alias("match_id"))

    if not elixir_costs:
        return base.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_avg_elixir"),
                pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_low_cost_cards"),
                pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_medium_cost_cards"),
                pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_high_cost_cards"),
            ]
        )

    temp_columns = []
    select_exprs = [pl.int_range(0, pl.len()).alias("match_id")]

    for col_name in card_cols:
        mapped = (
            pl.col(col_name)
            .replace_strict(elixir_costs, default=None)
            .cast(pl.Int32)
            .alias(f"_elixir_{col_name}")
        )
        select_exprs.append(mapped)
        temp_columns.append(f"_elixir_{col_name}")

    features = (
        data.select(select_exprs)
        .with_columns(
            [
                pl.concat_list(temp_columns).list.mean().alias(f"{prefix}_avg_elixir"),
                pl.sum_horizontal(
                    [
                        pl.when(pl.col(temp_col) <= 3).then(1).otherwise(0)
                        for temp_col in temp_columns
                    ]
                ).alias(f"{prefix}_low_cost_cards"),
                pl.sum_horizontal(
                    [
                        pl.when(pl.col(temp_col) == 4).then(1).otherwise(0)
                        for temp_col in temp_columns
                    ]
                ).alias(f"{prefix}_medium_cost_cards"),
                pl.sum_horizontal(
                    [
                        pl.when(pl.col(temp_col) >= 5).then(1).otherwise(0)
                        for temp_col in temp_columns
                    ]
                ).alias(f"{prefix}_high_cost_cards"),
            ]
        )
        .select(
            [
                "match_id",
                f"{prefix}_avg_elixir",
                f"{prefix}_low_cost_cards",
                f"{prefix}_medium_cost_cards",
                f"{prefix}_high_cost_cards",
            ]
        )
    )

    return features


def build_elixir_features(
    data: pl.DataFrame,
    elixir_costs: dict[int, int],
) -> pl.DataFrame:
    """
    Build deck elixir features for both player and opponent decks.
    """
    player_features = compute_elixir_features(
        data=data,
        card_cols=PLAYER_CARD_COLS,
        elixir_costs=elixir_costs,
        prefix="player",
    )

    opponent_features = compute_elixir_features(
        data=data,
        card_cols=OPPONENT_CARD_COLS,
        elixir_costs=elixir_costs,
        prefix="opp",
    )

    return player_features.join(opponent_features, on="match_id", how="inner")


def run_phase2(input_path: Path, output_dir: Path, row_limit: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_clean_data(input_path=input_path, row_limit=row_limit)

    card_list = build_card_list(data)
    card_ids = card_list.get_column("card_id").to_list()

    player_matrix = build_player_matrix(data=data, card_ids=card_ids)
    opponent_matrix = build_opponent_matrix(data=data, card_ids=card_ids)
    card_metadata = build_card_metadata_from_api(card_list)

    elixir_costs = fetch_card_elixir_costs()
    elixir_features = build_elixir_features(data=data, elixir_costs=elixir_costs)

    card_list.write_csv(output_dir / "card_list.csv")
    player_matrix.write_parquet(output_dir / "player_card_feature_matrix.parquet")
    opponent_matrix.write_parquet(output_dir / "opponent_card_feature_matrix.parquet")
    card_metadata.write_csv(output_dir / "card_metadata.csv")
    elixir_features.write_parquet(output_dir / "deck_elixir_features.parquet")

    print(f"Rows processed: {data.height:,}")
    print(f"Unique cards identified: {len(card_ids)}")
    print(f"Card metadata rows: {card_metadata.height:,}")
    print(f"Elixir cost mappings fetched: {len(elixir_costs):,}")
    print(f"card_list saved: {output_dir / 'card_list.csv'}")
    print(f"player matrix saved: {output_dir / 'player_card_feature_matrix.parquet'}")
    print(f"opponent matrix saved: {output_dir / 'opponent_card_feature_matrix.parquet'}")
    print(f"card_metadata saved: {output_dir / 'card_metadata.csv'}")
    print(f"deck_elixir_features saved: {output_dir / 'deck_elixir_features.parquet'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 core feature engineering for Clash Royale deck cards."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_DEFAULT,
        help="Path to cleaned parquet produced by 02_preprocess_clash_royale_data.py",
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