from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from utils.metadata_utils import get_raw_cards, get_elixir_costs, get_card_types

INPUT_DEFAULT = Path("data/processed/clash_royale_clean.parquet")
OUTPUT_DIR_DEFAULT = Path("data/processed")

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

    return card_ids.with_columns(
        [
            pl.format("card_{}", pl.col("card_id")).alias("player_feature_column"),
            pl.format("opp_card_{}", pl.col("card_id")).alias("opponent_feature_column"),
        ]
    )


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


def build_card_metadata_from_api(card_list: pl.DataFrame, cards: list[dict]) -> pl.DataFrame:
    """
    Build card metadata table using already-fetched API data.
    Falls back to placeholder metadata if the API data is unavailable.
    """
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


def compute_deck_summary_features(
    data: pl.DataFrame,
    card_cols: list[str],
    elixir_costs: dict[int, int],
    card_types: dict[int, str],
    prefix: str,
) -> pl.DataFrame:
    """
    Compute deck-level summary features.

    Elixir features:
    - {prefix}_avg_elixir
    - {prefix}_low_cost_cards    (elixir <= 3)
    - {prefix}_medium_cost_cards (elixir == 4)
    - {prefix}_high_cost_cards   (elixir >= 5)

    Cycle-speed features:
    - {prefix}_cycle_cards       (elixir <= 2)
    - {prefix}_cycle_ratio       (cycle_cards / 8)

    Card type distribution features:
    - {prefix}_troop_count
    - {prefix}_spell_count
    - {prefix}_building_count
    """
    temp_elixir_columns = []
    temp_type_columns = []
    select_exprs = [pl.int_range(0, pl.len()).alias("match_id")]

    for col_name in card_cols:
        if elixir_costs:
            mapped_elixir = (
                pl.col(col_name)
                .replace_strict(elixir_costs, default=None)
                .cast(pl.Int32)
                .alias(f"_elixir_{col_name}")
            )
        else:
            mapped_elixir = pl.lit(None, dtype=pl.Int32).alias(f"_elixir_{col_name}")

        if card_types:
            mapped_type = (
                pl.col(col_name)
                .replace_strict(card_types, default=None)
                .cast(pl.Utf8)
                .alias(f"_type_{col_name}")
            )
        else:
            mapped_type = pl.lit(None, dtype=pl.Utf8).alias(f"_type_{col_name}")

        select_exprs.append(mapped_elixir)
        select_exprs.append(mapped_type)
        temp_elixir_columns.append(f"_elixir_{col_name}")
        temp_type_columns.append(f"_type_{col_name}")

    if elixir_costs:
        avg_elixir_expr = (
            pl.concat_list(temp_elixir_columns)
            .list.mean()
            .alias(f"{prefix}_avg_elixir")
        )

        low_cost_expr = (
            pl.sum_horizontal(
                [
                    pl.when(pl.col(temp_col) <= 3).then(1).otherwise(0)
                    for temp_col in temp_elixir_columns
                ]
            )
            .cast(pl.Int32)
            .alias(f"{prefix}_low_cost_cards")
        )

        medium_cost_expr = (
            pl.sum_horizontal(
                [
                    pl.when(pl.col(temp_col) == 4).then(1).otherwise(0)
                    for temp_col in temp_elixir_columns
                ]
            )
            .cast(pl.Int32)
            .alias(f"{prefix}_medium_cost_cards")
        )

        high_cost_expr = (
            pl.sum_horizontal(
                [
                    pl.when(pl.col(temp_col) >= 5).then(1).otherwise(0)
                    for temp_col in temp_elixir_columns
                ]
            )
            .cast(pl.Int32)
            .alias(f"{prefix}_high_cost_cards")
        )

        cycle_cards_expr = (
            pl.sum_horizontal(
                [
                    pl.when(pl.col(temp_col) <= 2).then(1).otherwise(0)
                    for temp_col in temp_elixir_columns
                ]
            )
            .cast(pl.Int32)
            .alias(f"{prefix}_cycle_cards")
        )

        cycle_ratio_expr = (
            pl.sum_horizontal(
                [
                    pl.when(pl.col(temp_col) <= 2).then(1).otherwise(0)
                    for temp_col in temp_elixir_columns
                ]
            ).cast(pl.Float32)
            / pl.lit(8.0)
        ).alias(f"{prefix}_cycle_ratio")
    else:
        avg_elixir_expr = pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_avg_elixir")
        low_cost_expr = pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_low_cost_cards")
        medium_cost_expr = pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_medium_cost_cards")
        high_cost_expr = pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_high_cost_cards")
        cycle_cards_expr = pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_cycle_cards")
        cycle_ratio_expr = pl.lit(None, dtype=pl.Float32).alias(f"{prefix}_cycle_ratio")

    if card_types:
        troop_count_expr = (
            pl.sum_horizontal(
                [
                    pl.when(pl.col(type_col) == "troop").then(1).otherwise(0)
                    for type_col in temp_type_columns
                ]
            )
            .cast(pl.Int32)
            .alias(f"{prefix}_troop_count")
        )

        spell_count_expr = (
            pl.sum_horizontal(
                [
                    pl.when(pl.col(type_col) == "spell").then(1).otherwise(0)
                    for type_col in temp_type_columns
                ]
            )
            .cast(pl.Int32)
            .alias(f"{prefix}_spell_count")
        )

        building_count_expr = (
            pl.sum_horizontal(
                [
                    pl.when(pl.col(type_col) == "building").then(1).otherwise(0)
                    for type_col in temp_type_columns
                ]
            )
            .cast(pl.Int32)
            .alias(f"{prefix}_building_count")
        )
    else:
        troop_count_expr = pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_troop_count")
        spell_count_expr = pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_spell_count")
        building_count_expr = pl.lit(None, dtype=pl.Int32).alias(f"{prefix}_building_count")

    return (
        data.select(select_exprs)
        .with_columns(
            [
                avg_elixir_expr,
                low_cost_expr,
                medium_cost_expr,
                high_cost_expr,
                cycle_cards_expr,
                cycle_ratio_expr,
                troop_count_expr,
                spell_count_expr,
                building_count_expr,
            ]
        )
        .select(
            [
                "match_id",
                f"{prefix}_avg_elixir",
                f"{prefix}_low_cost_cards",
                f"{prefix}_medium_cost_cards",
                f"{prefix}_high_cost_cards",
                f"{prefix}_cycle_cards",
                f"{prefix}_cycle_ratio",
                f"{prefix}_troop_count",
                f"{prefix}_spell_count",
                f"{prefix}_building_count",
            ]
        )
    )


def build_deck_summary_features(
    data: pl.DataFrame,
    elixir_costs: dict[int, int],
    card_types: dict[int, str],
) -> pl.DataFrame:
    """
    Build deck summary features for both player and opponent decks.
    """
    player_features = compute_deck_summary_features(
        data=data,
        card_cols=PLAYER_CARD_COLS,
        elixir_costs=elixir_costs,
        card_types=card_types,
        prefix="player",
    )

    opponent_features = compute_deck_summary_features(
        data=data,
        card_cols=OPPONENT_CARD_COLS,
        elixir_costs=elixir_costs,
        card_types=card_types,
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

    cards = get_raw_cards()
    card_metadata = build_card_metadata_from_api(card_list, cards)
    elixir_costs = get_elixir_costs()
    card_types = get_card_types()
    deck_summary_features = build_deck_summary_features(
        data=data,
        elixir_costs=elixir_costs,
        card_types=card_types,
    )

    card_list.write_csv(output_dir / "card_list.csv")
    player_matrix.write_parquet(output_dir / "player_card_feature_matrix.parquet")
    opponent_matrix.write_parquet(output_dir / "opponent_card_feature_matrix.parquet")
    card_metadata.write_csv(output_dir / "card_metadata.csv")
    deck_summary_features.write_parquet(output_dir / "deck_elixir_features.parquet")

    print(f"Rows processed: {data.height:,}")
    print(f"Unique cards identified: {len(card_ids)}")
    print(f"Card metadata rows: {card_metadata.height:,}")
    print(f"Elixir cost mappings fetched: {len(elixir_costs):,}")
    print(f"Card type mappings fetched: {len(card_types):,}")
    print(f"card_list saved: {output_dir / 'card_list.csv'}")
    print(f"player matrix saved: {output_dir / 'player_card_feature_matrix.parquet'}")
    print(f"opponent matrix saved: {output_dir / 'opponent_card_feature_matrix.parquet'}")
    print(f"card_metadata saved: {output_dir / 'card_metadata.csv'}")
    print(f"deck summary features saved: {output_dir / 'deck_elixir_features.parquet'}")


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