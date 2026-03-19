"""
Phase 5 — Matchup Features
===========================
Task 12: Add opponent elixir features  (opp_avg_elixir, opp_cycle_cards).
Task 13: Compute deck difference features (elixir_difference, cycle_difference,
         troop_diff, spell_diff, building_diff).

Reads the deck elixir features produced by 03_build_deck_feature_matrices.py
and outputs the matchup feature table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

# ------------------------------------------------------------------
# Default paths
# ------------------------------------------------------------------
INPUT_ELIXIR = Path("data/processed/deck_elixir_features.parquet")
OUTPUT_DIR = Path("data/processed")


# ==================================================================
# Task 12 — Opponent Elixir Features
# ==================================================================

def extract_opponent_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Select the opponent deck features already computed in Phase 2/3:
        - opp_avg_elixir
        - opp_cycle_cards
        - opp_cycle_ratio
        - opp_low_cost_cards, opp_medium_cost_cards, opp_high_cost_cards
        - opp_troop_count, opp_spell_count, opp_building_count
    """
    opp_cols = [c for c in df.columns if c.startswith("opp_")]
    return df.select(["match_id"] + opp_cols)


# ==================================================================
# Task 13 — Deck Difference (Matchup) Features
# ==================================================================

def compute_matchup_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute player-minus-opponent differences for key deck statistics.

    Features created:
        - elixir_difference     = player_avg_elixir   - opp_avg_elixir
        - cycle_difference      = player_cycle_cards   - opp_cycle_cards
        - cycle_ratio_diff      = player_cycle_ratio   - opp_cycle_ratio
        - low_cost_diff         = player_low_cost_cards  - opp_low_cost_cards
        - medium_cost_diff      = player_medium_cost_cards - opp_medium_cost_cards
        - high_cost_diff        = player_high_cost_cards - opp_high_cost_cards
        - troop_count_diff      = player_troop_count   - opp_troop_count
        - spell_count_diff      = player_spell_count   - opp_spell_count
        - building_count_diff   = player_building_count - opp_building_count
    """
    diff_exprs = [
        (pl.col("player_avg_elixir") - pl.col("opp_avg_elixir"))
            .alias("elixir_difference"),
        (pl.col("player_cycle_cards") - pl.col("opp_cycle_cards"))
            .cast(pl.Int32)
            .alias("cycle_difference"),
        (pl.col("player_cycle_ratio") - pl.col("opp_cycle_ratio"))
            .alias("cycle_ratio_diff"),
        (pl.col("player_low_cost_cards") - pl.col("opp_low_cost_cards"))
            .cast(pl.Int32)
            .alias("low_cost_diff"),
        (pl.col("player_medium_cost_cards") - pl.col("opp_medium_cost_cards"))
            .cast(pl.Int32)
            .alias("medium_cost_diff"),
        (pl.col("player_high_cost_cards") - pl.col("opp_high_cost_cards"))
            .cast(pl.Int32)
            .alias("high_cost_diff"),
        (pl.col("player_troop_count") - pl.col("opp_troop_count"))
            .cast(pl.Int32)
            .alias("troop_count_diff"),
        (pl.col("player_spell_count") - pl.col("opp_spell_count"))
            .cast(pl.Int32)
            .alias("spell_count_diff"),
        (pl.col("player_building_count") - pl.col("opp_building_count"))
            .cast(pl.Int32)
            .alias("building_count_diff"),
    ]

    return df.with_columns(diff_exprs)


# ==================================================================
# Pipeline
# ==================================================================

def run_phase5(input_elixir: Path, output_dir: Path,
               row_limit: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_elixir.exists():
        raise FileNotFoundError(
            f"Missing {input_elixir}. Run 03_build_deck_feature_matrices.py first."
        )

    df = pl.read_parquet(input_elixir)
    if row_limit:
        df = df.head(row_limit)
    print(f"Loaded deck features: {df.height:,} rows, {df.width} columns")

    # ---- Task 12: Opponent elixir features ----
    opp_features = extract_opponent_features(df)
    print(f"\nTask 12 — Opponent feature columns: {opp_features.width - 1}")
    print("  Columns:", [c for c in opp_features.columns if c != "match_id"])

    # ---- Task 13: Matchup difference features ----
    df = compute_matchup_features(df)

    matchup_cols = [
        "elixir_difference", "cycle_difference", "cycle_ratio_diff",
        "low_cost_diff", "medium_cost_diff", "high_cost_diff",
        "troop_count_diff", "spell_count_diff", "building_count_diff",
    ]

    print(f"\nTask 13 — Matchup difference columns: {len(matchup_cols)}")
    print("  Columns:", matchup_cols)

    # Print summary statistics for the matchup features
    print("\n--- Matchup Feature Summary Statistics ---")
    summary = df.select(matchup_cols).describe()
    print(summary)

    # ---- Select and save ----
    # Opponent features (standalone)
    opp_features.write_parquet(output_dir / "opponent_elixir_features.parquet")

    # Matchup difference features
    matchup_df = df.select(["match_id"] + matchup_cols)
    matchup_df.write_parquet(output_dir / "matchup_features.parquet")

    # Combined Phase 5 output (opponent + differences)
    all_phase5_cols = (
        ["match_id"]
        + [c for c in opp_features.columns if c != "match_id"]
        + matchup_cols
    )
    combined = df.select(all_phase5_cols)
    combined.write_parquet(output_dir / "matchup_deck_diff_features.parquet")

    print(f"\nOpponent features saved   → {output_dir / 'opponent_elixir_features.parquet'}")
    print(f"Matchup features saved    → {output_dir / 'matchup_features.parquet'}")
    print(f"Combined output           → {output_dir / 'matchup_deck_diff_features.parquet'}")
    print(f"\nTotal Phase 5 columns:    {len(all_phase5_cols) - 1}")


# ==================================================================
# CLI
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5 — Matchup features (opponent elixir + differences)."
    )
    parser.add_argument("--elixir", type=Path, default=INPUT_ELIXIR,
                        help="Deck elixir features parquet from 03")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional row limit for smoke testing")

    args = parser.parse_args()
    run_phase5(
        input_elixir=args.elixir,
        output_dir=args.output_dir,
        row_limit=args.limit,
    )
