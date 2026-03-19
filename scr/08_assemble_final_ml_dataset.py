from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

INPUT_CLEAN = Path("data/processed/clash_royale_clean.parquet")
INPUT_PLAYER_MATRIX = Path("data/processed/player_card_feature_matrix.parquet")
INPUT_OPP_MATRIX = Path("data/processed/opponent_card_feature_matrix.parquet")
INPUT_DECK_SUMMARY = Path("data/processed/deck_elixir_features.parquet")
INPUT_ARCH_SYNERGY = Path("data/processed/archetype_synergy_features.parquet")
INPUT_MATCHUP = Path("data/processed/matchup_features.parquet")
OUTPUT_DIR = Path("data/processed")

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required {label}: {path}. "
            "Run prior phase scripts before Phase 6 assembly."
        )


def detect_target_column(clean_df: pl.DataFrame) -> str:
    if "target_win" in clean_df.columns:
        return "target_win"
    if "win_label" in clean_df.columns:
        return "win_label"
    raise ValueError(
        "Could not find target column. Expected one of: target_win, win_label"
    )


def ensure_match_id(df: pl.DataFrame) -> pl.DataFrame:
    if "match_id" in df.columns:
        return df
    return df.with_row_index("match_id")


def select_archetype_and_synergy_columns(df: pl.DataFrame) -> pl.DataFrame:
    keep_cols = ["match_id"] + [
        column
        for column in df.columns
        if ("_archetype_" in column) or ("_syn_" in column)
    ]
    return df.select(keep_cols)


def validate_unique_match_id(df: pl.DataFrame, label: str) -> None:
    if "match_id" not in df.columns:
        raise ValueError(f"{label} is missing required match_id column")

    total = df.height
    unique = df.select(pl.col("match_id").n_unique()).item()
    if total != unique:
        raise ValueError(
            f"{label} has duplicate match_id values (rows={total}, unique={unique})."
        )


def join_feature_tables(tables: list[pl.DataFrame]) -> pl.DataFrame:
    assembled = tables[0]
    for table in tables[1:]:
        assembled = assembled.join(table, on="match_id", how="inner")
    return assembled


def collect_duplicate_column_names(columns: list[str]) -> list[str]:
    seen = set()
    duplicates = []
    for col in columns:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)
    return duplicates


def ensure_numeric_training_frame(df: pl.DataFrame, target_col: str) -> tuple[pl.DataFrame, list[str]]:
    removed_non_numeric = []
    keep_cols = []

    for column, dtype in df.schema.items():
        if column == target_col:
            keep_cols.append(column)
            continue
        if dtype in NUMERIC_DTYPES:
            keep_cols.append(column)
        else:
            removed_non_numeric.append(column)

    numeric_df = df.select(keep_cols)

    feature_cols = [c for c in numeric_df.columns if c not in {"match_id", target_col}]
    if feature_cols:
        numeric_df = numeric_df.with_columns([pl.col(c).fill_null(0) for c in feature_cols])

    numeric_df = numeric_df.drop_nulls(subset=[target_col])

    return numeric_df, removed_non_numeric


def run_phase6(
    clean_path: Path,
    player_matrix_path: Path,
    opp_matrix_path: Path,
    deck_summary_path: Path,
    arch_synergy_path: Path,
    matchup_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_exists(clean_path, "clean parquet")
    ensure_exists(player_matrix_path, "player matrix")
    ensure_exists(opp_matrix_path, "opponent matrix")
    ensure_exists(deck_summary_path, "deck summary features")
    ensure_exists(arch_synergy_path, "archetype/synergy features")
    ensure_exists(matchup_path, "matchup features")

    clean_df = pl.read_parquet(clean_path)
    clean_df = ensure_match_id(clean_df)
    target_col = detect_target_column(clean_df)

    target_df = clean_df.select(["match_id", target_col])

    player_matrix = pl.read_parquet(player_matrix_path)
    opp_matrix = pl.read_parquet(opp_matrix_path)
    deck_summary = pl.read_parquet(deck_summary_path)
    arch_synergy = select_archetype_and_synergy_columns(pl.read_parquet(arch_synergy_path))
    matchup = pl.read_parquet(matchup_path)

    validate_unique_match_id(target_df, "target_df")
    validate_unique_match_id(player_matrix, "player_matrix")
    validate_unique_match_id(opp_matrix, "opp_matrix")
    validate_unique_match_id(deck_summary, "deck_summary")
    validate_unique_match_id(arch_synergy, "arch_synergy")
    validate_unique_match_id(matchup, "matchup")

    assembled = join_feature_tables(
        [target_df, player_matrix, opp_matrix, deck_summary, arch_synergy, matchup]
    )

    duplicate_columns = collect_duplicate_column_names(assembled.columns)

    missing_before = int(
        assembled.select(pl.all().null_count().sum()).to_series(0).item()
    )

    clean_training_df, removed_non_numeric = ensure_numeric_training_frame(
        assembled, target_col=target_col
    )

    missing_after = int(
        clean_training_df.select(pl.all().null_count().sum()).to_series(0).item()
    )

    clean_training_df = clean_training_df.unique(subset=["match_id"], keep="first")

    assembled.write_parquet(output_dir / "final_ml_dataset.parquet")
    clean_training_df.write_parquet(output_dir / "clean_training_dataset.parquet")

    quality_report = {
        "rows_final_ml_dataset": assembled.height,
        "rows_clean_training_dataset": clean_training_df.height,
        "columns_final_ml_dataset": assembled.width,
        "columns_clean_training_dataset": clean_training_df.width,
        "target_column": target_col,
        "missing_values_before_quality_handling": missing_before,
        "missing_values_after_quality_handling": missing_after,
        "duplicate_column_names": duplicate_columns,
        "duplicate_column_count": len(duplicate_columns),
        "removed_non_numeric_columns": removed_non_numeric,
        "removed_non_numeric_count": len(removed_non_numeric),
    }

    with (output_dir / "final_dataset_quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    print("Phase 6 complete.")
    print(f"Target column used: {target_col}")
    print(f"Assembled rows: {assembled.height:,}, columns: {assembled.width}")
    print(
        f"Clean training rows: {clean_training_df.height:,}, "
        f"columns: {clean_training_df.width}"
    )
    print(f"Missing values before: {missing_before:,}")
    print(f"Missing values after:  {missing_after:,}")
    print(f"Duplicate column names: {len(duplicate_columns)}")
    print(f"Removed non-numeric columns: {len(removed_non_numeric)}")
    print(f"Saved: {output_dir / 'final_ml_dataset.parquet'}")
    print(f"Saved: {output_dir / 'clean_training_dataset.parquet'}")
    print(f"Saved: {output_dir / 'final_dataset_quality_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 6 — assemble final ML dataset and run quality checks."
    )
    parser.add_argument("--clean", type=Path, default=INPUT_CLEAN)
    parser.add_argument("--player", type=Path, default=INPUT_PLAYER_MATRIX)
    parser.add_argument("--opponent", type=Path, default=INPUT_OPP_MATRIX)
    parser.add_argument("--deck-summary", type=Path, default=INPUT_DECK_SUMMARY)
    parser.add_argument("--arch-synergy", type=Path, default=INPUT_ARCH_SYNERGY)
    parser.add_argument("--matchup", type=Path, default=INPUT_MATCHUP)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    args = parser.parse_args()
    run_phase6(
        clean_path=args.clean,
        player_matrix_path=args.player,
        opp_matrix_path=args.opponent,
        deck_summary_path=args.deck_summary,
        arch_synergy_path=args.arch_synergy,
        matchup_path=args.matchup,
        output_dir=args.output_dir,
    )
