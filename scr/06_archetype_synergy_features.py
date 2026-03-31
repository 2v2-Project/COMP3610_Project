"""
Phase 4 — Archetypes + Synergy Feature Engineering
===================================================
Task  9: Implement archetype detection rules (Cycle, Beatdown, Siege, etc.)
Task 10: One-hot encode archetypes into ML-ready feature columns.
Task 11: Implement explicit synergy features (card-combo indicators).

Reads the cleaned parquet and the deck elixir features produced by
03_build_deck_feature_matrices.py.  Outputs archetype labels, one-hot
encoded archetype columns, and synergy indicator features for both the
player and the opponent deck.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

# ------------------------------------------------------------------
# Default paths
# ------------------------------------------------------------------
INPUT_CLEAN = Path("data/processed/clash_royale_clean.parquet")
INPUT_ELIXIR = Path("data/processed/deck_elixir_features.parquet")
OUTPUT_DIR = Path("data/processed")

PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
OPPONENT_CARD_COLS = [f"player2.card{i}" for i in range(1, 9)]

# ------------------------------------------------------------------
# Card-ID constants (from Royale API)
# ------------------------------------------------------------------
# Win conditions / archetype anchors
HOG_RIDER    = 26000021
GOLEM        = 26000009
GIANT        = 26000003
ROYAL_GIANT  = 26000024
XBOW         = 27000008
MORTAR       = 27000002
LAVA_HOUND   = 26000029
GRAVEYARD    = 28000010
GOBLIN_BARREL = 28000004
MINER        = 26000032
BALLOON      = 26000006
MEGA_KNIGHT  = 26000055
PEKKA        = 26000016  # Prince ID, P.E.K.K.A below
THREE_MUSK   = 26000028
ELIXIR_GOLEM = 26000067
ELECTRO_GIANT = 26000085
WALL_BREAKERS = 26000058
GOBLIN_DRILL = 27000013
SPARKY       = 26000033

# Support / synergy cards
ICE_SPIRIT   = 26000030
ICE_GOLEM    = 26000038
SKELETONS    = 26000010
NIGHT_WITCH  = 26000048
PRINCESS     = 26000026
GOBLIN_GANG  = 26000041
DARK_PRINCE  = 26000027
PRINCE       = 26000016
POISON       = 28000009
FREEZE       = 28000005
LIGHTNING    = 28000007
RAGE         = 28000002

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _has_card(card_cols: list[str], card_id: int) -> pl.Expr:
    """Return a boolean expression: True if any of *card_cols* equals *card_id*."""
    return pl.any_horizontal([pl.col(c) == card_id for c in card_cols])


def _has_all(card_cols: list[str], card_ids: list[int]) -> pl.Expr:
    """True when **every** card in *card_ids* appears somewhere in the deck."""
    return pl.all_horizontal([_has_card(card_cols, cid) for cid in card_ids])


def _has_any(card_cols: list[str], card_ids: list[int]) -> pl.Expr:
    """True when **at least one** card in *card_ids* appears in the deck."""
    return pl.any_horizontal([_has_card(card_cols, cid) for cid in card_ids])


# ==================================================================
# Task 9 — Archetype Detection Rules
# ==================================================================

BEATDOWN_CARDS = [GOLEM, GIANT, LAVA_HOUND, ELIXIR_GOLEM, ELECTRO_GIANT]
SIEGE_CARDS    = [XBOW, MORTAR]
BAIT_CARDS     = [GOBLIN_BARREL, PRINCESS, GOBLIN_GANG]
BRIDGE_SPAM_CARDS = [DARK_PRINCE, PRINCE, MEGA_KNIGHT, WALL_BREAKERS]

def assign_archetype(df: pl.DataFrame, card_cols: list[str],
                     avg_elixir_col: str, prefix: str) -> pl.DataFrame:
    """
    Classify each deck into one archetype using a rule cascade.

    Priority order (first match wins):
        1. Cycle      – avg_elixir ≤ 3.0 AND Hog Rider present
        2. Beatdown   – Golem / Giant / Lava Hound / Elixir Golem / E-Giant
        3. Siege       – X-Bow or Mortar present
        4. Bait        – Goblin Barrel + (Princess OR Goblin Gang)
        5. Bridge Spam – 2+ of Dark Prince, Prince, Mega Knight, Wall Breakers
        6. Graveyard   – Graveyard present
        7. Miner Control – Miner present AND avg_elixir ≤ 3.5
        8. Loon        – Balloon present (not already caught by Beatdown)
        9. Sparky      – Sparky present
       10. Control     – avg_elixir > 4.0  (heavy but no clear win-con above)
       11. Unknown     – everything else
    """
    col = f"{prefix}_archetype"

    # Count how many bridge-spam markers appear
    bridge_spam_count = pl.sum_horizontal(
        [_has_card(card_cols, cid).cast(pl.UInt8) for cid in BRIDGE_SPAM_CARDS]
    )

    bait_barrel = _has_card(card_cols, GOBLIN_BARREL)
    bait_support = _has_any(card_cols, [PRINCESS, GOBLIN_GANG])

    archetype_expr = (
        pl.when(
            (pl.col(avg_elixir_col) <= 3.0) & _has_card(card_cols, HOG_RIDER)
        ).then(pl.lit("Cycle"))
        .when(_has_any(card_cols, BEATDOWN_CARDS))
        .then(pl.lit("Beatdown"))
        .when(_has_any(card_cols, SIEGE_CARDS))
        .then(pl.lit("Siege"))
        .when(bait_barrel & bait_support)
        .then(pl.lit("Bait"))
        .when(bridge_spam_count >= 2)
        .then(pl.lit("Bridge Spam"))
        .when(_has_card(card_cols, GRAVEYARD))
        .then(pl.lit("Graveyard"))
        .when(
            _has_card(card_cols, MINER) & (pl.col(avg_elixir_col) <= 3.5)
        ).then(pl.lit("Miner Control"))
        .when(_has_card(card_cols, BALLOON))
        .then(pl.lit("Loon"))
        .when(_has_card(card_cols, SPARKY))
        .then(pl.lit("Sparky"))
        .when(pl.col(avg_elixir_col) > 4.0)
        .then(pl.lit("Control"))
        .otherwise(pl.lit("Unknown"))
        .alias(col)
    )

    return df.with_columns(archetype_expr)


# ==================================================================
# Task 10 — One-Hot Encode Archetypes
# ==================================================================

ARCHETYPE_LABELS = [
    "Cycle", "Beatdown", "Siege", "Bait", "Bridge Spam",
    "Graveyard", "Miner Control", "Loon", "Sparky", "Control", "Unknown",
]


def one_hot_archetypes(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """
    Create binary columns ``{prefix}_archetype_{label}`` for every label.
    """
    arch_col = f"{prefix}_archetype"
    ohe_exprs = [
        (pl.col(arch_col) == label)
        .cast(pl.UInt8)
        .alias(f"{prefix}_archetype_{label.lower().replace(' ', '_')}")
        for label in ARCHETYPE_LABELS
    ]
    return df.with_columns(ohe_exprs)


# ==================================================================
# Task 11 — Explicit Synergy Features
# ==================================================================

SYNERGY_COMBOS: list[tuple[str, list[int]]] = [
    ("hog_ice_spirit",        [HOG_RIDER, ICE_SPIRIT]),
    ("hog_ice_golem",         [HOG_RIDER, ICE_GOLEM]),
    ("golem_night_witch",     [GOLEM, NIGHT_WITCH]),
    ("golem_lightning",       [GOLEM, LIGHTNING]),
    ("lava_balloon",          [LAVA_HOUND, BALLOON]),
    ("lava_night_witch",      [LAVA_HOUND, NIGHT_WITCH]),
    ("graveyard_poison",      [GRAVEYARD, POISON]),
    ("graveyard_freeze",      [GRAVEYARD, FREEZE]),
    ("goblin_barrel_princess",[GOBLIN_BARREL, PRINCESS]),
    ("giant_graveyard",       [GIANT, GRAVEYARD]),
    ("balloon_freeze",        [BALLOON, FREEZE]),
    ("xbow_ice_spirit",       [XBOW, ICE_SPIRIT]),
    ("miner_wall_breakers",   [MINER, WALL_BREAKERS]),
    ("miner_poison",          [MINER, POISON]),
    ("mega_knight_balloon",   [MEGA_KNIGHT, BALLOON]),
    ("sparky_rage",           [SPARKY, RAGE]),
    ("electro_giant_lightning", [ELECTRO_GIANT, LIGHTNING]),
    ("dark_prince_prince",    [DARK_PRINCE, PRINCE]),
]


def build_synergy_features(df: pl.DataFrame, card_cols: list[str],
                           prefix: str) -> pl.DataFrame:
    """
    For each known synergy combo, add a binary column:
        ``{prefix}_syn_{combo_name}``  →  1 if all cards in combo are present.
    """
    syn_exprs = [
        _has_all(card_cols, ids)
        .cast(pl.UInt8)
        .alias(f"{prefix}_syn_{name}")
        for name, ids in SYNERGY_COMBOS
    ]
    return df.with_columns(syn_exprs)


# ==================================================================
# Pipeline
# ==================================================================

def run_phase4(input_clean: Path, input_elixir: Path,
               output_dir: Path, row_limit: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned match data
    if not input_clean.exists():
        raise FileNotFoundError(f"Missing {input_clean}. Run 02 first.")
    df = pl.read_parquet(input_clean)
    if row_limit:
        df = df.head(row_limit)
    print(f"Loaded {df.height:,} matches")

    # Load elixir features (produced by 03)
    if not input_elixir.exists():
        raise FileNotFoundError(f"Missing {input_elixir}. Run 03 first.")
    elixir_df = pl.read_parquet(input_elixir)
    if row_limit:
        elixir_df = elixir_df.head(row_limit)

    # Add match_id to main data for joining.
    # Cast to Int64 to match parquet-loaded join keys consistently.
    df = df.with_row_index("match_id").with_columns(
        pl.col("match_id").cast(pl.Int64)
    )

    # Normalize join key dtype on the right side as well.
    elixir_df = elixir_df.with_columns(pl.col("match_id").cast(pl.Int64))

    # Join avg_elixir columns needed by archetype rules
    df = df.join(
        elixir_df.select("match_id", "player_avg_elixir", "opp_avg_elixir"),
        on="match_id",
        how="left",
    )

    # ---- Task 9: Archetype detection ----
    df = assign_archetype(df, PLAYER_CARD_COLS, "player_avg_elixir", "player")
    df = assign_archetype(df, OPPONENT_CARD_COLS, "opp_avg_elixir", "opp")

    # Print distribution
    print("\n--- Player Archetype Distribution ---")
    dist = (
        df.group_by("player_archetype")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .with_columns((pl.col("count") / df.height * 100).round(2).alias("pct"))
    )
    print(dist)

    # ---- Task 10: One-hot encode archetypes ----
    df = one_hot_archetypes(df, "player")
    df = one_hot_archetypes(df, "opp")

    # ---- Task 11: Synergy features ----
    df = build_synergy_features(df, PLAYER_CARD_COLS, "player")
    df = build_synergy_features(df, OPPONENT_CARD_COLS, "opp")

    # Print synergy prevalence summary
    print("\n--- Player Synergy Prevalence ---")
    syn_cols = [c for c in df.columns if c.startswith("player_syn_")]
    syn_sums = df.select([pl.col(c).sum().alias(c) for c in syn_cols])
    for col_name in syn_cols:
        val = syn_sums[col_name][0]
        pct = val / df.height * 100
        print(f"  {col_name:<42s}  {val:>10,}  ({pct:.2f}%)")

    # ---- Select output columns ----
    arch_cols = (
        ["match_id", "player_archetype", "opp_archetype"]
        + [c for c in df.columns if "_archetype_" in c]
    )
    syn_out_cols = (
        ["match_id"]
        + [c for c in df.columns if "_syn_" in c]
    )

    arch_df = df.select(arch_cols)
    syn_df  = df.select(syn_out_cols)

    # Combined output for downstream phases
    all_phase4_cols = list(dict.fromkeys(
        arch_cols + [c for c in syn_out_cols if c != "match_id"]
    ))
    combined = df.select(all_phase4_cols)

    # ---- Save ----
    arch_df.write_parquet(output_dir / "archetype_features.parquet")
    syn_df.write_parquet(output_dir / "synergy_features.parquet")
    combined.write_parquet(output_dir / "archetype_synergy_features.parquet")

    # Also save a readable CSV sample
    combined.head(1000).write_csv(output_dir / "archetype_synergy_sample.csv")

    print(f"\nArchetype features saved  → {output_dir / 'archetype_features.parquet'}")
    print(f"Synergy features saved    → {output_dir / 'synergy_features.parquet'}")
    print(f"Combined output           → {output_dir / 'archetype_synergy_features.parquet'}")
    print(f"Sample CSV (1000 rows)    → {output_dir / 'archetype_synergy_sample.csv'}")

    # Summary stats
    print(f"\nTotal archetype columns:  {len([c for c in df.columns if '_archetype_' in c])}")
    print(f"Total synergy columns:    {len([c for c in df.columns if '_syn_' in c])}")
    print(f"Phase 4 feature columns:  {len(all_phase4_cols) - 1}")  # exclude match_id


# ==================================================================
# CLI entry point
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4 — Archetype detection + synergy features."
    )
    parser.add_argument("--input", type=Path, default=INPUT_CLEAN,
                        help="Cleaned parquet from 02")
    parser.add_argument("--elixir", type=Path, default=INPUT_ELIXIR,
                        help="Deck elixir features from 03")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional row limit for smoke testing")

    args = parser.parse_args()
    run_phase4(
        input_clean=args.input,
        input_elixir=args.elixir,
        output_dir=args.output_dir,
        row_limit=args.limit,
    )
