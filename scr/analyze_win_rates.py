"""
Task 4 – Calculate and visualise win rates for cards and decks.

Uses the cleaned dataset produced by preprocess_clash_royale_data.py,
or falls back to cleaning the raw CSVs directly.
Card IDs are mapped to human-readable names via the Clash Royale API.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import requests
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_polars, COLS

# ----------------------------
# Paths
# ----------------------------
PROCESSED_PARQUET = Path("data/processed/clash_royale_clean.parquet")
PROCESSED_CSV = Path("data/processed/clash_royale_clean.csv")

OUT_DIR = Path("data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Column helpers
# ----------------------------
P1_CARDS = [f"player1.card{i}" for i in range(1, 9)]
P2_CARDS = [f"player2.card{i}" for i in range(1, 9)]

CARD_API = "https://royaleapi.github.io/cr-api-data/json/cards.json"

# ----------------------------
# Load / clean helpers
# ----------------------------
NULL_LIKE = {"", "null", "none", "nan", "na"}
STRING_COLS = ["datetime", "player1.tag", "player2.tag"]


def load_clean_data() -> pl.DataFrame:
    """Load the cleaned dataset, or clean from raw if not available."""
    if PROCESSED_PARQUET.exists():
        return pl.read_parquet(PROCESSED_PARQUET)
    if PROCESSED_CSV.exists():
        return pl.read_csv(PROCESSED_CSV)

    # Fall back: load raw and apply the same cleaning pipeline
    print("Cleaned data not found – loading and cleaning raw CSVs …")
    df = load_polars()

    df = (
        df
        .with_columns([pl.col(c).cast(pl.Utf8).str.strip_chars() for c in STRING_COLS])
        .with_columns([
            pl.when(pl.col(c).str.to_lowercase().is_in(NULL_LIKE))
            .then(None).otherwise(pl.col(c)).alias(c)
            for c in STRING_COLS
        ])
        .with_columns([
            pl.col("datetime").str.strptime(pl.Datetime, format="%Y%m%dT%H%M%S%.3fZ", strict=False),
            pl.col("gamemode").cast(pl.Int64, strict=False),
            pl.col("player1.trophies").cast(pl.Int64, strict=False),
            pl.col("player2.trophies").cast(pl.Int64, strict=False),
            pl.col("player1.crowns").cast(pl.Int64, strict=False),
            pl.col("player2.crowns").cast(pl.Int64, strict=False),
            *[pl.col(c).cast(pl.Int64, strict=False) for c in P1_CARDS + P2_CARDS],
        ])
        .drop_nulls()
        .filter(pl.col("player1.crowns").is_between(0, 3))
        .filter(pl.col("player2.crowns").is_between(0, 3))
        .filter(pl.col("player1.trophies") >= 0)
        .filter(pl.col("player2.trophies") >= 0)
        .filter(pl.col("player1.tag") != pl.col("player2.tag"))
        .with_columns([
            pl.concat_list(P1_CARDS).list.n_unique().alias("p1_unique_cards"),
            pl.concat_list(P2_CARDS).list.n_unique().alias("p2_unique_cards"),
        ])
        .filter(pl.col("p1_unique_cards") == 8)
        .filter(pl.col("p2_unique_cards") == 8)
        .unique()
        .unique(subset=["datetime", "player1.tag", "player2.tag", "gamemode"], keep="first")
        .with_columns([
            (pl.col("player1.crowns") > pl.col("player2.crowns")).cast(pl.Int8).alias("win_label"),
            (pl.col("player1.crowns") - pl.col("player2.crowns")).alias("crown_diff"),
            (pl.col("player1.trophies") - pl.col("player2.trophies")).alias("trophy_diff"),
        ])
        .drop(["p1_unique_cards", "p2_unique_cards"])
    )
    return df


def fetch_card_names() -> dict[int, str]:
    """Download card metadata and return {card_id: card_name}."""
    try:
        resp = requests.get(CARD_API, timeout=20)
        resp.raise_for_status()
        return {int(c["id"]): c["name"] for c in resp.json()}
    except Exception as e:
        print(f"Could not fetch card names: {e}")
        return {}


# ----------------------------
# Win‑rate calculations
# ----------------------------

def card_win_rates(df: pl.DataFrame) -> pl.DataFrame:
    """
    For every card, compute:
        – times_played  (appearances across all matches, both sides)
        – wins           (appearances on the winning side)
        – win_rate        wins / times_played
    Player 1 wins when win_label == 1; player 2 wins when win_label == 0.
    """
    # Player 1 cards
    p1 = (
        df.select(P1_CARDS + ["win_label"])
        .unpivot(index="win_label", variable_name="slot", value_name="card")
        .drop("slot")
        .with_columns(pl.col("win_label").alias("won"))
    )

    # Player 2 cards – win when win_label == 0
    p2 = (
        df.select(P2_CARDS + ["win_label"])
        .unpivot(index="win_label", variable_name="slot", value_name="card")
        .drop("slot")
        .with_columns((1 - pl.col("win_label")).cast(pl.Int8).alias("won"))
    )

    combined = pl.concat([p1.select("card", "won"), p2.select("card", "won")])

    rates = (
        combined
        .group_by("card")
        .agg([
            pl.len().alias("times_played"),
            pl.col("won").sum().alias("wins"),
        ])
        .with_columns(
            (pl.col("wins") / pl.col("times_played") * 100).alias("win_rate")
        )
        .sort("win_rate", descending=True)
    )
    return rates


def deck_win_rates(df: pl.DataFrame, min_appearances: int = 5) -> pl.DataFrame:
    """
    Compute win rates for full 8‑card decks.
    A deck is identified by sorting its 8 card IDs.
    Only decks appearing >= min_appearances are kept.
    """
    # Build a canonical deck key for each side
    p1_deck = (
        df.with_columns(
            pl.concat_list(P1_CARDS).list.sort().list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("deck")
        )
        .select("deck", "win_label")
        .rename({"win_label": "won"})
    )

    p2_deck = (
        df.with_columns(
            pl.concat_list(P2_CARDS).list.sort().list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("deck")
        )
        .select("deck", "win_label")
        .with_columns((1 - pl.col("win_label")).cast(pl.Int8).alias("won"))
        .drop("win_label")
    )

    combined = pl.concat([p1_deck, p2_deck])

    rates = (
        combined
        .group_by("deck")
        .agg([
            pl.len().alias("times_played"),
            pl.col("won").sum().alias("wins"),
        ])
        .filter(pl.col("times_played") >= min_appearances)
        .with_columns(
            (pl.col("wins") / pl.col("times_played") * 100).alias("win_rate")
        )
        .sort("win_rate", descending=True)
    )
    return rates


# ----------------------------
# Visualisations
# ----------------------------

def resolve_name(card_id, card_map: dict[int, str]) -> str:
    try:
        return card_map.get(int(card_id), str(card_id))
    except (ValueError, TypeError):
        return str(card_id)


def plot_card_win_rates(rates: pl.DataFrame, card_map: dict[int, str],
                        top_n: int = 15, min_games: int = 50):
    """Bar chart: top‑N and bottom‑N cards by win rate (min sample filter)."""
    filtered = rates.filter(pl.col("times_played") >= min_games)

    top = filtered.head(top_n)
    bottom = filtered.tail(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Highest win‑rate cards ---
    names = [resolve_name(c, card_map) for c in top["card"].to_list()][::-1]
    wr = top["win_rate"].to_list()[::-1]
    played = top["times_played"].to_list()[::-1]

    bars = axes[0].barh(names, wr, color=sns.color_palette("Greens_d", len(names)))
    axes[0].set_title(f"Top {top_n} Cards by Win Rate")
    axes[0].set_xlabel("Win Rate (%)")
    for bar, w, p in zip(bars, wr, played):
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{w:.1f}%  (n={p:,})", va="center", fontsize=8)

    # --- Lowest win‑rate cards ---
    names_b = [resolve_name(c, card_map) for c in bottom["card"].to_list()][::-1]
    wr_b = bottom["win_rate"].to_list()[::-1]
    played_b = bottom["times_played"].to_list()[::-1]

    bars_b = axes[1].barh(names_b, wr_b, color=sns.color_palette("Reds_d", len(names_b)))
    axes[1].set_title(f"Bottom {top_n} Cards by Win Rate")
    axes[1].set_xlabel("Win Rate (%)")
    for bar, w, p in zip(bars_b, wr_b, played_b):
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{w:.1f}%  (n={p:,})", va="center", fontsize=8)

    plt.suptitle("Card Win Rates (Clash Royale)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "card_win_rates.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved card_win_rates.png")


def plot_win_rate_distribution(rates: pl.DataFrame, card_map: dict[int, str],
                               min_games: int = 50):
    """Enhanced dot-strip chart showing every card's win rate with annotations."""
    filtered = (
        rates.filter(pl.col("times_played") >= min_games)
        .sort("win_rate", descending=True)
    )
    names = [resolve_name(c, card_map) for c in filtered["card"].to_list()]
    wr = filtered["win_rate"].to_list()
    played = filtered["times_played"].to_list()

    mean_wr = sum(wr) / len(wr)
    sorted_wr = sorted(wr)
    median_wr = sorted_wr[len(sorted_wr) // 2]

    # Colour each bar: green if above 50 %, red if below
    colors = ["#2ecc71" if w >= 50 else "#e74c3c" for w in wr]

    fig, ax = plt.subplots(figsize=(14, max(8, len(names) * 0.32)))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, wr, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()  # highest win rate on top
    ax.set_xlabel("Win Rate (%)")
    ax.set_title("Card Win Rates – All Cards (min 50 games)",
                 fontsize=13, fontweight="bold")

    # Reference lines
    ax.axvline(50, color="black", linestyle="--", linewidth=1, label="50 % baseline")
    ax.axvline(mean_wr, color="dodgerblue", linestyle="-.", linewidth=1,
               label=f"Mean {mean_wr:.1f} %")
    ax.axvline(median_wr, color="orange", linestyle=":", linewidth=1,
               label=f"Median {median_wr:.1f} %")

    # Win-rate labels on each bar
    for bar, w, n in zip(bars, wr, played):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%  (n={n:,})", va="center", fontsize=6)

    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(left=min(wr) - 2, right=max(wr) + 5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "card_win_rate_distribution.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    print("Saved card_win_rate_distribution.png")


def plot_top_decks(deck_rates: pl.DataFrame, card_map: dict[int, str],
                   top_n: int = 10):
    """Horizontal bar chart for the highest win‑rate decks."""
    top = deck_rates.head(top_n)

    labels = []
    for deck_str in top["deck"].to_list():
        card_ids = deck_str.split(",")
        card_names = [resolve_name(cid, card_map) for cid in card_ids]
        labels.append(" | ".join(card_names))

    labels = labels[::-1]
    wr = top["win_rate"].to_list()[::-1]
    played = top["times_played"].to_list()[::-1]

    plt.figure(figsize=(16, 8))
    bars = plt.barh(range(len(labels)), wr, color=sns.color_palette("viridis", len(labels)))
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xlabel("Win Rate (%)")
    plt.title(f"Top {top_n} Decks by Win Rate")

    for bar, w, p in zip(bars, wr, played):
        plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{w:.1f}%  (n={p:,})", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_decks_win_rate.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved top_decks_win_rate.png")


# ----------------------------
# Main
# ----------------------------

def main():
    df = load_clean_data()
    print(f"Loaded {df.height:,} matches")

    card_map = fetch_card_names()

    # --- Card‑level win rates ---
    c_rates = card_win_rates(df)
    c_rates = c_rates.with_columns(
        pl.Series("card_name", [resolve_name(c, card_map) for c in c_rates["card"].to_list()])
    )

    print("\nTop 15 cards by win rate (min 50 games):")
    print(
        c_rates.filter(pl.col("times_played") >= 50)
        .select("card_name", "win_rate", "times_played")
        .head(15)
    )
    print("\nBottom 15 cards by win rate (min 50 games):")
    print(
        c_rates.filter(pl.col("times_played") >= 50)
        .select("card_name", "win_rate", "times_played")
        .tail(15)
    )

    plot_card_win_rates(c_rates, card_map)
    plot_win_rate_distribution(c_rates, card_map)

    # --- Deck‑level win rates (min 50 appearances for statistical reliability) ---
    d_rates = deck_win_rates(df, min_appearances=50)
    print(f"\nUnique decks with ≥ 50 appearances: {d_rates.height:,}")
    print("Top 10 decks by win rate:")
    for row in d_rates.head(10).iter_rows(named=True):
        card_ids = row["deck"].split(",")
        names = [resolve_name(cid, card_map) for cid in card_ids]
        print(f"  {row['win_rate']:.1f}% (n={row['times_played']})  –  {' | '.join(names)}")

    plot_top_decks(d_rates, card_map)

    # --- Save tables ---
    c_rates.write_csv(OUT_DIR / "card_win_rates.csv")
    d_rates.write_csv(OUT_DIR / "deck_win_rates.csv")
    print(f"\nCSV tables saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
