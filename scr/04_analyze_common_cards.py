from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import requests


# ----------------------------
# Paths
# ----------------------------
INPUT_PARQUET = Path("data/processed/clash_royale_clean.parquet")
INPUT_CSV = Path("data/processed/clash_royale_clean.csv")

OUT_DIR = Path("data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Card columns
# ----------------------------
P1_CARDS = [f"player1.card{i}" for i in range(1, 9)]
P2_CARDS = [f"player2.card{i}" for i in range(1, 9)]
ALL_CARD_COLS = P1_CARDS + P2_CARDS


# ----------------------------
# Card metadata API
# ----------------------------
CARD_API = "https://royaleapi.github.io/cr-api-data/json/cards.json"


def load_clean_data() -> pl.DataFrame:
    """
    Load the cleaned dataset.
    """
    if INPUT_PARQUET.exists():
        return pl.read_parquet(INPUT_PARQUET)

    if INPUT_CSV.exists():
        return pl.read_csv(INPUT_CSV)

    raise FileNotFoundError(
        "Cleaned dataset not found. Run preprocessing first."
    )


def fetch_card_names() -> dict[int, str]:
    """
    Download Clash Royale card metadata
    and build {card_id: card_name}.
    """
    try:
        response = requests.get(CARD_API, timeout=20)
        response.raise_for_status()

        cards = response.json()

        card_map = {
            int(card["id"]): card["name"]
            for card in cards
        }

        print(f"Fetched {len(card_map)} card names.")
        return card_map

    except Exception as e:
        print(f"Could not fetch card names: {e}")
        return {}


def count_cards(df: pl.DataFrame, card_cols: list[str]) -> pl.DataFrame:
    """
    Count card frequency and calculate percentage usage.
    """

    counts = (
        df.select(card_cols)
        .melt(variable_name="slot", value_name="card")
        .drop("slot")
        .drop_nulls()
        .group_by("card")
        .count()
        .rename({"count": "usage_count"})
        .sort("usage_count", descending=True)
    )

    total_cards = counts["usage_count"].sum()

    counts = counts.with_columns(
        (pl.col("usage_count") / total_cards * 100)
        .alias("usage_percent")
    )

    return counts


def add_card_names(df: pl.DataFrame, card_map: dict[int, str]) -> pl.DataFrame:
    """
    Add readable card names.
    """

    names = [
        card_map.get(int(cid), str(cid))
        for cid in df["card"].to_list()
    ]

    return df.with_columns(pl.Series("card_name", names))


def plot_top_cards(df: pl.DataFrame, top_n: int = 10):
    """
    Create a horizontal bar chart of the most used cards.
    """

    top = df.head(top_n)

    card_names = top["card_name"].to_list()[::-1]
    percentages = top["usage_percent"].to_list()[::-1]

    plt.figure(figsize=(12, 7))

    bars = plt.barh(card_names, percentages)

    plt.title("Top 10 Most Commonly Used Cards")
    plt.xlabel("Percentage of All Card Appearances (%)")
    plt.ylabel("Card")

    # add percentage labels
    for bar, pct in zip(bars, percentages):
        plt.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}%",
            va="center"
        )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "top10_common_cards.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def _resolve_name(card_id, card_map: dict[int, str]) -> str:
    try:
        return card_map.get(int(card_id), str(card_id))
    except (ValueError, TypeError):
        return str(card_id)


def count_decks(df: pl.DataFrame, card_map: dict[int, str]) -> pl.DataFrame:
    """
    Count how often each unique 8-card deck appears (both players combined).
    A deck is identified by sorting its 8 card IDs.
    """
    p1 = df.with_columns(
        pl.concat_list(P1_CARDS)
        .list.sort()
        .list.eval(pl.element().cast(pl.Utf8))
        .list.join(",")
        .alias("deck")
    ).select("deck")

    p2 = df.with_columns(
        pl.concat_list(P2_CARDS)
        .list.sort()
        .list.eval(pl.element().cast(pl.Utf8))
        .list.join(",")
        .alias("deck")
    ).select("deck")

    combined = pl.concat([p1, p2])

    counts = (
        combined
        .group_by("deck")
        .agg(pl.len().alias("usage_count"))
        .sort("usage_count", descending=True)
    )

    total = counts["usage_count"].sum()
    counts = counts.with_columns(
        (pl.col("usage_count") / total * 100).alias("usage_percent")
    )

    # Add a readable deck name column
    labels = []
    for deck_str in counts["deck"].to_list():
        card_ids = deck_str.split(",")
        card_names = [_resolve_name(cid, card_map) for cid in card_ids]
        labels.append(" | ".join(card_names))
    counts = counts.with_columns(pl.Series("deck_name", labels))

    return counts


def plot_top_decks(deck_counts: pl.DataFrame, top_n: int = 10):
    """
    Horizontal bar chart of the most commonly used decks.
    """
    top = deck_counts.head(top_n)

    labels = top["deck_name"].to_list()[::-1]
    counts = top["usage_count"].to_list()[::-1]
    pcts = top["usage_percent"].to_list()[::-1]

    plt.figure(figsize=(16, 8))
    bars = plt.barh(range(len(labels)), counts, color="steelblue")

    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xlabel("Number of Appearances")
    plt.title(f"Top {top_n} Most Commonly Used Decks")

    for bar, cnt, pct in zip(bars, counts, pcts):
        plt.text(
            bar.get_width() + bar.get_width() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,}  ({pct:.3f}%)",
            va="center", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "top10_common_decks.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():

    df = load_clean_data()
    print(f"Loaded cleaned data: {df.height:,} rows")

    card_map = fetch_card_names()

    # --- Top cards ---
    card_counts = count_cards(df, ALL_CARD_COLS)

    card_counts = add_card_names(card_counts, card_map)

    print("\nTop 10 most commonly used cards:")
    print(
        card_counts.select(
            ["card_name", "usage_count", "usage_percent"]
        ).head(10)
    )

    plot_top_cards(card_counts)

    # --- Top decks ---
    deck_counts = count_decks(df, card_map)

    print("\nTop 10 most commonly used decks:")
    for row in deck_counts.head(10).iter_rows(named=True):
        print(f"  {row['usage_count']:,} ({row['usage_percent']:.3f}%)  –  {row['deck_name']}")

    plot_top_decks(deck_counts)

    print("\nGraphs saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()