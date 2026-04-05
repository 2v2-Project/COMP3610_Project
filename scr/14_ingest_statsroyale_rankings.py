"""
StatsRoyale Card Rankings Ingestion
====================================
Scrapes the top‑cards page from StatsRoyale and saves a clean
parquet file with card name, usage rate, and rank.

Source: https://statsroyale.com/top/cards

Usage:
    python scr/14_ingest_statsroyale_rankings.py
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUTPUT_PATH = Path("data/processed/card_rankings.parquet")
URL = "https://statsroyale.com/top/cards"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _parse_usage_rate(text: str) -> float | None:
    """Convert a string like '12.3%' or '12.3' to a float."""
    m = re.search(r"([\d]+(?:\.[\d]+)?)\s*%?", text)
    if m:
        return float(m.group(1))
    return None


def scrape_statsroyale() -> pd.DataFrame:
    """Scrape card rankings from StatsRoyale top cards page."""
    print(f"Fetching {URL} ...")
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    rows: list[dict] = []

    # StatsRoyale renders cards inside <a> elements that each contain
    # the card name and a usage‑rate span.  Multiple CSS class names
    # have been used over the years, so we try several selectors.
    card_elements = (
        soup.select("div.ui__mediumCard")
        or soup.select("div.card")
        or soup.select("a.ui__mediumCard")
        or soup.select("a[href*='/cards/']")
    )

    for idx, el in enumerate(card_elements, start=1):
        # Card name — look for a dedicated element first, fall back to text
        name_el = (
            el.select_one(".ui__mediumCard__name")
            or el.select_one(".card__name")
            or el.select_one("[class*='name']")
        )
        card_name = name_el.get_text(strip=True) if name_el else None

        if not card_name:
            # Try the alt attribute of the card image
            img = el.select_one("img")
            if img and img.get("alt"):
                card_name = img["alt"].strip()

        if not card_name:
            continue

        # Usage rate
        usage_el = (
            el.select_one(".ui__mediumCard__count")
            or el.select_one("[class*='usage']")
            or el.select_one("[class*='percent']")
            or el.select_one("[class*='count']")
        )
        usage_rate = None
        if usage_el:
            usage_rate = _parse_usage_rate(usage_el.get_text(strip=True))

        rows.append({
            "card_name": card_name,
            "usage_rate": usage_rate,
            "rank": idx,
        })

    df = pd.DataFrame(rows)
    return df


def _compute_usage_from_match_data() -> pd.DataFrame:
    """
    Fallback: compute card usage rates from the local match dataset
    when StatsRoyale is unreachable or blocks the request.
    """
    meta_path = Path("data/processed/card_metadata.csv")
    parquet_path = Path("data/processed/clash_royale_clean.parquet")

    if not meta_path.exists():
        print("ERROR: cannot build fallback — missing card_metadata.csv")
        sys.exit(1)

    card_meta = pd.read_csv(meta_path)

    # Read only the 8 player-1 card columns from parquet (fast, low memory)
    p1_card_cols = [f"player1.card{i}" for i in range(1, 9)]

    if parquet_path.exists():
        print("Loading card columns from parquet...")
        match_df = pd.read_parquet(parquet_path, columns=p1_card_cols)
    else:
        csv_path = Path("data/processed/clash_royale_clean.csv")
        if csv_path.exists():
            print("Loading card columns from CSV (this may take a moment)...")
            match_df = pd.read_csv(csv_path, usecols=p1_card_cols)
        else:
            print("ERROR: no match data found to compute usage rates from.")
            sys.exit(1)

    total_matches = len(match_df)
    print(f"Computing usage rates from {total_matches:,} matches...")
    usage_counts: dict[int, int] = {}

    for col in p1_card_cols:
        for cid, count in match_df[col].value_counts().items():
            cid_int = int(cid)
            usage_counts[cid_int] = usage_counts.get(cid_int, 0) + count

    # Build dataframe
    rows = []
    for cid, count in usage_counts.items():
        name_row = card_meta.loc[card_meta["card_id"] == cid, "name"]
        card_name = name_row.values[0] if len(name_row) > 0 else None
        if card_name is None:
            continue
        usage_rate = round((count / total_matches) * 100, 2)
        rows.append({"card_name": card_name, "usage_rate": usage_rate})

    df = pd.DataFrame(rows)
    df = df.sort_values("usage_rate", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules: drop missing names, dedup, ensure numeric usage."""
    df = df.dropna(subset=["card_name"]).copy()
    df = df.drop_duplicates(subset=["card_name"], keep="first")
    df["usage_rate"] = pd.to_numeric(df["usage_rate"], errors="coerce")
    df = df.sort_values("rank").reset_index(drop=True)
    return df


def main() -> None:
    try:
        df = scrape_statsroyale()
        if df.empty:
            raise ValueError("Scrape returned 0 cards")
        print(f"Scraped {len(df)} cards from StatsRoyale.")
    except Exception as exc:
        print(f"StatsRoyale scrape failed ({exc}). "
              "Falling back to local match data usage rates...")
        df = _compute_usage_from_match_data()
        print(f"Computed usage rates for {len(df)} cards from local data.")

    df = clean(df)

    # Add metadata columns
    df["date_scraped"] = date.today().isoformat()
    df["source"] = "StatsRoyale"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    # Also save a lightweight CSV backup for deployment fallback
    csv_path = OUTPUT_PATH.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSaved {len(df)} cards to {OUTPUT_PATH}")
    print(f"CSV backup saved to {csv_path}")
    print(f"\n{df.head(15).to_string(index=False)}")


if __name__ == "__main__":
    main()
