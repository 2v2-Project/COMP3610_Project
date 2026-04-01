from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from utils.metadata import (
    get_card_metadata,
    get_card_names,
    get_card_types,
    get_elixir_costs,
    get_icon_urls,
)
from utils.deck_helpers import (
    build_deck_key,
    enrich_deck_record,
)

st.set_page_config(page_title="Popular Decks", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 12px;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATHS = [
    Path("data/processed/clash_royale_clean.parquet"),
    Path("data/processed/final_ml_dataset.parquet"),
]

PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
PLAYER_CROWNS_COL = "player1.crowns"
OPPONENT_CROWNS_COL = "player2.crowns"

DEFAULT_MIN_MATCHES = 50
DEFAULT_TOP_N = 30


def load_card_assets():
    metadata_df = get_card_metadata(force_refresh=True)
    name_map = get_card_names(force_refresh=True)
    type_map = get_card_types(force_refresh=True)
    elixir_map = get_elixir_costs(force_refresh=True)
    icon_map = get_icon_urls(force_refresh=True)
    return metadata_df, name_map, type_map, elixir_map, icon_map


@st.cache_data(show_spinner=True)
def load_match_data() -> pl.DataFrame:
    for path in DATA_PATHS:
        if path.exists():
            df = pl.read_parquet(path)
            required_cols = set(PLAYER_CARD_COLS + [PLAYER_CROWNS_COL, OPPONENT_CROWNS_COL])

            if required_cols.issubset(set(df.columns)):
                return df.select(PLAYER_CARD_COLS + [PLAYER_CROWNS_COL, OPPONENT_CROWNS_COL])

    raise FileNotFoundError(
        "Could not find a suitable parquet file with player deck cards and crown columns."
    )


@st.cache_data(show_spinner=True)
def build_popular_decks_table(min_matches: int) -> pd.DataFrame:
    df = load_match_data()
    pdf = df.to_pandas()

    pdf["deck_key"] = pdf[PLAYER_CARD_COLS].apply(
        lambda row: build_deck_key([int(x) for x in row.tolist()]),
        axis=1,
    )

    pdf["player_win"] = (pdf[PLAYER_CROWNS_COL] > pdf[OPPONENT_CROWNS_COL]).astype(int)

    grouped = (
        pdf.groupby("deck_key", as_index=False)
        .agg(
            matches_played=("player_win", "count"),
            wins=("player_win", "sum"),
        )
    )

    grouped = grouped[grouped["matches_played"] >= min_matches].copy()

    _, name_map, type_map, elixir_map, _ = load_card_assets()

    records = [
        enrich_deck_record(
            deck_key=row["deck_key"],
            matches_played=int(row["matches_played"]),
            wins=int(row["wins"]),
            name_map=name_map,
            elixir_map=elixir_map,
            type_map=type_map,
        )
        for _, row in grouped.iterrows()
    ]

    result = pd.DataFrame(records)
    result = result.sort_values(["matches_played", "win_rate"], ascending=[False, False]).reset_index(drop=True)
    return result


def render_confidence_badge(confidence: str):
    color_map = {
        "High": "#16a34a",
        "Medium": "#f59e0b",
        "Low": "#ef4444",
    }
    color = color_map.get(confidence, "#6b7280")

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            background-color: {color};
            color: white;
            font-weight: 600;
            font-size: 14px;
            margin-top: 6px;
        ">
            {confidence} Confidence
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_deck_images(card_ids: list[int], icon_map: dict[int, str], name_map: dict[int, str]) -> None:
    cols = st.columns(8, gap="small")

    for i, raw_card_id in enumerate(card_ids):
        with cols[i]:
            card_id = int(raw_card_id)
            icon_url = icon_map.get(card_id)
            card_name = name_map.get(card_id, str(card_id))

            if icon_url and isinstance(icon_url, str) and icon_url.strip():
                st.image(icon_url, use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #333;
                        border-radius: 10px;
                        padding: 12px 6px;
                        text-align: center;
                        min-height: 90px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 12px;
                        color: #bbb;
                        background-color: #111827;
                    ">
                        {card_name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_deck_card(row: pd.Series, icon_map: dict[int, str], name_map: dict[int, str]) -> None:
    deck_names = [name_map.get(int(card_id), str(card_id)) for card_id in row["card_ids"]]

    with st.container(border=True):
        render_deck_images(row["card_ids"], icon_map, name_map)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Win Rate", f'{row["win_rate"]:.2f}%')
        m2.metric("Matches", f'{int(row["matches_played"]):,}')
        m3.metric("Archetype", row["archetype"])
        m4.metric("Avg Elixir", f'{row["avg_elixir"]:.2f}')
        m5.metric("Cycle Cost", int(row["cycle_cost"]))

        sub1, sub2 = st.columns([1, 2])

        with sub1:
            render_confidence_badge(row["confidence"])

        with sub2:
            st.markdown(
                f"""
                <div style="
                    padding-top: 10px;
                    font-size: 15px;
                    color: #d1d5db;
                ">
                    <strong>Deck Composition:</strong>
                    {row["troop_count"]} Troops •
                    {row["spell_count"]} Spells •
                    {row["building_count"]} Buildings
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(" • ".join(deck_names))


def main():
    st.title("🏆 Popular Decks")
    st.markdown(
        """
        <div style="margin-bottom: 10px; color: #cbd5e1; font-size: 16px;">
            Explore the most-played decks in the dataset and compare their historical performance,
            archetype, and deck profile.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metadata_df, name_map, type_map, elixir_map, icon_map = load_card_assets()

    with st.sidebar:
        st.header("Filters")

        min_matches = st.slider(
            "Minimum matches",
            min_value=10,
            max_value=1000,
            value=DEFAULT_MIN_MATCHES,
            step=10,
        )

        top_n = st.slider(
            "Decks to show",
            min_value=10,
            max_value=100,
            value=DEFAULT_TOP_N,
            step=5,
        )

        sort_by = st.selectbox(
            "Sort by",
            options=[
                "Most Played",
                "Highest Win Rate",
                "Lowest Elixir",
                "Highest Elixir",
                "Best Cycle",
            ],
        )

    decks_df = build_popular_decks_table(min_matches=min_matches)

    st.write("Metadata rows:", len(metadata_df))
    st.write("Icon count:", len(icon_map))
    st.write("Sample icons:", list(icon_map.items())[:5])
    st.write("First deck card ids:", decks_df["card_ids"].iloc[0] if not decks_df.empty else "No deck data")

    if decks_df.empty:
        st.warning("No decks matched the selected minimum match threshold.")
        return

    archetype_options = ["All"] + sorted(decks_df["archetype"].dropna().unique().tolist())
    confidence_options = ["All", "High", "Medium", "Low"]

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.1])

    with filter_col1:
        selected_archetype = st.selectbox("Archetype", archetype_options)

    with filter_col2:
        selected_confidence = st.selectbox("Confidence", confidence_options)

    with filter_col3:
        elixir_range = st.slider(
            "Average elixir range",
            min_value=0.0,
            max_value=8.0,
            value=(2.0, 6.0),
            step=0.1,
        )

    filtered_df = decks_df.copy()

    if selected_archetype != "All":
        filtered_df = filtered_df[filtered_df["archetype"] == selected_archetype]

    if selected_confidence != "All":
        filtered_df = filtered_df[filtered_df["confidence"] == selected_confidence]

    filtered_df = filtered_df[
        (filtered_df["avg_elixir"] >= elixir_range[0]) &
        (filtered_df["avg_elixir"] <= elixir_range[1])
    ]

    if sort_by == "Most Played":
        filtered_df = filtered_df.sort_values(["matches_played", "win_rate"], ascending=[False, False])
    elif sort_by == "Highest Win Rate":
        filtered_df = filtered_df.sort_values(["win_rate", "matches_played"], ascending=[False, False])
    elif sort_by == "Lowest Elixir":
        filtered_df = filtered_df.sort_values(["avg_elixir", "win_rate"], ascending=[True, False])
    elif sort_by == "Highest Elixir":
        filtered_df = filtered_df.sort_values(["avg_elixir", "win_rate"], ascending=[False, False])
    elif sort_by == "Best Cycle":
        filtered_df = filtered_df.sort_values(["cycle_cost", "win_rate"], ascending=[True, False])

    filtered_df = filtered_df.head(top_n).reset_index(drop=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Decks Shown", len(filtered_df))
    metric2.metric("Average Win Rate", f'{filtered_df["win_rate"].mean():.2f}%')
    metric3.metric("Average Elixir", f'{filtered_df["avg_elixir"].mean():.2f}')
    metric4.metric(
        "Top Archetype",
        filtered_df["archetype"].mode().iloc[0] if not filtered_df.empty else "N/A"
    )

    st.divider()

    for _, row in filtered_df.iterrows():
        render_deck_card(row, icon_map, name_map)


if __name__ == "__main__":
    main()