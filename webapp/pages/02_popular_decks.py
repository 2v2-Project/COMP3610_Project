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

# ---------------------------------------------------------------------------
# RoyaleAPI-inspired dark theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Page background ---- */
    section.main > div { max-width: 1100px; margin: auto; }

    /* ---- Metric tiles ---- */
    div[data-testid="stMetric"] {
        background-color: #1e2a3a;
        border: 1px solid #2d3f54;
        padding: 10px 14px;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 12px !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 18px !important; }

    /* ---- Deck card container ---- */
    .deck-tile {
        background: #1a2332;
        border: 1px solid #2d3f54;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }

    /* ---- Win/loss bar ---- */
    .winloss-bar {
        display: flex;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 6px 0 10px 0;
    }
    .winloss-bar .win-segment  { background: #4ade80; }
    .winloss-bar .loss-segment { background: #f87171; }

    /* ---- Stats table ---- */
    .stats-table { width: 100%; border-collapse: collapse; margin-top: 4px; }
    .stats-table th {
        background: #253347;
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        padding: 8px 10px;
        text-align: center;
        border-bottom: 1px solid #2d3f54;
    }
    .stats-table td {
        color: #e2e8f0;
        font-size: 15px;
        padding: 8px 10px;
        text-align: center;
    }
    .stats-table .win-val  { color: #4ade80; }
    .stats-table .loss-val { color: #f87171; }

    /* ---- Elixir / Cycle info row ---- */
    .elixir-cycle-row {
        display: flex; align-items: center; gap: 24px; margin-top: 14px;
    }
    .elixir-cycle-row .stat-item {
        display: flex; align-items: center; gap: 6px;
    }
    .elixir-cycle-row .stat-item span {
        color: #e2e8f0; font-size: 14px; font-weight: 600;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] { background: #0f1923; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label { color: #e2e8f0 !important; }

    /* ---- Confidence badges ---- */
    .conf-badge {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-weight: 600; font-size: 12px; color: white; margin-top: 4px;
    }
    .conf-high   { background: #16a34a; }
    .conf-medium { background: #f59e0b; }
    .conf-low    { background: #ef4444; }

    /* ---- Deck header row ---- */
    .deck-header {
        display: flex; align-items: center; gap: 12px;
        margin-bottom: 10px; color: #94a3b8; font-size: 13px;
    }
    .deck-header .archetype-tag {
        background: #253347; padding: 5px 14px; border-radius: 6px;
        font-weight: 700; color: #e2e8f0; font-size: 16px;
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


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------
ELIXIR_ICON = "https://cdn.royaleapi.com/static/img/ui/elixir.png"
CARDS_ICON = "https://cdn.royaleapi.com/static/img/ui/cards.png"


# ---------------------------------------------------------------------------
# Card image renderer
# ---------------------------------------------------------------------------
def _render_card_image(card_id: int, icon_map: dict[int, str], name_map: dict[int, str]) -> None:
    icon_url = icon_map.get(card_id)
    card_name = name_map.get(card_id, str(card_id))
    if icon_url and isinstance(icon_url, str) and icon_url.strip():
        st.image(icon_url, use_container_width=True)
    else:
        st.markdown(
            f"<div style='border:1px solid #333;border-radius:10px;padding:12px 6px;"
            f"text-align:center;min-height:90px;display:flex;align-items:center;"
            f"justify-content:center;font-size:12px;color:#bbb;background:#111827'>"
            f"{card_name}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def render_winloss_bar(win_rate: float) -> str:
    """Return HTML for a green/red win-loss bar."""
    loss_rate = 100.0 - win_rate
    return (
        f"<div class='winloss-bar'>"
        f"<div class='win-segment' style='width:{win_rate:.1f}%'></div>"
        f"<div class='loss-segment' style='width:{loss_rate:.1f}%'></div>"
        f"</div>"
    )


def render_stats_table(row: pd.Series) -> str:
    """Return an HTML stats table similar to RoyaleAPI."""
    wins = int(row["wins"])
    matches = int(row["matches_played"])
    losses = matches - wins
    win_pct = row["win_rate"]

    return f"""
    <table class='stats-table'>
      <tr>
        <th>Win Rate</th><th>Matches</th><th>Wins</th><th>Losses</th>
      </tr>
      <tr>
        <td class='win-val'>{win_pct:.1f}%</td>
        <td>{matches:,}</td>
        <td class='win-val'>{wins:,}</td>
        <td class='loss-val'>{losses:,}</td>
      </tr>
    </table>
    """


def render_deck_card(
    row: pd.Series,
    icon_map: dict[int, str],
    name_map: dict[int, str],
) -> None:
    conf = row["confidence"]
    conf_cls = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}.get(conf, "conf-medium")

    st.markdown(
        f"<div class='deck-header'>"
        f"<span class='archetype-tag'>{row['archetype']}</span>"
        f"<span>{row['troop_count']}T / {row['spell_count']}S / {row['building_count']}B</span>"
        f"<span class='conf-badge {conf_cls}'>{conf} Confidence</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    card_col, stats_col = st.columns([1, 1], gap="medium")

    with card_col:
        top_row = st.columns(4, gap="small")
        for i in range(4):
            cid = int(row["card_ids"][i])
            with top_row[i]:
                _render_card_image(cid, icon_map, name_map)

        bottom_row = st.columns(4, gap="small")
        for i in range(4, 8):
            cid = int(row["card_ids"][i])
            with bottom_row[i - 4]:
                _render_card_image(cid, icon_map, name_map)

    with stats_col:
        st.markdown(render_winloss_bar(row["win_rate"]), unsafe_allow_html=True)
        st.markdown(render_stats_table(row), unsafe_allow_html=True)

        st.markdown(
            f"<div class='elixir-cycle-row'>"
            f"<div class='stat-item'>"
            f"<img src='{ELIXIR_ICON}' width='24' height='24'>"
            f"<span>Avg Elixir : {row['avg_elixir']:.1f}</span>"
            f"</div>"
            f"<div class='stat-item'>"
            f"<img src='{CARDS_ICON}' width='24' height='24'>"
            f"<span>4-Card Cycle : {int(row['cycle_cost'])}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("🏆 Popular Decks")
    st.markdown(
        "<div style='margin-bottom:10px;color:#94a3b8;font-size:15px;'>"
        "Explore the most-played decks and compare their historical performance, "
        "archetype, and deck profile. Click any card for details."
        "</div>",
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

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Decks Shown", len(filtered_df))
    metric2.metric("Average Win Rate", f'{filtered_df["win_rate"].mean():.2f}%')
    metric3.metric("Average Elixir", f'{filtered_df["avg_elixir"].mean():.2f}')
    metric4.metric(
        "Top Archetype",
        filtered_df["archetype"].mode().iloc[0] if not filtered_df.empty else "N/A",
    )

    st.divider()

    for idx, row in filtered_df.iterrows():
        render_deck_card(row, icon_map, name_map)


if __name__ == "__main__":
    main()