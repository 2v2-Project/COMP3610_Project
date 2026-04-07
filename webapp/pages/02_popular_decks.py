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
from utils.deck_helpers import enrich_deck_record
from utils.uncertainty import confidence_from_match_count
from utils.ui_helpers import inject_fonts
from utils.data_loader import get_clean_parquet_source, get_final_ml_parquet_source

st.set_page_config(page_title="Popular Decks", layout="wide")
inject_fonts()

st.markdown(
    """
    <style>
    section.main > div {
        max-width: 1180px;
        margin: auto;
    }

    .page-subtitle {
        margin-bottom: 14px;
        color: #5a7394;
        font-size: 15px;
    }

    .filter-box {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(208, 219, 232, 0.5);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 4px rgba(26, 86, 219, 0.06);
        margin-top: 10px;
        margin-bottom: 16px;
    }

    .summary-chip {
        display: inline-block;
        background: rgba(238, 244, 255, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: #1a3a6e;
        border: 1px solid rgba(215, 227, 247, 0.5);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }

    .deck-card-shell {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(208, 219, 232, 0.5);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 2px 8px rgba(26, 86, 219, 0.06);
        margin-bottom: 18px;
    }

    .deck-card-shell [data-testid="stHorizontalBlock"] {
        align-items: flex-start;
    }

    .deck-card-shell [data-testid="stImage"] img {
        border-radius: 12px;
    }

    .deck-header {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 14px;
        color: #6b7fa3;
        font-size: 13px;
    }

    .deck-header .archetype-tag {
        background: #dce6f5;
        padding: 5px 14px;
        border-radius: 999px;
        font-weight: 700;
        color: #1a3a6e;
        font-size: 15px;
    }

    .conf-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 12px;
        color: white;
    }

    .conf-high   { background: #16a34a; }
    .conf-medium { background: #f59e0b; }
    .conf-low    { background: #ef4444; }

    .stats-table {
        width: 100%;
        border-collapse: collapse;
        background: rgba(248, 251, 255, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(215, 227, 247, 0.5);
        border-radius: 12px;
        overflow: hidden;
    }

    .stats-table th {
        color: #4a6280;
        font-size: 13px;
        font-weight: 600;
        padding: 8px 10px;
        text-align: center;
        border-bottom: 1px solid #c5d5ea;
    }

    .stats-table td {
        color: #1a3a6e;
        font-size: 15px;
        padding: 8px 10px;
        text-align: center;
    }

    .stats-table .win-val  { color: #16a34a; }
    .stats-table .loss-val { color: #dc2626; }

    .winloss-bar {
        width: 100%;
        height: 16px;
        background: rgba(229, 237, 248, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 999px;
        overflow: hidden;
        display: flex;
        margin-bottom: 12px;
        border: 1px solid rgba(215, 227, 247, 0.5);
    }

    .win-segment {
        background: #22c55e;
        height: 100%;
    }

    .loss-segment {
        background: #ef4444;
        height: 100%;
    }

    .elixir-cycle-row {
        display: flex;
        align-items: center;
        gap: 24px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .elixir-cycle-row .stat-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .elixir-cycle-row .stat-item span {
        color: #1a3a6e;
        font-size: 15px;
        font-weight: 700;
        white-space: nowrap;
    }

    .deck-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 999px;
        background: #1a56db;
        color: white;
        font-weight: 800;
        font-size: 14px;
    }

    .card-caption {
        text-align: center;
        color: #4a6280;
        font-size: 11px;
        line-height: 1.2;
        min-height: 28px;
        margin-top: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
PLAYER_CROWNS_COL = "player1.crowns"
OPPONENT_CROWNS_COL = "player2.crowns"

DEFAULT_MIN_MATCHES = 50
DEFAULT_TOP_N = 30
MAX_ENRICHED_DECKS = 1500
SAMPLE_SIZE = 200_000

ELIXIR_ICON = "https://cdn.royaleapi.com/static/img/ui/elixir.png"
CARDS_ICON = "https://cdn.royaleapi.com/static/img/ui/cards.png"


@st.cache_data(show_spinner=False, ttl=3600)
def load_card_assets():
    metadata_df = get_card_metadata(force_refresh=False)
    name_map = get_card_names(force_refresh=False)
    type_map = get_card_types(force_refresh=False)
    elixir_map = get_elixir_costs(force_refresh=False)
    icon_map = get_icon_urls(force_refresh=False)

    card_df = pd.DataFrame(
        {
            "card_id": list(name_map.keys()),
            "name": [name_map[cid] for cid in name_map.keys()],
        }
    )

    banned_keywords = [
        "super ",
        "santa ",
        "terry",
        "party",
        "evolved ",
        "evolution",
        "mirror mode",
        "placeholder",
    ]

    mask = ~card_df["name"].str.lower().apply(
        lambda x: any(keyword in x for keyword in banned_keywords)
    )

    card_df = card_df[mask].copy()
    card_df = card_df.sort_values("name").reset_index(drop=True)

    valid_ids = set(card_df["card_id"].tolist())

    metadata_df = metadata_df[metadata_df["card_id"].isin(valid_ids)].copy()
    name_map = {cid: name for cid, name in name_map.items() if cid in valid_ids}
    type_map = {cid: ctype for cid, ctype in type_map.items() if cid in valid_ids}
    elixir_map = {cid: cost for cid, cost in elixir_map.items() if cid in valid_ids}
    icon_map = {cid: url for cid, url in icon_map.items() if cid in valid_ids}

    return metadata_df, card_df, name_map, type_map, elixir_map, icon_map


@st.cache_data(show_spinner=True, ttl=3600)
def load_match_data() -> pl.DataFrame:
    needed = PLAYER_CARD_COLS + [PLAYER_CROWNS_COL, OPPONENT_CROWNS_COL]
    for get_src in (get_clean_parquet_source, get_final_ml_parquet_source):
        try:
            df = pl.scan_parquet(get_src()).select(needed).collect()
            if df.height > SAMPLE_SIZE:
                df = df.sample(n=SAMPLE_SIZE, seed=42)
            return df
        except Exception:
            continue

    raise FileNotFoundError(
        "Could not find a suitable parquet file with player deck cards and crown columns."
    )


def _confidence_label_from_matches(matches: int) -> str:
    return confidence_from_match_count(matches_played=int(matches))


def _render_card_image(card_id: int, icon_map: dict[int, str], name_map: dict[int, str]) -> None:
    icon_url = icon_map.get(int(card_id))
    card_name = name_map.get(int(card_id), str(card_id))

    if icon_url and isinstance(icon_url, str) and icon_url.strip():
        st.image(icon_url, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="
                border:1px solid #c5d5ea;
                border-radius:10px;
                padding:12px 6px;
                text-align:center;
                min-height:90px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:12px;
                color:#6b7fa3;
                background:#ffffff;">
                {card_name}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_winloss_bar(win_rate: float) -> str:
    loss_rate = max(0.0, 100.0 - float(win_rate))
    return (
        f"<div class='winloss-bar'>"
        f"<div class='win-segment' style='width:{float(win_rate):.1f}%'></div>"
        f"<div class='loss-segment' style='width:{loss_rate:.1f}%'></div>"
        f"</div>"
    )


def render_stats_table(row: pd.Series) -> str:
    matches = int(row["matches_played"])
    wins = int(row["wins"])
    losses = int(row["losses"])
    win_rate = float(row["win_rate"])

    return f"""
    <table class="stats-table">
      <tr>
        <th>Win Rate</th>
        <th>Matches</th>
        <th>Wins</th>
        <th>Losses</th>
      </tr>
      <tr>
        <td>{win_rate:.1f}%</td>
        <td>{matches:,}</td>
        <td class='win-val'>{wins:,}</td>
        <td class='loss-val'>{losses:,}</td>
      </tr>
    </table>
    """


@st.cache_data(show_spinner=True, ttl=3600)
def build_all_popular_decks_table() -> pd.DataFrame:
    df = load_match_data()

    df = df.with_columns(
        pl.concat_list(PLAYER_CARD_COLS).alias("deck_cards")
    ).with_columns(
        pl.col("deck_cards")
        .list.eval(pl.element().cast(pl.Int64))
        .list.sort()
        .list.eval(pl.element().cast(pl.Utf8))
        .list.join(",")
        .alias("deck_key"),
        (pl.col(PLAYER_CROWNS_COL) > pl.col(OPPONENT_CROWNS_COL)).cast(pl.Int8).alias("player_win"),
    )

    grouped = (
        df.group_by("deck_key")
        .agg(
            pl.len().alias("matches_played"),
            pl.col("player_win").sum().alias("wins"),
        )
        .with_columns(
            (pl.col("matches_played") - pl.col("wins")).alias("losses"),
            (pl.col("wins") / pl.col("matches_played") * 100).alias("win_rate"),
        )
        .sort(["matches_played", "win_rate"], descending=[True, True])
        .head(MAX_ENRICHED_DECKS)
    )

    grouped_pd = grouped.to_pandas()

    _, _, name_map, type_map, elixir_map, _ = load_card_assets()

    records = []
    for _, row in grouped_pd.iterrows():
        enriched = enrich_deck_record(
            deck_key=row["deck_key"],
            matches_played=int(row["matches_played"]),
            wins=int(row["wins"]),
            name_map=name_map,
            elixir_map=elixir_map,
            type_map=type_map,
        )

        enriched["losses"] = int(row["losses"])
        enriched["win_rate"] = float(row["win_rate"])
        enriched["confidence"] = _confidence_label_from_matches(int(row["matches_played"]))
        records.append(enriched)

    result = pd.DataFrame(records)

    if result.empty:
        return result

    result = result.sort_values(
        ["matches_played", "win_rate"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return result


def render_deck_card(
    row: pd.Series,
    icon_map: dict[int, str],
    name_map: dict[int, str],
    rank_num: int,
) -> None:
    conf = row.get("confidence", "Medium")
    conf_cls = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}.get(conf, "conf-medium")

    st.markdown(
        f"<div class='deck-header'>"
        f"<span class='archetype-tag'>{row['archetype']}</span>"
        f"<span>{int(row['troop_count'])}T / {int(row['spell_count'])}S / {int(row['building_count'])}B</span>"
        f"<span class='conf-badge {conf_cls}'>{conf} Confidence</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    card_col, stats_col = st.columns([1.15, 0.95], gap="large")

    with card_col:
        top_row = st.columns(4, gap="small")
        for i in range(4):
            cid = int(row["card_ids"][i])
            with top_row[i]:
                _render_card_image(cid, icon_map, name_map)
                st.markdown(
                    f"<div class='card-caption'>{name_map.get(cid, str(cid))}</div>",
                    unsafe_allow_html=True,
                )

        bottom_row = st.columns(4, gap="small")
        for i in range(4, 8):
            cid = int(row["card_ids"][i])
            with bottom_row[i - 4]:
                _render_card_image(cid, icon_map, name_map)
                st.markdown(
                    f"<div class='card-caption'>{name_map.get(cid, str(cid))}</div>",
                    unsafe_allow_html=True,
                )

    with stats_col:
        st.markdown(render_winloss_bar(float(row["win_rate"])), unsafe_allow_html=True)
        st.markdown(render_stats_table(row), unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='elixir-cycle-row'>
                <div class='stat-item'>
                    <img src='{ELIXIR_ICON}' width='26' height='26'>
                    <span>Avg Elixir: {float(row['avg_elixir']):.1f}</span>
                </div>
                <div class='stat-item'>
                    <img src='{CARDS_ICON}' width='26' height='26'>
                    <span>4-Card Cycle: {int(row['cycle_cost'])}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.title("🏆 Popular Decks")
    st.markdown(
        """
        <div class='page-subtitle'>
            Explore the most-played decks and compare their historical performance,
            archetype, and deck profile.
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, _, name_map, _, _, icon_map = load_card_assets()

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

    decks_df = build_all_popular_decks_table()

    if decks_df.empty:
        st.warning("No popular deck data is available.")
        return

    decks_df = decks_df[decks_df["matches_played"] >= min_matches].copy()

    if decks_df.empty:
        st.warning("No decks matched the selected minimum match threshold.")
        return

    archetype_options = ["All"] + sorted(
        [x for x in decks_df["archetype"].dropna().unique().tolist() if str(x).strip()]
    )
    confidence_options = ["All", "High", "Medium", "Low"]

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.15], gap="large")

    with filter_col1:
        selected_archetype = st.selectbox("Archetype", archetype_options)

    with filter_col2:
        selected_confidence = st.selectbox("Confidence", confidence_options)

    with filter_col3:
        elixir_min = float(max(0.0, decks_df["avg_elixir"].min()))
        elixir_max = float(min(8.0, decks_df["avg_elixir"].max()))
        selected_elixir = st.slider(
            "Average elixir range",
            min_value=0.0,
            max_value=8.0,
            value=(round(elixir_min, 1), round(elixir_max, 1)),
            step=0.1,
        )

    filtered_df = decks_df.copy()

    if selected_archetype != "All":
        filtered_df = filtered_df[filtered_df["archetype"] == selected_archetype]

    if selected_confidence != "All":
        filtered_df = filtered_df[filtered_df["confidence"] == selected_confidence]

    filtered_df = filtered_df[
        (filtered_df["avg_elixir"] >= selected_elixir[0]) &
        (filtered_df["avg_elixir"] <= selected_elixir[1])
    ]

    if filtered_df.empty:
        st.info("No decks matched the current filters.")
        return

    if sort_by == "Most Played":
        filtered_df = filtered_df.sort_values(["matches_played", "win_rate"], ascending=[False, False])
    elif sort_by == "Highest Win Rate":
        filtered_df = filtered_df.sort_values(["win_rate", "matches_played"], ascending=[False, False])
    elif sort_by == "Lowest Elixir":
        filtered_df = filtered_df.sort_values(["avg_elixir", "matches_played"], ascending=[True, False])
    elif sort_by == "Highest Elixir":
        filtered_df = filtered_df.sort_values(["avg_elixir", "matches_played"], ascending=[False, False])
    elif sort_by == "Best Cycle":
        filtered_df = filtered_df.sort_values(["cycle_cost", "win_rate"], ascending=[True, False])

    filtered_df = filtered_df.head(top_n).reset_index(drop=True)

    total_matches = int(filtered_df["matches_played"].sum())
    total_wins = int(filtered_df["wins"].sum())
    total_losses = int(filtered_df["losses"].sum())
    avg_wr = float(filtered_df["win_rate"].mean())

    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;'>"
        f"<span class='summary-chip'>📊 {len(filtered_df)} Decks</span>"
        f"<span class='summary-chip'>🎮 {total_matches:,} Matches</span>"
        f"<span class='summary-chip'>✅ {total_wins:,} Wins</span>"
        f"<span class='summary-chip'>❌ {total_losses:,} Losses</span>"
        f"<span class='summary-chip'>⚖️ Avg Win Rate: {avg_wr:.1f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    for idx, (_, row) in enumerate(filtered_df.iterrows(), start=1):
        render_deck_card(
            row=row,
            icon_map=icon_map,
            name_map=name_map,
            rank_num=idx,
        )


if __name__ == "__main__":
    main()