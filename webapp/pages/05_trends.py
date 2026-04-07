"""
Historical Trends Page — Task 29
=================================
Visualize strategic and historical patterns:
  • Most used cards chart
  • Daily card usage trends
  • Archetype distribution & win rates
  • Top-performing decks
  • Average elixir distribution
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="Historical Trends", layout="wide")

from utils.ui_helpers import inject_fonts
inject_fonts()

from utils.metadata import get_card_names
from utils.deck_helpers import build_deck_key, enrich_deck_record
from utils.data_loader import get_clean_parquet_source, get_archetype_parquet_source, get_elixir_parquet_source

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_DIR = Path("data/processed")
CLEAN = get_clean_parquet_source()
ARCH = get_archetype_parquet_source()
ELIXIR = get_elixir_parquet_source()

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#1e3a5f",
)

BLUE_SEQ = [[0, "#a3c4f3"], [1, "#1a56db"]]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_data
def query(sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.sql(sql).df()
    finally:
        con.close()


@st.cache_data
def load_card_meta():
    meta = pd.read_csv(DATA_DIR / "card_metadata.csv")
    return dict(zip(meta["card_id"], meta["name"])), dict(zip(meta["card_id"], meta["type"])), dict(zip(meta["card_id"], meta["elixir"]))


card_name_map, card_type_map, card_elixir_map = load_card_meta()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("📈 Historical Trends")
st.caption("Strategic patterns and meta evolution across Oct 2 – 11, 2023.")
st.divider()

# ==================================================================
# 1 — Most Used Cards
# ==================================================================
st.subheader("🃏 Most Used Cards")

top_n_cards = st.slider("Number of cards to show", 10, 50, 20, 5, key="top_cards")

card_usage = query(f"""
    WITH all_cards AS (
        SELECT "player1.card1" AS cid FROM '{CLEAN}'
        UNION ALL SELECT "player1.card2" FROM '{CLEAN}'
        UNION ALL SELECT "player1.card3" FROM '{CLEAN}'
        UNION ALL SELECT "player1.card4" FROM '{CLEAN}'
        UNION ALL SELECT "player1.card5" FROM '{CLEAN}'
        UNION ALL SELECT "player1.card6" FROM '{CLEAN}'
        UNION ALL SELECT "player1.card7" FROM '{CLEAN}'
        UNION ALL SELECT "player1.card8" FROM '{CLEAN}'
        UNION ALL SELECT "player2.card1" FROM '{CLEAN}'
        UNION ALL SELECT "player2.card2" FROM '{CLEAN}'
        UNION ALL SELECT "player2.card3" FROM '{CLEAN}'
        UNION ALL SELECT "player2.card4" FROM '{CLEAN}'
        UNION ALL SELECT "player2.card5" FROM '{CLEAN}'
        UNION ALL SELECT "player2.card6" FROM '{CLEAN}'
        UNION ALL SELECT "player2.card7" FROM '{CLEAN}'
        UNION ALL SELECT "player2.card8" FROM '{CLEAN}'
    )
    SELECT cid AS card_id, count(*) AS usage_count
    FROM all_cards
    GROUP BY cid
    ORDER BY usage_count DESC
""")

total_card_slots = int(query(f"SELECT count(*) AS n FROM '{CLEAN}'").iloc[0]["n"]) * 16
card_usage["card_name"] = card_usage["card_id"].map(card_name_map).fillna(card_usage["card_id"].astype(str))
card_usage["usage_pct"] = card_usage["usage_count"] / total_card_slots * 100
card_usage["card_type"] = card_usage["card_id"].map(card_type_map).fillna("unknown")

top_cards = card_usage.head(top_n_cards).iloc[::-1]

fig_usage = px.bar(
    top_cards,
    x="usage_pct",
    y="card_name",
    orientation="h",
    text=top_cards["usage_pct"].apply(lambda v: f"{v:.2f}%"),
    color="card_type",
    color_discrete_map={"troop": "#1a56db", "spell": "#5b9cf5", "building": "#a3c4f3"},
)
fig_usage.update_layout(
    xaxis_title="Usage (%)",
    yaxis_title="",
    margin=dict(t=10, b=10),
    height=max(400, top_n_cards * 22),
    legend_title_text="Card Type",
    **CHART_LAYOUT,
)
st.plotly_chart(fig_usage, use_container_width=True)
st.markdown("""
The **most popular cards** on ladder. Usage % = how often a card appears across all deck slots.
Cards near the top are **meta staples** that fit into many deck types. Colour shows card type:
troop (attackers/defenders), spell (damage/utility), or building (defensive structures).

Cards outside the top 20 aren't necessarily bad; they may be strong niche picks in specific archetypes.
""")

st.divider()

# ==================================================================
# 2 — Daily Card Usage Trends
# ==================================================================
st.subheader("📊 Daily Card Usage Trends")

top_10_ids = card_usage.head(10)["card_id"].tolist()
top_10_names = [card_name_map.get(cid, str(cid)) for cid in top_10_ids]

selected_cards_for_trend = st.multiselect(
    "Select cards to track (top 10 pre-selected):",
    options=list(card_name_map.values()),
    default=top_10_names[:5],
    max_selections=10,
)

if selected_cards_for_trend:
    reverse_name_map = {v: k for k, v in card_name_map.items()}
    selected_ids = [reverse_name_map[n] for n in selected_cards_for_trend if n in reverse_name_map]

    if selected_ids:
        id_list = ", ".join(str(int(cid)) for cid in selected_ids)
        daily_usage = query(f"""
            WITH daily_matches AS (
                SELECT datetime::DATE AS match_date, count(*) AS day_matches
                FROM '{CLEAN}'
                GROUP BY match_date
            ),
            all_cards AS (
                SELECT datetime::DATE AS match_date, "player1.card1" AS cid FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player1.card2" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player1.card3" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player1.card4" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player1.card5" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player1.card6" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player1.card7" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player1.card8" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player2.card1" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player2.card2" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player2.card3" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player2.card4" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player2.card5" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player2.card6" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player2.card7" FROM '{CLEAN}'
                UNION ALL SELECT datetime::DATE, "player2.card8" FROM '{CLEAN}'
            )
            SELECT a.match_date, a.cid AS card_id,
                   count(*) AS card_count,
                   count(*) * 100.0 / (d.day_matches * 16) AS usage_pct
            FROM all_cards a
            JOIN daily_matches d ON a.match_date = d.match_date
            WHERE a.cid IN ({id_list})
            GROUP BY a.match_date, a.cid, d.day_matches
            ORDER BY a.match_date, card_count DESC
        """)

        daily_usage["card_name"] = daily_usage["card_id"].map(card_name_map)

        fig_daily = px.line(
            daily_usage,
            x="match_date",
            y="usage_pct",
            color="card_name",
            markers=True,
            labels={"match_date": "Date", "usage_pct": "Usage %", "card_name": "Card"},
        )
        fig_daily.update_layout(
            height=420,
            margin=dict(t=10, b=10),
            legend_title_text="Card",
            xaxis=dict(dtick="D1"),
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_daily, use_container_width=True)

st.markdown("""
Track how card popularity shifts day-by-day. **Rising lines** mean a card is gaining traction,
**falling lines** mean it's losing popularity. Select up to 10 cards to compare and spot emerging trends.
""")

st.divider()

# ==================================================================
# 3 — Archetype Distribution & Win Rates
# ==================================================================
st.subheader("🎯 Archetype Distribution & Win Rates")

arch_dist = query(f"""
    SELECT player_archetype AS "Archetype",
           count(*) AS "Count"
    FROM '{ARCH}'
    GROUP BY player_archetype
    ORDER BY "Count" DESC
""")
arch_dist["Pct"] = arch_dist["Count"] / arch_dist["Count"].sum() * 100

col_a1, col_a2 = st.columns(2)

with col_a1:
    fig_arch_bar = px.bar(
        arch_dist.sort_values("Count"),
        x="Count",
        y="Archetype",
        orientation="h",
        text=arch_dist.sort_values("Count")["Pct"].apply(lambda v: f"{v:.1f}%"),
        color="Count",
        color_continuous_scale=BLUE_SEQ,
    )
    fig_arch_bar.update_layout(
        title="Archetype Popularity",
        coloraxis_showscale=False,
        margin=dict(t=40, b=10),
        height=400,
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig_arch_bar, use_container_width=True)
    st.markdown("""
How many matches each archetype appeared in. Archetypes are detected from card composition
(e.g. Golem/Giant = Beatdown, cheap Hog Rider = Cycle). "Unknown" covers creative decks
that don't fit a standard archetype pattern.
    """)

with col_a2:
    arch_wr = query(f"""
        SELECT a.player_archetype AS "Archetype",
               avg(c.target_win) * 100 AS "Win Rate",
               count(*) AS "Matches"
        FROM '{ARCH}' a
        POSITIONAL JOIN '{CLEAN}' c
        GROUP BY a.player_archetype
        ORDER BY "Win Rate" DESC
    """)

    fig_awr = px.bar(
        arch_wr.sort_values("Win Rate"),
        x="Win Rate",
        y="Archetype",
        orientation="h",
        text=arch_wr.sort_values("Win Rate")["Win Rate"].apply(lambda v: f"{v:.1f}%"),
        color="Win Rate",
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#a3c4f3"], [1, "#16a34a"]],
        range_color=[40, 60],
    )
    fig_awr.update_layout(
        title="Archetype Win Rates",
        coloraxis_showscale=False,
        margin=dict(t=40, b=10),
        height=400,
        **CHART_LAYOUT,
    )
    fig_awr.add_vline(x=50, line_dash="dash", line_color="#8395a7")
    st.plotly_chart(fig_awr, use_container_width=True)
    st.markdown("""
Average win rate per archetype. The **dashed line at 50%** is break-even; archetypes to the right win more than they lose.

**Key insight:** popular archetypes often sit near 50% because opponents know how to counter them.
Rare archetypes can have higher win rates because fewer players prepare for them.
The best edge comes from picking an archetype that's both **effective and underrepresented**.
    """)

st.divider()

# ==================================================================
# 4 — Top-Performing Decks
# ==================================================================
st.subheader("🏆 Top-Performing Decks")

min_deck_matches = st.slider("Minimum matches for a deck", 20, 500, 50, 10, key="deck_min")

PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]

@st.cache_data
def build_top_decks(min_matches: int) -> pd.DataFrame:
    """Build top decks entirely in DuckDB for speed — avoids row-by-row Python."""
    raw = query(f"""
        WITH decks AS (
            SELECT
                list_sort(list_value(
                    "player1.card1","player1.card2","player1.card3","player1.card4",
                    "player1.card5","player1.card6","player1.card7","player1.card8"
                ))::VARCHAR AS deck_key,
                CASE WHEN "player1.crowns" > "player2.crowns" THEN 1 ELSE 0 END AS win
            FROM '{CLEAN}'
        )
        SELECT deck_key,
               count(*) AS matches_played,
               sum(win) AS wins,
               round(sum(win) * 100.0 / count(*), 1) AS win_rate
        FROM decks
        GROUP BY deck_key
        HAVING count(*) >= {int(min_matches)}
        ORDER BY win_rate DESC
        LIMIT 50
    """)

    if raw.empty:
        return pd.DataFrame()

    records = []
    for _, row in raw.iterrows():
        try:
            card_ids = [int(x.strip()) for x in row["deck_key"].strip("[]").split(",")]
        except Exception:
            continue
        rec = enrich_deck_record(
            deck_key=build_deck_key(card_ids),
            matches_played=int(row["matches_played"]),
            wins=int(row["wins"]),
            name_map=card_name_map,
            elixir_map=card_elixir_map,
            type_map=card_type_map,
        )
        names = [card_name_map.get(cid, str(cid)) for cid in rec["card_ids"]]
        rec["card_names_str"] = " / ".join(names)
        records.append(rec)

    result = pd.DataFrame(records)
    return result.sort_values("win_rate", ascending=False).reset_index(drop=True)


top_decks = build_top_decks(min_deck_matches)

if top_decks.empty:
    st.info("No decks meet the minimum match threshold. Lower the slider.")
else:
    col_sort = st.radio(
        "Sort decks by:", ["Win Rate", "Matches Played"], horizontal=True, key="deck_sort",
    )
    sort_col = "win_rate" if col_sort == "Win Rate" else "matches_played"
    display_decks = top_decks.sort_values(sort_col, ascending=False).head(15)

    fig_decks = px.bar(
        display_decks.iloc[::-1],
        x="win_rate",
        y="card_names_str",
        orientation="h",
        text=display_decks.iloc[::-1]["win_rate"].apply(lambda v: f"{v:.1f}%"),
        color="matches_played",
        color_continuous_scale=BLUE_SEQ,
        labels={"win_rate": "Win Rate %", "card_names_str": "Deck", "matches_played": "Matches"},
        hover_data=["archetype", "avg_elixir", "matches_played"],
    )
    fig_decks.update_layout(
        height=max(450, len(display_decks) * 32),
        margin=dict(t=10, b=10, l=280),
        yaxis=dict(tickfont=dict(size=10)),
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig_decks, use_container_width=True)

    with st.expander("Deck Details Table"):
        show_df = display_decks[["card_names_str", "archetype", "win_rate", "matches_played", "avg_elixir", "cycle_cost", "troop_count", "spell_count", "building_count"]].copy()
        show_df.columns = ["Deck", "Archetype", "Win Rate %", "Matches", "Avg Elixir", "Cycle Cost", "Troops", "Spells", "Buildings"]
        st.dataframe(show_df.reset_index(drop=True), use_container_width=True)

st.markdown("""
The highest-performing exact 8-card decks, filtered by minimum matches so win rates are reliable.
Decks with a high win rate **and** a dark bar (many matches) are the most trustworthy picks.
A high win rate with few matches could be a fluke. Expand the table for elixir and card-type details.
""")

st.divider()

# ==================================================================
# 5 — Average Elixir Distribution
# ==================================================================
st.subheader("⚗️ Elixir Cost Distribution")

elixir_sample = query(f"""
    SELECT player_avg_elixir FROM '{ELIXIR}' USING SAMPLE 200000
""")

fig_elixir_hist = px.histogram(
    elixir_sample,
    x="player_avg_elixir",
    nbins=40,
    color_discrete_sequence=["#1a56db"],
    labels={"player_avg_elixir": "Average Elixir Cost"},
)
fig_elixir_hist.update_layout(
    yaxis_title="Number of Decks",
    margin=dict(t=10, b=10),
    height=340,
    **CHART_LAYOUT,
)
overall_mean = float(elixir_sample["player_avg_elixir"].mean())
fig_elixir_hist.add_vline(
    x=overall_mean, line_dash="dash", line_color="#0e3a8c",
    annotation_text=f"Mean: {overall_mean:.2f}",
)
st.plotly_chart(fig_elixir_hist, use_container_width=True)
st.markdown("""
Shows **how expensive most decks are** by average elixir cost. The dashed line is the overall average.

- **< 3.0:** Ultra-fast cycle decks. Quick rotations but fragile troops.
- **3.0 to 3.8:** The competitive sweet spot. Balanced offence and defence.
- **3.8 to 4.3:** Heavier decks with strong pushes but slower rotations.
- **> 4.3:** Beatdown territory (Golem, Lava Hound). Big pushes, high commitment.

If your deck's average elixir is far from the peak, you're running a niche strategy.
""")

st.divider()

st.caption("Data: Clash Royale ladder matches, Oct 2 – 11 2023 (4 000+ trophies).")
