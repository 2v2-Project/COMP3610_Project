"""
Overview Page — Task 25
=======================
Data-driven dashboard showing key metrics from the Clash Royale dataset:
  • Total matches, win/loss distribution
  • Top cards and archetypes
  • Elixir and cycle statistics

Uses DuckDB to query parquet files directly — no need to load 12 M+ rows
into pandas memory.
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from webapp/utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="Overview", layout="wide")

from utils.ui_helpers import inject_fonts
inject_fonts()

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_DIR = Path("data/processed")
CLEAN = str(DATA_DIR / "clash_royale_clean.parquet")
ARCH = str(DATA_DIR / "archetype_features.parquet")
ELIXIR = str(DATA_DIR / "deck_elixir_features.parquet")


# ------------------------------------------------------------------
# DuckDB helper — cached per query
# ------------------------------------------------------------------
@st.cache_data
def query(sql: str) -> pd.DataFrame:
    return duckdb.sql(sql).df()


# ------------------------------------------------------------------
# Card metadata (tiny CSV — fine to load fully)
# ------------------------------------------------------------------
@st.cache_data
def load_card_metadata():
    return pd.read_csv(DATA_DIR / "card_metadata.csv")


card_meta = load_card_metadata()
card_name_map: dict[int, str] = dict(
    zip(card_meta["card_id"], card_meta["card_name"])
)

# ------------------------------------------------------------------
# Key metrics (aggregated in DuckDB — returns 1 row)
# ------------------------------------------------------------------
metrics = query(f"""
    SELECT
        count(*)                         AS total_matches,
        sum(target_win)                  AS p1_wins,
        avg("player1.trophies")          AS avg_p1_trophies,
        avg("player2.trophies")          AS avg_p2_trophies
    FROM '{CLEAN}'
""").iloc[0]

total_matches = int(metrics["total_matches"])
p1_wins = int(metrics["p1_wins"])
p2_wins = total_matches - p1_wins
win_rate = p1_wins / total_matches * 100
avg_trophies = (metrics["avg_p1_trophies"] + metrics["avg_p2_trophies"]) / 2

# Unique cards — UNPIVOT the 16 card columns
unique_cards = query(f"""
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
    SELECT count(DISTINCT cid) AS n FROM all_cards
""").iloc[0]["n"]

unique_players = query(f"""
    WITH tags AS (
        SELECT "player1.tag" AS tag FROM '{CLEAN}'
        UNION
        SELECT "player2.tag" FROM '{CLEAN}'
    )
    SELECT count(*) AS n FROM tags
""").iloc[0]["n"]

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("📊 Overview Analytics")
st.caption("Real-time summary computed from the full Clash Royale dataset.")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Matches", f"{total_matches:,}")
col2.metric("Unique Cards", f"{int(unique_cards):,}")
col3.metric("Unique Players", f"{int(unique_players):,}")
col4.metric("Avg Trophies", f"{avg_trophies:,.0f}")

st.divider()

# ------------------------------------------------------------------
# Row 1 — Win/Loss Distribution  |  Crown Distribution
# ------------------------------------------------------------------
st.subheader("Win / Loss Distribution")

r1c1, r1c2 = st.columns(2)

with r1c1:
    win_loss_df = pd.DataFrame({
        "Outcome": ["Player 1 Win", "Player 2 Win"],
        "Count": [p1_wins, p2_wins],
    })
    fig_wl = px.pie(
        win_loss_df,
        names="Outcome",
        values="Count",
        color="Outcome",
        color_discrete_map={
            "Player 1 Win": "#1a56db",
            "Player 2 Win": "#93b5f5",
        },
        hole=0.4,
    )
    fig_wl.update_layout(
        margin=dict(t=30, b=10),
        legend=dict(orientation="h", y=-0.1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e3a5f",
    )
    st.plotly_chart(fig_wl, use_container_width=True)
    st.caption(
        f"Player 1 wins **{win_rate:.2f}%** of matches — a slight second-mover "
        f"advantage exists (Player 2 sees Player 1's move timing)."
    )

with r1c2:
    crown_df = query(f"""
        SELECT "player1.crowns"::VARCHAR AS "Crowns",
               count(*) AS "Matches"
        FROM '{CLEAN}'
        GROUP BY "player1.crowns"
        ORDER BY "player1.crowns"
    """)
    fig_cr = px.bar(
        crown_df,
        x="Crowns",
        y="Matches",
        color="Crowns",
        color_discrete_sequence=["#a3c4f3", "#5b9cf5", "#1a56db", "#0e3a8c"],
        text_auto=True,
    )
    fig_cr.update_layout(
        title="Player 1 Crown Distribution",
        showlegend=False,
        margin=dict(t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e3a5f",
    )
    st.plotly_chart(fig_cr, use_container_width=True)
    st.caption(
        "How many crowns Player 1 earns per match. A 3-crown result means a full tower "
        "destruction, while 0 crowns means Player 1 lost without taking a tower."
    )

st.divider()

# ------------------------------------------------------------------
# Row 2 — Top 10 Most Used Cards
# ------------------------------------------------------------------
st.subheader("Top 10 Most Used Cards")

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
    LIMIT 10
""")
total_card_slots = total_matches * 16
card_usage["card_name"] = card_usage["card_id"].map(card_name_map).fillna(
    card_usage["card_id"].astype(str)
)
card_usage["usage_pct"] = card_usage["usage_count"] / total_card_slots * 100

fig_cards = px.bar(
    card_usage.iloc[::-1],
    x="usage_pct",
    y="card_name",
    orientation="h",
    text=card_usage.iloc[::-1]["usage_pct"].apply(lambda v: f"{v:.2f}%"),
    color="usage_pct",
    color_continuous_scale=[[0, "#a3c4f3"], [1, "#1a56db"]],
)
fig_cards.update_layout(
    xaxis_title="Usage (%)",
    yaxis_title="",
    coloraxis_showscale=False,
    margin=dict(t=10, b=10),
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#1e3a5f",
)
st.plotly_chart(fig_cards, use_container_width=True)
st.caption(
    "The 10 most frequently used cards across all player decks. Usage percentage "
    "is calculated from total deck slots (matches × 16). These cards define the "
    "core meta — they appear in many different deck archetypes and trophy ranges."
)

st.divider()

# ------------------------------------------------------------------
# Row 3 — Archetype Distribution  |  Archetype Win Rates
# ------------------------------------------------------------------
st.subheader("Archetype Breakdown")

r3c1, r3c2 = st.columns(2)

with r3c1:
    player_arch = query(f"""
        SELECT player_archetype AS "Archetype",
               count(*) AS "Count"
        FROM '{ARCH}'
        GROUP BY player_archetype
        ORDER BY "Count" DESC
    """)
    player_arch["Pct"] = player_arch["Count"] / player_arch["Count"].sum() * 100

    fig_arch = px.bar(
        player_arch.sort_values("Count"),
        x="Count",
        y="Archetype",
        orientation="h",
        text=player_arch.sort_values("Count")["Pct"].apply(lambda v: f"{v:.1f}%"),
        color="Count",
        color_continuous_scale=[[0, "#a3c4f3"], [1, "#1a56db"]],
    )
    fig_arch.update_layout(
        title="Player Archetype Distribution",
        coloraxis_showscale=False,
        margin=dict(t=40, b=10),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e3a5f",
    )
    st.plotly_chart(fig_arch, use_container_width=True)
    st.caption(
        "Breakdown of deck archetypes detected from card composition. "
        "'Unknown' decks don't match a standard pattern. "
        "Beatdown and Control are the most popular strategies on ladder."
    )

with r3c2:
    # Archetype win rates via DuckDB — join archetype + clean on row number
    arch_wr = query(f"""
        SELECT a.player_archetype AS "Archetype",
               avg(c.target_win) * 100 AS "Win Rate",
               count(*) AS "Matches"
        FROM '{ARCH}' a
        POSITIONAL JOIN '{CLEAN}' c
        GROUP BY a.player_archetype
        ORDER BY "Win Rate"
    """)

    fig_awr = px.bar(
        arch_wr,
        x="Win Rate",
        y="Archetype",
        orientation="h",
        text=arch_wr["Win Rate"].apply(lambda v: f"{v:.1f}%"),
        color="Win Rate",
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#a3c4f3"], [1, "#16a34a"]],
        range_color=[40, 60],
    )
    fig_awr.update_layout(
        title="Archetype Win Rates",
        coloraxis_showscale=False,
        margin=dict(t=40, b=10),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e3a5f",
    )
    fig_awr.add_vline(x=50, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_awr, use_container_width=True)
    st.caption(
        "Win rate per archetype — the dashed line is 50%% (break-even). "
        "Archetypes above the line over-perform during this period. "
        "High win rate with low popularity can indicate a skilled-player niche."
    )

st.divider()

# ------------------------------------------------------------------
# Row 4 — Elixir & Cycle Stats
# ------------------------------------------------------------------
st.subheader("Elixir & Cycle Statistics")

# Sample for histograms + pre-compute stats
elixir_sample = query(f"""
    SELECT player_avg_elixir, player_cycle_cards
    FROM '{ELIXIR}'
    USING SAMPLE 200000
""")

elixir_stats = query(f"""
    SELECT
        avg(player_avg_elixir)     AS mean_elixir,
        median(player_avg_elixir)  AS median_elixir,
        avg(player_troop_count)    AS avg_troops,
        avg(player_spell_count)    AS avg_spells,
        avg(player_building_count) AS avg_buildings,
        avg(player_cycle_cards)    AS avg_cycle
    FROM '{ELIXIR}'
""").iloc[0]

r4c1, r4c2 = st.columns(2)

with r4c1:
    fig_elixir = px.histogram(
        elixir_sample,
        x="player_avg_elixir",
        nbins=40,
        color_discrete_sequence=["#1a56db"],
        labels={"player_avg_elixir": "Average Elixir Cost"},
    )
    fig_elixir.update_layout(
        title="Average Elixir Cost Distribution",
        yaxis_title="Number of Decks",
        margin=dict(t=40, b=10),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e3a5f",
    )
    mean_elixir = float(elixir_stats["mean_elixir"])
    fig_elixir.add_vline(
        x=mean_elixir, line_dash="dash", line_color="#0e3a8c",
        annotation_text=f"Mean: {mean_elixir:.2f}",
    )
    st.plotly_chart(fig_elixir, use_container_width=True)
    st.caption(
        "Distribution of average elixir cost across decks. Most competitive "
        "decks fall between 3.0 and 4.0 — lower means fast cycle, higher means heavy beatdown."
    )

with r4c2:
    fig_cycle = px.histogram(
        elixir_sample,
        x="player_cycle_cards",
        nbins=9,
        color_discrete_sequence=["#5b9cf5"],
        labels={"player_cycle_cards": "Cycle Cards (elixir \u2264 2)"},
    )
    fig_cycle.update_layout(
        title="Cycle Card Count Distribution",
        yaxis_title="Number of Decks",
        margin=dict(t=40, b=10),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e3a5f",
    )
    mean_cycle = float(elixir_stats["avg_cycle"])
    fig_cycle.add_vline(
        x=mean_cycle, line_dash="dash", line_color="#0e3a8c",
        annotation_text=f"Mean: {mean_cycle:.1f}",
    )
    st.plotly_chart(fig_cycle, use_container_width=True)
    st.caption(
        "Number of cheap cards (elixir ≤ 2) per deck. More cycle cards means "
        "faster rotations — cycle decks typically have 3–4 of these."
    )

# Elixir summary metrics
e1, e2, e3, e4, e5, e6 = st.columns(6)
e1.metric("Mean Elixir", f"{elixir_stats['mean_elixir']:.2f}")
e2.metric("Median Elixir", f"{elixir_stats['median_elixir']:.2f}")
e3.metric("Avg Troops", f"{elixir_stats['avg_troops']:.1f}")
e4.metric("Avg Spells", f"{elixir_stats['avg_spells']:.1f}")
e5.metric("Avg Buildings", f"{elixir_stats['avg_buildings']:.1f}")
e6.metric("Avg Cycle Cards", f"{elixir_stats['avg_cycle']:.1f}")

st.divider()

# ------------------------------------------------------------------
# Row 5 — Card Type Composition  |  Trophy Distribution
# ------------------------------------------------------------------
st.subheader("Deck Composition & Trophy Range")

r5c1, r5c2 = st.columns(2)

with r5c1:
    type_means = {
        "Troops": float(elixir_stats["avg_troops"]),
        "Spells": float(elixir_stats["avg_spells"]),
        "Buildings": float(elixir_stats["avg_buildings"]),
    }
    type_df = pd.DataFrame(
        list(type_means.items()), columns=["Card Type", "Avg Count"]
    )
    fig_type = px.pie(
        type_df,
        names="Card Type",
        values="Avg Count",
        color="Card Type",
        color_discrete_map={
            "Troops": "#1a56db",
            "Spells": "#5b9cf5",
            "Buildings": "#a3c4f3",
        },
        hole=0.35,
    )
    fig_type.update_layout(
        title="Average Deck Composition (Card Types)",
        margin=dict(t=40, b=10),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e3a5f",
    )
    st.plotly_chart(fig_type, use_container_width=True)
    st.caption(
        "Average split of card types in a deck. Troops dominate since "
        "they are the primary win condition and defensive units; most decks "
        "carry 2–3 spells and 0–1 buildings."
    )

with r5c2:
    trophy_sample = query(f"""
        SELECT "player1.trophies" AS value
        FROM '{CLEAN}'
        USING SAMPLE 200000
    """)
    fig_trophies = px.histogram(
        trophy_sample,
        nbins=50,
        color_discrete_sequence=["#1a56db"],
        labels={"value": "Trophies"},
    )
    fig_trophies.update_layout(
        title="Player Trophy Distribution",
        xaxis_title="Trophies",
        yaxis_title="Players",
        margin=dict(t=40, b=10),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e3a5f",
    )
    st.plotly_chart(fig_trophies, use_container_width=True)
    st.caption(
        "Trophy distribution of players in the dataset. The data only includes "
        "ladder matches above 4 000 trophies, so the distribution is right-skewed "
        "toward mid-to-high ladder."
    )

st.divider()

# ------------------------------------------------------------------
# Footer — Dataset info
# ------------------------------------------------------------------
st.subheader("📋 Dataset Information")

st.info(f"""
- **Time Period**: October 2–12, 2023  
- **Total Matches**: {total_matches:,}  
- **Unique Players**: {unique_players:,}  
- **Unique Cards**: {unique_cards}  
- **Data Source**: Clash Royale Games Dataset (Kaggle)  
- **Match Type**: Ladder matches (4 000+ trophies)  
- **Features Engineered**: deck composition, elixir analysis, archetypes, synergies, matchup diffs
""")
