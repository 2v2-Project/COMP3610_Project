from pathlib import Path

import duckdb
import streamlit as st

st.set_page_config(
    page_title="Clash Royale Analytics Engine",
    layout="wide",
)

from utils.ui_helpers import inject_fonts
inject_fonts()

# ------------------------------------------------------------------
# Real metrics from the dataset (same DuckDB queries as Overview)
# ------------------------------------------------------------------
DATA_DIR = Path("data/processed")
CLEAN = str(DATA_DIR / "clash_royale_clean.parquet")


@st.cache_data
def _load_home_metrics() -> dict:
    metrics = duckdb.sql(f"""
        SELECT count(*) AS total_matches,
               sum(target_win) AS p1_wins,
               avg("player1.trophies") AS avg_p1,
               avg("player2.trophies") AS avg_p2
        FROM '{CLEAN}'
    """).df().iloc[0]

    unique_cards = int(duckdb.sql(f"""
        WITH ac AS (
            SELECT "player1.card1" c FROM '{CLEAN}' UNION ALL SELECT "player1.card2" FROM '{CLEAN}'
            UNION ALL SELECT "player1.card3" FROM '{CLEAN}' UNION ALL SELECT "player1.card4" FROM '{CLEAN}'
            UNION ALL SELECT "player1.card5" FROM '{CLEAN}' UNION ALL SELECT "player1.card6" FROM '{CLEAN}'
            UNION ALL SELECT "player1.card7" FROM '{CLEAN}' UNION ALL SELECT "player1.card8" FROM '{CLEAN}'
            UNION ALL SELECT "player2.card1" FROM '{CLEAN}' UNION ALL SELECT "player2.card2" FROM '{CLEAN}'
            UNION ALL SELECT "player2.card3" FROM '{CLEAN}' UNION ALL SELECT "player2.card4" FROM '{CLEAN}'
            UNION ALL SELECT "player2.card5" FROM '{CLEAN}' UNION ALL SELECT "player2.card6" FROM '{CLEAN}'
            UNION ALL SELECT "player2.card7" FROM '{CLEAN}' UNION ALL SELECT "player2.card8" FROM '{CLEAN}'
        ) SELECT count(DISTINCT c) AS n FROM ac
    """).df().iloc[0]["n"])

    unique_players = int(duckdb.sql(f"""
        WITH tags AS (
            SELECT "player1.tag" AS tag FROM '{CLEAN}' UNION SELECT "player2.tag" FROM '{CLEAN}'
        ) SELECT count(*) AS n FROM tags
    """).df().iloc[0]["n"])

    return {
        "total_matches": int(metrics["total_matches"]),
        "unique_cards": unique_cards,
        "unique_players": unique_players,
        "avg_trophies": (metrics["avg_p1"] + metrics["avg_p2"]) / 2,
    }


m = _load_home_metrics()

# ------------------------------------------------------------------
# Page content
# ------------------------------------------------------------------
st.title("🎴 Clash Royale Analytics Engine")

st.write("""
Welcome to the comprehensive Clash Royale analytics dashboard! This application provides deep insights 
into deck performance, card meta-game analysis, and match outcome predictions using machine learning.
""")

st.divider()

st.subheader("📚 Available Pages")

col1, col2 = st.columns(2)

with col1:
    st.write("""
    **Core Analytics:**
    
    1. 📊 **Overview** — Dashboard summary and key statistics
    2. 🏆 **Popular Decks** — Most played and highest win-rate decks
    3. 🎯 **Win Predictor** — ML-powered match outcome predictions
    4. ⚔️ **Matchup Analysis** — Card and deck matchup deep dives
    5. 📈 **Trends** — Time-series analysis of meta shifts
    """)

with col2:
    st.write("""
    **Advanced Features:**
    
    6. 🎨 **Archetype Insights** — Beatdown, Cycle, Siege, Control analysis
    7. 🎲 **Game Theory** — Payoff matrices, Nash equilibrium, strategy reasoning
    8. 💡 **Recommendations** — Personalized deck suggestions
    """)

st.divider()

st.subheader("🚀 Quick Start")

st.write("""
**Getting Started:**
- Start with **Overview** to understand the dataset and key metrics
- Explore **Popular Decks** to see what's meta
- Try **Win Predictor** to test your deck matchups
- Check **Trends** to see how the meta is evolving
- Read **Recommendations** for personalized suggestions

**For Advanced Users:**
- Dive into **Archetype Insights** for strategic analysis
- Use **Matchup Analysis** for competitive strategy planning
- Explore **Game Theory** for Nash equilibrium and payoff analysis
""")

st.divider()

st.subheader("📊 Dataset Information")

col_info1, col_info2, col_info3, col_info4 = st.columns(4)

with col_info1:
    st.metric("Total Matches", f"{m['total_matches']:,}", "Oct 2-12, 2023")

with col_info2:
    st.metric("Unique Cards", f"{m['unique_cards']:,}", "In Dataset")

with col_info3:
    st.metric("Unique Players", f"{m['unique_players']:,}", "In Dataset")

with col_info4:
    st.metric("Avg Trophies", f"{m['avg_trophies']:,.0f}", "Ladder 4 000+")

st.info(f"""
**Data Source:** Clash Royale Games Dataset (Kaggle)  
**Time Period:** October 2–12, 2023  
**Match Type:** Ladder matches (4 000+ trophies)  
**Total Matches:** {m['total_matches']:,}  
**Features:** Deck composition, elixir analysis, archetypes, synergies
""")

st.divider()

st.subheader("💡 Tips")

st.write("""
- Use the **sidebar** to navigate between pages (click the arrow icon if sidebar is closed)
- Each page includes interactive filters and visualizations
- Recommendations are personalized based on your profile and goals
- Model predictions are updated with the latest match data
""")

st.success("✅ Navigation is ready! Use the sidebar to explore all pages.")