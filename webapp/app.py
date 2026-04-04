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
# Page-level CSS (Intercom-inspired hero + card layout)
# ------------------------------------------------------------------
st.markdown("""
<style>
section.main > div { max-width: 1140px; margin: auto; }

.hero-section {
    text-align: center;
    padding: 48px 20px 32px;
}
.hero-section h1 {
    font-size: 42px !important;
    line-height: 1.15;
    margin-bottom: 16px;
}
.hero-subtitle {
    color: #5a7394;
    font-size: 17px;
    max-width: 640px;
    margin: 0 auto 32px;
    line-height: 1.6;
}

.stat-pill-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 14px;
    margin-bottom: 40px;
}
.stat-pill {
    background: #ffffff;
    border: 1px solid #d0dbe8;
    border-radius: 999px;
    padding: 10px 22px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 1px 4px rgba(26,86,219,0.06);
}
.stat-pill .stat-value {
    font-weight: 700;
    color: #1a3a6e;
    font-size: 16px;
}
.stat-pill .stat-label {
    color: #6b7fa3;
    font-size: 13px;
}

.feature-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    justify-content: center;
    margin-bottom: 36px;
}
.feature-card {
    background: #ffffff;
    border: 1px solid #d0dbe8;
    border-radius: 16px;
    padding: 24px 22px 20px;
    box-shadow: 0 2px 8px rgba(26,86,219,0.06);
    transition: transform 0.15s, box-shadow 0.15s;
    width: 320px;
    max-width: 100%;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(26,86,219,0.12);
}
.feature-card .card-icon {
    font-size: 28px;
    margin-bottom: 10px;
}
.feature-card .card-title {
    font-weight: 700;
    font-size: 16px;
    color: #1a3a6e;
    margin-bottom: 6px;
}
.feature-card .card-desc {
    color: #5a7394;
    font-size: 13.5px;
    line-height: 1.55;
}

.section-heading {
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    color: #1e3a5f;
    margin: 40px 0 8px;
}
.section-subheading {
    text-align: center;
    color: #5a7394;
    font-size: 14px;
    margin-bottom: 28px;
}

.info-banner {
    background: #ffffff;
    border: 1px solid #d0dbe8;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(26,86,219,0.06);
    color: #3b536e;
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 24px;
}
.info-banner strong { color: #1a3a6e; }

.quick-start-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    justify-content: center;
    margin-bottom: 36px;
}
.qs-card {
    flex: 1 1 calc(50% - 8px);
    min-width: 280px;
    max-width: 540px;
}
.qs-card {
    background: #ffffff;
    border: 1px solid #d0dbe8;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(26,86,219,0.06);
}
.qs-card .qs-title {
    font-weight: 700;
    color: #1a3a6e;
    font-size: 14px;
    margin-bottom: 6px;
}
.qs-card .qs-text {
    color: #5a7394;
    font-size: 13px;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Real metrics from the dataset
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
# Hero section
# ------------------------------------------------------------------
st.markdown("""
<div class="hero-section">
    <h1>Clash Royale Analytics Engine</h1>
    <div class="hero-subtitle">
        Deep-dive into deck performance, card meta-game analysis, and match outcome
        predictions &mdash; all powered by machine learning on real ladder data.
    </div>
</div>
""", unsafe_allow_html=True)

# Stat pills row
st.markdown(f"""
<div class="stat-pill-row">
    <div class="stat-pill">
        <span class="stat-value">{m['total_matches']:,}</span>
        <span class="stat-label">Matches Analysed</span>
    </div>
    <div class="stat-pill">
        <span class="stat-value">{m['unique_cards']:,}</span>
        <span class="stat-label">Unique Cards</span>
    </div>
    <div class="stat-pill">
        <span class="stat-value">{m['unique_players']:,}</span>
        <span class="stat-label">Unique Players</span>
    </div>
    <div class="stat-pill">
        <span class="stat-value">{m['avg_trophies']:,.0f}</span>
        <span class="stat-label">Avg Trophies</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Feature cards (page directory)
# ------------------------------------------------------------------
st.markdown('<div class="section-heading">Explore the Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheading">Everything you need to master the Clash Royale meta</div>', unsafe_allow_html=True)

features = [
    ("📊", "Overview", "Dashboard summary with key statistics, trophy distributions, and dataset health at a glance."),
    ("🏆", "Popular Decks", "Browse the most-played decks, filter by archetype, confidence, and elixir cost."),
    ("🎯", "Win Predictor", "Build a deck and get an ML-powered win probability estimate with explanations."),
    ("⚔️", "Matchup Analysis", "Card and deck matchup deep-dives to find your deck's strengths and weaknesses."),
    ("📈", "Trends", "Time-series analysis of meta shifts, card popularity, and win-rate evolution."),
    ("🎨", "Archetype Insights", "Archetype vs archetype win-rate heatmaps, usage stats, and strategic breakdowns."),
    ("🎲", "Game Theory", "Payoff matrices, Nash equilibrium computation, and strategic reasoning tools."),
    ("💡", "Recommendations", "Personalised deck suggestions based on your playstyle and meta position."),
]

cards_html = '<div class="feature-grid">'
for icon, title, desc in features:
    cards_html += f"""
    <div class="feature-card">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-desc">{desc}</div>
    </div>"""
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Quick start + dataset banner
# ------------------------------------------------------------------
st.markdown('<div class="section-heading">Quick Start</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheading">Not sure where to begin? Follow these paths</div>', unsafe_allow_html=True)

st.markdown("""
<div class="quick-start-grid">
    <div class="qs-card">
        <div class="qs-title">🆕 First Time Here</div>
        <div class="qs-text">Start with <strong>Overview</strong> to understand the dataset, then browse <strong>Popular Decks</strong> to see what's meta.</div>
    </div>
    <div class="qs-card">
        <div class="qs-title">🧪 Test My Deck</div>
        <div class="qs-text">Jump straight to <strong>Win Predictor</strong> — build your 8-card deck and get an instant win-rate estimate.</div>
    </div>
    <div class="qs-card">
        <div class="qs-title">📊 Competitive Edge</div>
        <div class="qs-text">Use <strong>Matchup Analysis</strong> and <strong>Game Theory</strong> for deep strategic planning.</div>
    </div>
    <div class="qs-card">
        <div class="qs-title">🔍 Meta Research</div>
        <div class="qs-text">Check <strong>Trends</strong> and <strong>Archetype Insights</strong> to understand how the meta is shifting.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="info-banner">
    <strong>Data Source:</strong> Clash Royale Games Dataset (Kaggle)<br>
    <strong>Time Period:</strong> October 2 &ndash; 12, 2023<br>
    <strong>Match Type:</strong> Ladder matches (4,000+ trophies)<br>
    <strong>Features:</strong> Deck composition, elixir analysis, archetypes, synergies, ML predictions
</div>
""", unsafe_allow_html=True)