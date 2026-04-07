from pathlib import Path
import base64

import duckdb
import streamlit as st

from utils.data_loader import get_clean_parquet_source

st.set_page_config(
    page_title="Clash Royale Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.ui_helpers import inject_fonts
inject_fonts()

# ------------------------------------------------------------------
# Page-level CSS (Esports hero banner + card layout)
# ------------------------------------------------------------------
st.markdown("""
<style>
section.main > div { max-width: 1140px; margin: auto; }

/* ---- Hero banner carousel ---- */
.hero-banner {
    position: relative;
    width: 100%;
    height: 380px;
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.hero-slides {
    position: absolute; inset: 0;
    animation: heroFade 15s infinite;
}
.hero-slides .slide {
    position: absolute; inset: 0;
    background-size: cover;
    background-position: center;
    opacity: 0;
    transition: opacity 1.2s ease-in-out;
}
.hero-slides .slide:nth-child(1) { animation: slide1 15s infinite; }
.hero-slides .slide:nth-child(2) { animation: slide2 15s infinite; }
.hero-slides .slide:nth-child(3) { animation: slide3 15s infinite; }

@keyframes slide1 {
    0%,5%   { opacity:1; }
    33%,38% { opacity:0; }
    95%,100%{ opacity:1; }
}
@keyframes slide2 {
    0%,28%  { opacity:0; }
    33%,38% { opacity:1; }
    66%,71% { opacity:0; }
}
@keyframes slide3 {
    0%,61%  { opacity:0; }
    66%,71% { opacity:1; }
    95%,100%{ opacity:0; }
}

.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(
        135deg,
        rgba(5,8,20,0.90) 0%,
        rgba(10,18,45,0.82) 50%,
        rgba(12,25,55,0.75) 100%
    );
    z-index: 2;
}
.hero-content {
    position: relative;
    z-index: 3;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    padding: 20px 24px;
}
.hero-content h1 {
    font-size: 46px !important;
    font-weight: 800;
    color: #ffffff !important;
    line-height: 1.1;
    margin-bottom: 14px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.7), 0 4px 20px rgba(0,0,0,0.5);
    letter-spacing: -0.5px;
}
.hero-content .hero-accent {
    color: #6db8ff;
    text-shadow: 0 2px 8px rgba(0,0,0,0.7), 0 0 30px rgba(77,163,255,0.25);
}
.hero-tagline {
    color: rgba(255,255,255,0.95);
    font-size: 17px;
    max-width: 620px;
    line-height: 1.65;
    text-shadow: 0 1px 6px rgba(0,0,0,0.6), 0 2px 12px rgba(0,0,0,0.3);
}
.hero-badge {
    display: inline-block;
    margin-top: 18px;
    background: rgba(30,80,160,0.65);
    border: 1px solid rgba(100,180,255,0.50);
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
    padding: 8px 20px;
    border-radius: 999px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    backdrop-filter: blur(6px);
    text-shadow: 0 1px 4px rgba(0,0,0,0.4);
}

/* ---- Indicators ---- */
.hero-indicators {
    position: absolute;
    bottom: 18px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 4;
    display: flex;
    gap: 8px;
}
.hero-indicators span {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: rgba(255,255,255,0.35);
    display: inline-block;
}
.hero-indicators span:nth-child(1) { animation: dot1 15s infinite; }
.hero-indicators span:nth-child(2) { animation: dot2 15s infinite; }
.hero-indicators span:nth-child(3) { animation: dot3 15s infinite; }
@keyframes dot1 {
    0%,5%   { background:rgba(255,255,255,0.9); }
    33%,100%{ background:rgba(255,255,255,0.35); }
    95%     { background:rgba(255,255,255,0.9); }
}
@keyframes dot2 {
    0%,28%  { background:rgba(255,255,255,0.35); }
    33%,38% { background:rgba(255,255,255,0.9); }
    66%,100%{ background:rgba(255,255,255,0.35); }
}
@keyframes dot3 {
    0%,61%  { background:rgba(255,255,255,0.35); }
    66%,71% { background:rgba(255,255,255,0.9); }
    95%,100%{ background:rgba(255,255,255,0.35); }
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
CLEAN = get_clean_parquet_source()


@st.cache_data
def _load_home_metrics() -> dict:
    con = duckdb.connect()
    con.execute("SET memory_limit = '512MB'")

    metrics = con.sql(f"""
        SELECT count(*) AS total_matches,
               sum(target_win) AS p1_wins,
               avg("player1.trophies") AS avg_p1,
               avg("player2.trophies") AS avg_p2
        FROM '{CLEAN}'
    """).df().iloc[0]

    unique_cards = int(con.sql(f"""
        WITH sample AS (SELECT * FROM '{CLEAN}' USING SAMPLE 500000),
        ac AS (
            SELECT "player1.card1" c FROM sample UNION ALL SELECT "player1.card2" FROM sample
            UNION ALL SELECT "player1.card3" FROM sample UNION ALL SELECT "player1.card4" FROM sample
            UNION ALL SELECT "player1.card5" FROM sample UNION ALL SELECT "player1.card6" FROM sample
            UNION ALL SELECT "player1.card7" FROM sample UNION ALL SELECT "player1.card8" FROM sample
            UNION ALL SELECT "player2.card1" FROM sample UNION ALL SELECT "player2.card2" FROM sample
            UNION ALL SELECT "player2.card3" FROM sample UNION ALL SELECT "player2.card4" FROM sample
            UNION ALL SELECT "player2.card5" FROM sample UNION ALL SELECT "player2.card6" FROM sample
            UNION ALL SELECT "player2.card7" FROM sample UNION ALL SELECT "player2.card8" FROM sample
        ) SELECT count(DISTINCT c) AS n FROM ac
    """).df().iloc[0]["n"])

    unique_players = int(con.sql(f"""
        WITH tags AS (
            SELECT "player1.tag" AS tag FROM '{CLEAN}' USING SAMPLE 500000
            UNION
            SELECT "player2.tag" FROM '{CLEAN}' USING SAMPLE 500000
        ) SELECT count(*) AS n FROM tags
    """).df().iloc[0]["n"])

    con.close()

    return {
        "total_matches": int(metrics["total_matches"]),
        "unique_cards": unique_cards,
        "unique_players": unique_players,
        "avg_trophies": (metrics["avg_p1"] + metrics["avg_p2"]) / 2,
    }


m = _load_home_metrics()

# ------------------------------------------------------------------
# Hero banner
# ------------------------------------------------------------------
def _img_to_data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"

_b1 = _img_to_data_uri("webapp/static/banner1.jpg")
_b2 = _img_to_data_uri("webapp/static/banner2.jpg")
_b3 = _img_to_data_uri("webapp/static/banner3.jpg")

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-slides">
        <div class="slide" style="background-image:url('{_b1}');"></div>
        <div class="slide" style="background-image:url('{_b2}');"></div>
        <div class="slide" style="background-image:url('{_b3}');"></div>
    </div>
    <div class="hero-overlay"></div>
    <div class="hero-content">
        <h1>Clash Royale<br><span class="hero-accent">Analytics Engine</span></h1>
        <div class="hero-tagline">
            Deep-dive into deck performance, card meta-game analysis, and match outcome
            predictions &mdash; all powered by machine learning on real ladder data.
        </div>
        <div class="hero-badge">12.4 M+ Matches Analysed</div>
    </div>
    <div class="hero-indicators">
        <span></span><span></span><span></span>
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