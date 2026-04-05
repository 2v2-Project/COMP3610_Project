"""
Meta Insights Page
==================
Displays card meta rankings sourced from StatsRoyale, including
top-card usage rates, a bar chart of the most-used cards, and a
deck meta-strength analyser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow imports from webapp/utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="Meta Insights", layout="wide")

from utils.ui_helpers import inject_fonts
from utils.metadata import get_card_metadata, get_icon_urls, get_card_names
from utils.data_loader import load_card_rankings

inject_fonts()

# ── Page CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
section.main > div { max-width: 1200px; margin: auto; }

div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #d0dbe8;
    padding: 10px 14px;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(26,86,219,0.08);
}
div[data-testid="stMetric"] label {
    color: #6b7fa3 !important; font-size: 12px !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #1a3a6e !important; font-size: 18px !important;
}
.section-label {
    color: #6b7fa3; font-size: 13px; font-weight: 700;
    margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em;
}
.meta-card {
    background: #ffffff;
    border: 1px solid #d0dbe8;
    border-radius: 12px;
    padding: 12px 8px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(26,86,219,0.06);
    margin-bottom: 8px;
}
.meta-card img {
    width: 64px; height: 64px; object-fit: contain;
    border-radius: 8px; margin-bottom: 4px;
}
.meta-card .card-rank {
    font-size: 11px; color: #6b7fa3; font-weight: 600;
}
.meta-card .card-name {
    font-size: 13px; color: #1a3a6e; font-weight: 700;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.meta-card .card-usage {
    font-size: 12px; color: #1a56db; font-weight: 600;
}
/* Hero card spotlight */
.hero-card {
    background: linear-gradient(135deg, rgba(26,58,110,0.55) 0%, rgba(26,86,219,0.45) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(26,86,219,0.25);
    border-radius: 18px;
    padding: 32px 24px 28px;
    text-align: center;
    max-width: 340px;
    margin: 20px auto 28px;
    box-shadow: 0 8px 32px rgba(26,86,219,0.18);
}
.hero-card img {
    width: 160px; height: 160px; object-fit: contain;
    border-radius: 14px;
    margin-bottom: 12px;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.3));
}
.hero-card .hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: #fbbf24;
    font-size: 12px; font-weight: 700;
    padding: 3px 12px; border-radius: 20px;
    margin-bottom: 8px; letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-card .hero-name {
    font-size: 26px; color: #ffffff; font-weight: 800;
    margin-bottom: 4px;
}
.hero-card .hero-usage {
    font-size: 18px; color: #93c5fd; font-weight: 600;
}
.note-box {
    background: #ffffff; border: 1px solid #d0dbe8; border-radius: 10px;
    padding: 14px 16px; color: #3b536e; font-size: 14px;
    box-shadow: 0 1px 3px rgba(26,86,219,0.06);
}
</style>
""", unsafe_allow_html=True)


# ── Data ────────────────────────────────────────────────────────────
@st.cache_data
def load_rankings() -> pd.DataFrame | None:
    df = load_card_rankings()
    if df is not None and not df.empty:
        return df.sort_values("rank").reset_index(drop=True)
    return None


rankings = load_rankings()

if rankings is None or rankings.empty:
    st.title("\U0001f525 Meta Insights")
    st.error(
        "Card rankings data is missing. Run the ingestion script first:\n\n"
        "```\npython scr/14_ingest_statsroyale_rankings.py\n```"
    )
    st.stop()


# Card metadata for icons
card_meta = get_card_metadata()
icon_urls = get_icon_urls()
name_to_id: dict[str, int] = {}
for cid, cname in get_card_names().items():
    name_to_id[cname] = cid

# Map icon URLs onto rankings by matching card_name -> card_id -> icon_url
rankings["icon_url"] = rankings["card_name"].map(
    lambda n: icon_urls.get(name_to_id.get(n, -1), "")
)

# ── Title ───────────────────────────────────────────────────────────
st.title("\U0001f525 Meta Insights")
st.markdown(
    "<div style='margin-bottom:10px;color:#5a7394;font-size:15px;'>"
    "Card rankings based on usage rates across <strong>12.4 M+</strong> ladder matches, "
    "sourced from <strong>StatsRoyale</strong>."
    "</div>",
    unsafe_allow_html=True,
)

# ── Section 1: Metrics ─────────────────────────────────────────────
st.markdown("<div class='section-label'>Key Metrics</div>", unsafe_allow_html=True)

top_card = rankings.iloc[0]
highest_usage = rankings["usage_rate"].max()
total_cards = len(rankings)

c1, c2, c3 = st.columns(3)
c1.metric("Top Card (Rank 1)", top_card["card_name"])
c2.metric("Highest Usage Rate", f"{highest_usage:.1f}%")
c3.metric("Total Cards Tracked", total_cards)

st.divider()

# ── Section 2: Top Card Spotlight ───────────────────────────────────
st.markdown("<div class='section-label'>Most Used Card</div>", unsafe_allow_html=True)

hero = rankings.iloc[0]
hero_icon = hero.get("icon_url", "")
hero_img = (
    f"<img src='{hero_icon}' alt='{hero['card_name']}'/>"
    if hero_icon
    else f"<div style='height:160px;display:flex;align-items:center;"
         f"justify-content:center;font-size:18px;color:#93c5fd;'>"
         f"{hero['card_name']}</div>"
)
st.markdown(
    f"<div class='hero-card'>"
    f"<div class='hero-badge'>\U0001f451 #1 Most Used</div><br>"
    f"{hero_img}"
    f"<div class='hero-name'>{hero['card_name']}</div>"
    f"<div class='hero-usage'>{hero['usage_rate']:.1f}% usage rate</div>"
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()

# ── Section 3: Top 25 Card Grid (cards #2–#25) ─────────────────────
st.markdown("<div class='section-label'>Top 25 Cards</div>", unsafe_allow_html=True)

top25 = rankings.iloc[1:25]
COLS_PER_ROW = 5

for row_start in range(0, len(top25), COLS_PER_ROW):
    chunk = top25.iloc[row_start:row_start + COLS_PER_ROW]
    cols = st.columns(COLS_PER_ROW, gap="small")
    for i, (_, row) in enumerate(chunk.iterrows()):
        with cols[i]:
            icon = row.get("icon_url", "")
            img_html = (
                f"<img src='{icon}' alt='{row['card_name']}'/>"
                if icon
                else f"<div style='height:64px;display:flex;align-items:center;"
                     f"justify-content:center;font-size:11px;color:#6b7fa3;'>"
                     f"{row['card_name']}</div>"
            )
            st.markdown(
                f"<div class='meta-card'>"
                f"{img_html}"
                f"<div class='card-rank'>#{int(row['rank'])}</div>"
                f"<div class='card-name'>{row['card_name']}</div>"
                f"<div class='card-usage'>{row['usage_rate']:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.divider()

# ── Section 4: Bar Chart — Top 10 by Usage Rate ────────────────────
st.markdown("<div class='section-label'>Top 10 Cards by Usage Rate</div>", unsafe_allow_html=True)

top10 = rankings.head(10).copy()

fig = px.bar(
    top10,
    x="card_name",
    y="usage_rate",
    text=top10["usage_rate"].apply(lambda v: f"{v:.1f}%"),
    labels={"card_name": "Card", "usage_rate": "Usage Rate (%)"},
    color="usage_rate",
    color_continuous_scale=["#93c5fd", "#1a56db"],
)
fig.update_layout(
    plot_bgcolor="#f8fbff",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1a3a6e", family="Poppins, sans-serif"),
    height=420,
    margin=dict(l=40, r=20, t=20, b=60),
    xaxis=dict(tickangle=-30),
    coloraxis_showscale=False,
    showlegend=False,
)
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**What this shows:** The 10 most popular cards across all ladder matches.
Cards with higher usage rates appear in more decks and define the current meta.
""")
