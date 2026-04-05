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
/* Podium layout */
.podium-row {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 18px;
    margin: 20px auto 28px;
    max-width: 720px;
}
.podium-card {
    background: linear-gradient(135deg, rgba(26,58,110,0.55) 0%, rgba(26,86,219,0.45) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(26,86,219,0.25);
    border-radius: 18px;
    padding: 20px 16px 18px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(26,86,219,0.15);
    flex: 1;
    max-width: 220px;
}
.podium-card.gold {
    padding: 28px 20px 24px;
    max-width: 260px;
    box-shadow: 0 8px 32px rgba(26,86,219,0.22);
    border: 1px solid rgba(251,191,36,0.4);
}
.podium-card img {
    object-fit: contain;
    border-radius: 12px;
    margin-bottom: 8px;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.3));
}
.podium-card.gold img  { width: 140px; height: 140px; }
.podium-card.silver img { width: 100px; height: 100px; }
.podium-card.bronze img { width: 100px; height: 100px; }
.podium-badge {
    display: inline-block;
    padding: 3px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.podium-badge.gold-badge   { background: rgba(251,191,36,0.25); color: #fbbf24; }
.podium-badge.silver-badge { background: rgba(192,192,192,0.25); color: #d1d5db; }
.podium-badge.bronze-badge { background: rgba(205,127,50,0.25); color: #d97706; }
.podium-name {
    font-weight: 800; color: #ffffff;
    margin-bottom: 2px;
}
.podium-card.gold .podium-name   { font-size: 22px; }
.podium-card.silver .podium-name { font-size: 17px; }
.podium-card.bronze .podium-name { font-size: 17px; }
.podium-usage {
    font-weight: 600; color: #93c5fd;
}
.podium-card.gold .podium-usage   { font-size: 16px; }
.podium-card.silver .podium-usage { font-size: 13px; }
.podium-card.bronze .podium-usage { font-size: 13px; }
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

# ── Section 2: Top 3 Podium ─────────────────────────────────────────
st.markdown("<div class='section-label'>Top 3 Most Used Cards</div>", unsafe_allow_html=True)

def _podium_img(row):
    icon = row.get("icon_url", "")
    if icon:
        return f"<img src='{icon}' alt='{row['card_name']}'/>"
    return (f"<div style='height:100px;display:flex;align-items:center;"
            f"justify-content:center;font-size:14px;color:#93c5fd;'>"
            f"{row['card_name']}</div>")

podium_cfg = [
    (1, "silver", "\U0001f948 #2", "silver-badge"),  # left
    (0, "gold",   "\U0001f451 #1", "gold-badge"),    # center
    (2, "bronze", "\U0001f949 #3", "bronze-badge"),  # right
]

podium_html = "<div class='podium-row'>"
for idx, tier, badge_text, badge_cls in podium_cfg:
    r = rankings.iloc[idx]
    podium_html += (
        f"<div class='podium-card {tier}'>"
        f"<div class='podium-badge {badge_cls}'>{badge_text}</div><br>"
        f"{_podium_img(r)}"
        f"<div class='podium-name'>{r['card_name']}</div>"
        f"<div class='podium-usage'>{r['usage_rate']:.1f}%</div>"
        f"</div>"
    )
podium_html += "</div>"
st.markdown(podium_html, unsafe_allow_html=True)

st.divider()

# ── Section 3: Top 25 Card Grid (cards #4–#25) ─────────────────────
st.markdown("<div class='section-label'>Top 25 Cards</div>", unsafe_allow_html=True)

top25 = rankings.iloc[3:25]
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
