"""
Trends Analysis Page
====================
Analyze trends in card usage, win rates, and meta shifts over time.
"""

import streamlit as st

from utils.ui_helpers import inject_fonts

st.set_page_config(page_title="Trends", layout="wide")
inject_fonts()

st.title("📈 Trends Analysis")

st.write("Track how the meta game evolves and which cards/decks are gaining or losing popularity.")

st.divider()

st.subheader("Card Usage Trends")

selected_card_trend = st.selectbox("Select card to track:", 
    ["Hog Rider", "Golem", "X-Bow", "Balloon", "Mega Knight", "Ice Spirit", "Miner"])

col_trend1, col_trend2 = st.columns(2)

with col_trend1:
    st.write(f"Usage Rate Over Time: {selected_card_trend}")
    st.info("Placeholder: Line chart showing usage percentage trend over last 30 days")

with col_trend2:
    st.write(f"Win Rate Over Time: {selected_card_trend}")
    st.success("Placeholder: Win rate trend (may differ from usage trend)")

st.divider()

st.subheader("Archetype Popularity")

col_arch1, col_arch2 = st.columns(2)

with col_arch1:
    st.write("Archetype Distribution Over Time")
    st.info("Placeholder: Stacked area chart showing how much each archetype dominates the meta")

with col_arch2:
    st.write("Archetype Win Rates")
    st.success("Placeholder: Box plot showing win rate distribution by archetype over time")

st.divider()

st.subheader("Meta Timeline")

st.write("Placeholder: Key events and meta shifts timeline")
st.warning("""
Notable events:
- Oct 3: X-Bow usage spikes (+15%)
- Oct 5: Mega Knight nerfed, usage drops
- Oct 8: Cycle decks dominate ladder
- Oct 10: Beatdown resurgence begins
- Oct 12: Current meta snapshot
""")

st.divider()

st.subheader("Rising & Falling Cards")

col_rising, col_falling = st.columns(2)

with col_rising:
    st.write("🚀 Rising Cards (Gaining Usage)")
    st.info("Placeholder: Cards gaining popularity in last 7 days")

with col_falling:
    st.write("📉 Falling Cards (Losing Usage)")
    st.warning("Placeholder: Cards losing popularity in last 7 days")
