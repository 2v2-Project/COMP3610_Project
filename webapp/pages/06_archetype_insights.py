"""
Archetype Insights Page
=======================
Deep dive into deck archetypes and their characteristics.
"""

import streamlit as st

from utils.ui_helpers import inject_fonts

st.set_page_config(page_title="Archetype Insights", layout="wide")
inject_fonts()

st.title("🎨 Archetype Insights")

st.write("Explore different deck archetypes and how they perform in the current meta.")

st.divider()

archetypes = ["Beatdown", "Cycle", "Siege", "Control", "Bait", "Bridge Spam", "Miner Control", "Graveyard", "Unknown"]

selected_archetype = st.selectbox("Choose an archetype:", archetypes)

st.subheader(f"📋 {selected_archetype} Archetype")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Usage Rate", "15.2%", "+2.3%")

with col_info2:
    st.metric("Average Win Rate", "52.1%", "+1.2%")

with col_info3:
    st.metric("Most Common Variant", "Golem + NW", "45% of arch")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Popular Cards", "Matchups", "Examples"])

with tab1:
    st.write(f"Placeholder: Detailed description of {selected_archetype} archetype philosophy")
    st.info("Archetypes are defined by playstyle, win conditions, and key synergies.")

with tab2:
    st.write(f"Most common cards in {selected_archetype}:")
    st.success("Placeholder: Shows core cards, support cards, and spell choices")

with tab3:
    st.write(f"How {selected_archetype} performs against other archetypes:")
    st.info("Placeholder: Matchup table showing win rates vs each archetype")

with tab4:
    st.write(f"Example {selected_archetype} decks:")
    st.write("""
    - Deck 1: [Placeholder]
    - Deck 2: [Placeholder]
    - Deck 3: [Placeholder]
    """)

st.divider()

st.subheader("All Archetypes Comparison")

col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    st.write("Archetype Popularity Distribution")
    st.info("Placeholder: Bar chart showing how many decks use each archetype")

with col_comp2:
    st.write("Archetype Win Rate Comparison")
    st.success("Placeholder: Comparison of average win rates across all archetypes")
