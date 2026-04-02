"""
Matchup Analysis Page
=====================
Detailed analysis of card and deck matchups.
"""

import streamlit as st

from utils.ui_helpers import inject_fonts

st.set_page_config(page_title="Matchup Analysis", layout="wide")
inject_fonts()

st.title("⚔️ Matchup Analysis")

st.write("Analyze how different cards and decks perform against each other.")

st.divider()

st.subheader("Card Matchups")

col_card1, col_card2 = st.columns(2)

with col_card1:
    card1 = st.selectbox("Select Card 1:", ["Hog Rider", "Golem", "X-Bow", "Balloon", "Select a card..."])

with col_card2:
    card2 = st.selectbox("Select Card 2:", ["Ice Spirit", "Goblin Barrel", "Miner", "Sparky", "Select a card..."])

if st.button("Calculate Matchup"):
    st.write(f"Placeholder: Detailed matchup analysis between {card1} and {card2}")
    st.info("Shows head-to-head statistics, effective counters, and synergies.")

st.divider()

st.subheader("Deck Matchup Matrix")

col_matrix1, col_matrix2 = st.columns(2)

with col_matrix1:
    archetype1 = st.selectbox("Player 1 Archetype:", 
        ["Beatdown", "Cycle", "Siege", "Control", "Bait", "Bridge Spam", "Graveyard", "Miner Control"])

with col_matrix2:
    archetype2 = st.selectbox("Player 2 Archetype:", 
        ["Beatdown", "Cycle", "Siege", "Control", "Bait", "Bridge Spam", "Graveyard", "Miner Control"])

st.write(f"Placeholder: Heatmap showing {archetype1} vs {archetype2} matchup statistics")
st.success("Based on 1,000+ historical matches")

st.divider()

st.subheader("Counter Analysis")

selected_card = st.selectbox("Choose a card to find counters:", 
    ["Hog Rider", "Golem", "X-Bow", "Balloon", "Mega Knight", "Sparky"])

col_counter1, col_counter2 = st.columns(2)

with col_counter1:
    st.write(f"Best Counters to {selected_card}:")
    st.info("Placeholder: List of cards with highest win rate against selected card")

with col_counter2:
    st.write(f"Supports for {selected_card}:")
    st.info("Placeholder: List of cards that synergize well with selected card")
