"""
Popular Decks Page
==================
Analyze the most commonly used and successful deck compositions.
"""

import streamlit as st

st.set_page_config(page_title="Popular Decks", layout="wide")

st.title("🏆 Popular Decks")

st.write("Explore the most frequently used deck compositions and their performance metrics.")

# Tabs for different metrics
tab1, tab2, tab3 = st.tabs(["Most Played", "Highest Win Rate", "Meta Analysis"])

with tab1:
    st.subheader("Most Played Decks")
    st.write("Placeholder: Display top 10 most commonly played decks with usage percentages.")
    st.info("This section will show frequency-ranked decks with player count and trend indicators.")

with tab2:
    st.subheader("Decks with Best Win Rates")
    st.write("Placeholder: Display decks with highest win rates (min 100 games).")
    st.success("Filter by minimum games played to ensure statistical relevance.")

with tab3:
    st.subheader("Meta Snapshot")
    st.write("Placeholder: Current meta overview showing dominant archetypes and strategies.")
    st.warning("Meta data is updated daily based on latest match data.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Deck Distribution by Archetype")
    st.write("Placeholder: Pie/bar chart showing breakdown of decks by archetype (Beatdown, Cycle, Siege, Control, etc.)")

with col2:
    st.subheader("Average Elixir Cost Distribution")
    st.write("Placeholder: Histogram of average elixir costs across popular decks.")

st.divider()

st.subheader("Deck Details")
selected_deck = st.selectbox("Choose a deck to analyze:", 
    ["Hog Rider + Ice Spirit", "Golem + Night Witch", "X-Bow Cycle", "Placeholder Deck 1", "Placeholder Deck 2"])
st.write(f"Detailed analysis for: {selected_deck}")
st.info("Placeholder: Show cards, win rate, matchup matrix, and trends for selected deck.")
