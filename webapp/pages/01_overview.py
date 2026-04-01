"""
Overview Page
=============
High-level dashboard showing key metrics and statistics from the Clash Royale dataset.
"""

import streamlit as st

st.set_page_config(page_title="Overview", layout="wide")

st.title("📊 Overview")

st.write("""
This dashboard provides comprehensive analytics for Clash Royale match data, including:
- Overview of dataset and key statistics
- Popular deck compositions
- Win rate predictions
- Matchup analysis
- Trend analysis
- Archetype insights
- Machine learning model performance
- Personalized recommendations
""")

# Placeholder metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Matches Analyzed", "1,234,567", "+5%")

with col2:
    st.metric("Unique Cards", "107", "Current Set")

with col3:
    st.metric("Unique Decks", "15,234", "In Dataset")

with col4:
    st.metric("Average Win Rate", "50.2%", "±0.1%")

st.divider()

st.subheader("Dataset Summary")
st.info("""
- **Time Period**: October 2-12, 2023
- **Data Source**: Clash Royale Games (Kaggle)
- **Matches**: Ladder matches from 4000+ trophies
- **Features Engineered**: 
  - Deck composition matrices
  - Elixir cost analysis
  - Card synergy indicators
  - Archetype classifications
""")

st.subheader("Quick Links")
col_link1, col_link2, col_link3 = st.columns(3)

with col_link1:
    st.write("📈 [View Popular Decks](/02_popular_decks)")
    
with col_link2:
    st.write("🎯 [Try Win Predictor](/03_win_predictor)")
    
with col_link3:
    st.write("🔍 [Check Trends](/05_trends)")
