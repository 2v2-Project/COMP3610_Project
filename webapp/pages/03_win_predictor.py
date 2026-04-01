"""
Win Predictor Page
==================
Predict match outcomes using machine learning models trained on historical data.
"""

import streamlit as st

st.set_page_config(page_title="Win Predictor", layout="wide")

st.title("🎯 Win Predictor")

st.write("Input two player decks to predict the likelihood of victory for Player 1.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Player 1 Deck")
    st.write("Build your deck by selecting 8 cards")
    st.write("Placeholder: Interactive card selector for Player 1")
    st.info("Select exactly 8 unique cards for the deck.")

with col2:
    st.subheader("Player 2 Deck")
    st.write("Build opponent deck by selecting 8 cards")
    st.write("Placeholder: Interactive card selector for Player 2")
    st.info("Select exactly 8 unique cards for the deck.")

st.divider()

col_pred1, col_pred2, col_pred3 = st.columns(3)

with col_pred1:
    if st.button("🔮 Predict Winner"):
        st.write("Placeholder: ML model prediction would go here")
        st.metric("Player 1 Win Probability", "62.3%", "+2.5%")

with col_pred2:
    st.metric("Player 2 Win Probability", "37.7%", "-2.5%")

with col_pred3:
    st.metric("Prediction Confidence", "78%", "High")

st.divider()

st.subheader("Prediction Details")

tab1, tab2, tab3 = st.tabs(["Feature Importance", "Matchup Analysis", "Similar Decks"])

with tab1:
    st.write("Placeholder: Feature importance chart showing which attributes most influenced the prediction")
    st.info("Based on trained Random Forest and XGBoost models")

with tab2:
    st.write("Placeholder: Head-to-head card matchup analysis")
    st.success("Card-level win rates and synergy indicators")

with tab3:
    st.write("Placeholder: Show historical similar deck matchups from the dataset")
    st.warning("Compare with real match outcomes for validation")
