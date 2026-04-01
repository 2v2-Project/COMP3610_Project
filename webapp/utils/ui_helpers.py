"""
Reusable Streamlit UI components shared across pages.
Provides consistent styling and interaction patterns.
"""

import streamlit as st
from typing import List, Optional, Tuple
import pandas as pd


def render_page_header(title: str, subtitle: Optional[str] = None, icon: Optional[str] = None) -> None:
    """
    Render a consistent page header with title and optional subtitle.
    
    Args:
        title: Main page title
        subtitle: Optional subtitle/description
        icon: Optional emoji or icon for the title
    """
    if icon:
        st.title(f"{icon} {title}")
    else:
        st.title(title)
    
    if subtitle:
        st.write(subtitle)
    
    st.divider()


def deck_selector(
    card_options: List[str],
    key: Optional[str] = None,
    max_cards: int = 8,
    title: str = "Select cards for your deck"
) -> List[str]:
    """
    Interactive deck card selector.
    
    Args:
        card_options: List of card names to choose from
        key: Streamlit key for the widget (for managing state)
        max_cards: Maximum number of cards to select (default 8)
        title: Label for the selector
        
    Returns:
        List of selected card names
    """
    st.subheader(title)
    
    selected_cards = st.multiselect(
        "Choose your 8 cards:",
        options=card_options,
        max_selections=max_cards,
        key=key
    )
    
    st.write(f"Selected: {len(selected_cards)}/{max_cards} cards")
    
    if len(selected_cards) == max_cards:
        st.success(f"✓ Deck complete!")
    
    return selected_cards


def display_prediction_result(
    probability: float,
    confidence: Optional[float] = None,
    title: str = "Prediction Result"
) -> None:
    """
    Display prediction result using Streamlit metrics.
    
    Args:
        probability: Win probability (0-1)
        confidence: Optional confidence level (0-1)
        title: Title for the metrics section
    """
    st.subheader(title)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Win Probability",
            f"{probability * 100:.1f}%",
            delta=f"{(probability - 0.5) * 100:.1f}%" if probability != 0.5 else None
        )
    
    with col2:
        opponent_prob = 1.0 - probability
        st.metric("Opponent Probability", f"{opponent_prob * 100:.1f}%")
    
    if confidence is not None:
        with col3:
            st.metric("Confidence", f"{confidence * 100:.1f}%")
    
    # Visual indicator
    if probability > 0.6:
        st.success("✓ Favorable matchup!")
    elif probability < 0.4:
        st.warning("⚠️ Difficult matchup")
    else:
        st.info("⚖️ Balanced matchup")


def display_deck(deck_names: List[str], title: str = "Selected Deck") -> None:
    """
    Display a deck in a nice format.
    
    Args:
        deck_names: List of card names in the deck
        title: Title for the deck display
    """
    st.subheader(title)
    
    if not deck_names:
        st.info("No cards selected")
        return
    
    # Display as columns
    cols = st.columns(len(deck_names))
    for col, card_name in zip(cols, deck_names):
        with col:
            st.write(f"🎴 {card_name}")
    
    st.write(f"**Total: {len(deck_names)} cards**")


def comparison_metric(
    label: str,
    value1: float,
    value2: float,
    value1_label: str = "Player",
    value2_label: str = "Opponent"
) -> None:
    """
    Display a comparison metric with two values.
    
    Args:
        label: Metric label
        value1: First value
        value2: Second value
        value1_label: Label for first value
        value2_label: Label for second value
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{label} - {value1_label}", f"{value1:.2f}")
    with col2:
        st.metric(f"{label} - {value2_label}", f"{value2:.2f}")
