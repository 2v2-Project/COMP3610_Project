"""
Reusable Streamlit UI components shared across pages.
Provides consistent styling and interaction patterns.
"""

import streamlit as st
from typing import List, Optional, Tuple
import pandas as pd

# ------------------------------------------------------------------
# Global font + blue-white theme injection
# ------------------------------------------------------------------
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

@font-face {
    font-family: 'Supercell-Magic';
    src: url('https://fonts.cdnfonts.com/s/127237/supercell-magic.woff') format('woff');
    font-weight: normal;
    font-style: normal;
    font-display: swap;
}

/* ---- Fonts ---- */
html, body, .stMarkdown, .stText, p, li, label,
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"],
div[data-testid="stCaptionContainer"],
.stSelectbox label, .stSlider label, .stTextInput label,
div[data-testid="stMarkdownContainer"],
div[data-testid="stText"],
div[data-testid="stCaptionContainer"],
.stAlert, .stTabs, .stExpander,
input, select, textarea, button {
    font-family: 'Poppins', sans-serif !important;
}

/* Preserve Material Symbols icons everywhere — MUST come AFTER the Poppins rule */
.material-symbols-rounded,
.material-symbols-outlined,
span[class*="material-symbols"],
button[data-testid="stSidebarCollapseButton"] span,
[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="baseButton-headerNoPadding"] span {
    font-family: 'Material Symbols Rounded' !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
}

/* Supercell-Magic ONLY on page titles (h1 from st.title) */
h1,
[data-testid="stHeading"] h1 {
    font-family: 'Supercell-Magic', 'Poppins', sans-serif !important;
    color: #1a56db !important;
}

/* ---- Blue & White Theme ---- */

/* Main background */
.stApp, section.main {
    background-color: #f0f4fa !important;
}

.stApp {
    background-image: url('/app/static/background.png') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(220, 230, 245, 0.82);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    z-index: 0;
    pointer-events: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3a6e 0%, #1e4d8c 100%) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
[data-testid="stSidebarNavItems"] span {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
    background-color: rgba(255, 255, 255, 0.12) !important;
}

/* Headings (h2, h3) */
h2, h3,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {
    color: #1e3a5f !important;
}

/* Body text */
p, li, label, .stMarkdown {
    color: #2d3748 !important;
}

/* Metric tiles */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #d0dbe8 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(26, 86, 219, 0.08) !important;
    padding: 12px 16px !important;
}
div[data-testid="stMetric"] label {
    color: #6b7fa3 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #1a3a6e !important;
}
div[data-testid="stMetricDelta"] {
    color: #2563eb !important;
}

/* Buttons — primary */
button[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {
    background-color: #1a56db !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #1347b8 !important;
}

/* Buttons — secondary */
.stButton > button {
    border: 1px solid #1a56db !important;
    color: #1a56db !important;
    background-color: #ffffff !important;
    border-radius: 8px !important;
}
.stButton > button:hover {
    background-color: #e8f0fe !important;
}

/* Inputs, selects, sliders */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stTextInput > div > div {
    background-color: #ffffff !important;
    border-color: #c5d5ea !important;
    color: #2d3748 !important;
    border-radius: 8px !important;
}

/* Dividers */
hr, [data-testid="stDivider"] {
    border-color: #c5d5ea !important;
}

/* Info / success / warning boxes */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #6b7fa3 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1a56db !important;
    border-bottom-color: #1a56db !important;
}

/* Captions */
[data-testid="stCaptionContainer"] {
    color: #8395a7 !important;
}

/* Top header / deploy bar */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Expander */
details[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #d0dbe8 !important;
    border-radius: 8px !important;
}

/* Dataframes / tables */
.stDataFrame, .stTable {
    background-color: #ffffff !important;
    border-radius: 8px !important;
}
</style>
"""


def inject_fonts():
    """Inject Supercell-Magic title font, Poppins body font, and blue-white theme. Call once per page."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


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
