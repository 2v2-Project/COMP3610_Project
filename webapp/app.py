import streamlit as st

st.set_page_config(
    page_title="Clash Royale Analytics Engine",
    layout="wide"
)

st.title("🎴 Clash Royale Analytics Engine")

st.write("""
Welcome to the comprehensive Clash Royale analytics dashboard! This application provides deep insights 
into deck performance, card meta-game analysis, and match outcome predictions using machine learning.
""")

st.divider()

st.subheader("📚 Available Pages")

col1, col2 = st.columns(2)

with col1:
    st.write("""
    **Core Analytics:**
    
    1. 📊 **Overview** - Dashboard summary and key statistics
    2. 🏆 **Popular Decks** - Most played and highest win-rate decks
    3. 🎯 **Win Predictor** - ML-powered match outcome predictions
    4. ⚔️ **Matchup Analysis** - Card and deck matchup deep dives
    5. 📈 **Trends** - Time-series analysis of meta shifts
    """)

with col2:
    st.write("""
    **Advanced Features:**
    
    6. 🎨 **Archetype Insights** - Beatdown, Cycle, Siege, Control analysis
    7. 💡 **Recommendations** - Personalized deck suggestions
    """)

st.divider()

st.subheader("🚀 Quick Start")

st.write("""
**Getting Started:**
- Start with **Overview** to understand the dataset and key metrics
- Explore **Popular Decks** to see what's meta
- Try **Win Predictor** to test your deck matchups
- Check **Trends** to see how the meta is evolving
- Read **Recommendations** for personalized suggestions

**For Advanced Users:**
- Dive into **Archetype Insights** for strategic analysis
- Use **Matchup Analysis** for competitive strategy planning
""")

st.divider()

st.subheader("📊 Dataset Information")

col_info1, col_info2, col_info3, col_info4 = st.columns(4)

with col_info1:
    st.metric("Total Matches", "1,234,567", "Oct 2-12, 2023")

with col_info2:
    st.metric("Unique Cards", "107", "Current Balance")

with col_info3:
    st.metric("Unique Decks", "15,234", "In Dataset")

with col_info4:
    st.metric("Features Engineered", "87", "ML Variables")

st.info("""
**Data Source:** Clash Royale Games Dataset (Kaggle)  
**Time Period:** October 2-12, 2023  
**Match Type:** Ladder matches (4000+ trophies)  
**Features:** Deck composition, elixir analysis, archetypes, synergies
""")

st.divider()

st.subheader("💡 Tips")

st.write("""
- Use the **sidebar** to navigate between pages (click the arrow icon if sidebar is closed)
- Each page includes interactive filters and visualizations
- Recommendations are personalized based on your profile and goals
- Model predictions are updated with the latest match data
""")

st.success("✅ Navigation is ready! Use the sidebar to explore all pages.")