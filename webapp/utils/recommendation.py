"""
Deck recommendation utilities.
Ranks candidate decks based on predicted win probability.
"""

from typing import List, Dict, Any, Union, Tuple
import pandas as pd
from .preprocess import build_feature_vector
from .prediction import predict_win_probability


def rank_candidate_decks(
    model: Any,
    candidate_decks: List[List[Union[str, int]]],
    metadata_df: pd.DataFrame,
    feature_schema: Union[List[str], Dict[str, Any]],
    top_k: int = 5
) -> pd.DataFrame:
    """
    Rank a list of candidate decks by predicted win probability.
    
    Args:
        model: Trained prediction model
        candidate_decks: List of decks, each deck is a list of 8 card names or IDs
        metadata_df: Card metadata
        feature_schema: Feature schema
        top_k: Number of top decks to return
        
    Returns:
        DataFrame with columns:
        - deck_index: Original index of deck in candidate_decks
        - cards: Tuple of card names/IDs
        - win_probability: Predicted win probability
        - rank: Rank (1 = best)
    """
    results = []
    
    for idx, deck in enumerate(candidate_decks):
        try:
            feature_df = build_feature_vector(
                deck_cards=deck,
                metadata_df=metadata_df,
                feature_schema=feature_schema,
                prefix="player"
            )
            
            win_prob = predict_win_probability(model, feature_df)
            
            results.append({
                "deck_index": idx,
                "cards": tuple(deck),
                "win_probability": win_prob,
            })
        
        except Exception as e:
            # Skip invalid decks with silent failure for robustness
            continue
    
    # Convert to DataFrame and sort
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("win_probability", ascending=False).reset_index(drop=True)
    df_results["rank"] = range(1, len(df_results) + 1)
    
    # Return top-k
    return df_results.head(top_k)


def recommend_best_decks(
    model: Any,
    candidate_decks: List[List[Union[str, int]]],
    metadata_df: pd.DataFrame,
    feature_schema: Union[List[str], Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Get top recommended decks as a list of dictionaries (easy for Streamlit display).
    
    Args:
        model: Trained model
        candidate_decks: List of candidate decks
        metadata_df: Card metadata
        feature_schema: Feature schema
        top_k: Number of recommendations
        
    Returns:
        List of dicts with keys:
        - rank: int (1-based)
        - cards: list of card names/IDs
        - win_probability: float
        - formatted_probability: string like "63.2%"
    """
    ranked_df = rank_candidate_decks(
        model, candidate_decks, metadata_df, feature_schema, top_k
    )
    
    if ranked_df.empty:
        return []
    
    recommendations = []
    for _, row in ranked_df.iterrows():
        recommendations.append({
            "rank": int(row["rank"]),
            "cards": list(row["cards"]),
            "win_probability": float(row["win_probability"]),
            "formatted_probability": f"{row['win_probability'] * 100:.1f}%",
        })
    
    return recommendations
