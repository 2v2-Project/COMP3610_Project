"""
Prediction utilities for model inference.
Wraps model prediction logic for reuse across pages.
"""

import pandas as pd
from typing import Dict, Any, Optional, Union, List
import numpy as np


def predict_win_probability(model: Any, feature_df: pd.DataFrame) -> float:
    """
    Predict win probability using a trained model.
    
    Args:
        model: Trained model with predict or predict_proba method
        feature_df: Feature DataFrame with shape (1, n_features)
        
    Returns:
        Win probability as float between 0 and 1
        
    Raises:
        ValueError: If feature_df has wrong shape
    """
    if feature_df.shape[0] != 1:
        raise ValueError(f"Expected 1 sample, got {feature_df.shape[0]}")
    
    # Try predict_proba first (for probabilistic models)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(feature_df)[0]
        # Assume positive class is at index 1
        return float(proba[1]) if len(proba) > 1 else float(proba[0])
    
    # Fall back to predict
    elif hasattr(model, 'predict'):
        pred = model.predict(feature_df)[0]
        # If prediction is binary (0/1), treat as class label
        if pred in [0, 1]:
            return float(pred)
        # If prediction is continuous, assume it's already a probability
        return float(pred)
    
    else:
        raise ValueError("Model must have either predict_proba or predict method")


def predict_matchup(
    model: Any,
    player_deck: Union[List[str], List[int]],
    metadata_df: pd.DataFrame,
    feature_schema: Union[List[str], Dict[str, Any]],
    opponent_deck: Optional[Union[List[str], List[int]]] = None
) -> Dict[str, Any]:
    """
    Predict matchup outcome for a player deck vs opponent deck.
    
    Internally calls preprocessing utilities to build feature vector.
    
    Args:
        model: Trained model for prediction
        player_deck: List of 8 player card names or IDs
        metadata_df: Card metadata
        feature_schema: Feature schema
        opponent_deck: Optional list of 8 opponent card names or IDs
        
    Returns:
        Dictionary with keys:
        - "win_probability": float between 0 and 1
        - "formatted_probability": string like "63.0%"
        - "opponent_win_probability": 1 - player win_prob
        
    Raises:
        ValueError: If deck inputs are invalid
    """
    from .preprocess import build_feature_vector
    
    # Build feature vector
    feature_df = build_feature_vector(
        deck_cards=player_deck,
        metadata_df=metadata_df,
        feature_schema=feature_schema,
        opponent_cards=opponent_deck,
        prefix="player"
    )
    
    # Predict
    player_win_prob = predict_win_probability(model, feature_df)
    opponent_win_prob = 1.0 - player_win_prob
    
    return {
        "win_probability": player_win_prob,
        "formatted_probability": f"{player_win_prob * 100:.1f}%",
        "opponent_win_probability": opponent_win_prob,
        "formatted_opponent_probability": f"{opponent_win_prob * 100:.1f}%",
    }


def format_probability(prob: float) -> str:
    """
    Format probability as percentage string.
    
    Args:
        prob: Probability as float between 0 and 1
        
    Returns:
        Formatted string like "63.0%" or "63.2%"
    """
    if not (0 <= prob <= 1):
        raise ValueError(f"Probability must be between 0 and 1, got {prob}")
    
    return f"{prob * 100:.1f}%"
