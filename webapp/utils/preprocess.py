"""
Preprocessing utilities for converting deck inputs to feature vectors.
Handles deck normalization, feature engineering, and schema alignment.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Set
import numpy as np


def normalize_card_input(
    deck_cards: Union[List[str], List[int]],
    metadata_df: pd.DataFrame
) -> List[int]:
    """
    Normalize card input to card IDs.
    
    Supports input as either card names or card IDs.
    If names are provided, lookup in metadata.
    
    Args:
        deck_cards: List of card names (str) or card IDs (int)
        metadata_df: DataFrame with at least columns: card_id, name
        
    Returns:
        List of card IDs (int)
        
    Raises:
        ValueError: If card not found in metadata or duplicate cards
    """
    card_ids = []
    
    for card in deck_cards:
        if isinstance(card, int):
            card_id = card
        else:
            # Look up name in metadata
            matches = metadata_df[metadata_df['name'].str.lower() == str(card).lower()]
            if len(matches) == 0:
                raise ValueError(f"Card '{card}' not found in metadata")
            card_id = int(matches.iloc[0]['card_id'])
        
        card_ids.append(card_id)
    
    # Check for duplicates
    if len(card_ids) != len(set(card_ids)):
        raise ValueError("Deck contains duplicate cards")
    
    return card_ids


def validate_deck_size(deck_card_ids: List[int], expected_size: int = 8) -> None:
    """
    Validate that deck has the correct number of cards.
    
    Args:
        deck_card_ids: List of card IDs
        expected_size: Expected number of cards (default 8)
        
    Raises:
        ValueError: If deck size is incorrect
    """
    if len(deck_card_ids) != expected_size:
        raise ValueError(f"Deck must have exactly {expected_size} cards, got {len(deck_card_ids)}")


def build_deck_one_hot(
    deck_card_ids: List[int],
    all_card_ids: List[int],
    prefix: str = "card"
) -> pd.DataFrame:
    """
    Build one-hot encoded features for a deck.
    
    Args:
        deck_card_ids: List of card IDs in the deck
        all_card_ids: Sorted list of all card IDs in the schema
        prefix: Prefix for column names (e.g., "card", "player_card", "opp_card")
        
    Returns:
        DataFrame with one-hot encoded card columns
    """
    validate_deck_size(deck_card_ids, expected_size=8)
    
    # Create one-hot vector
    one_hot = {f"{prefix}_{cid}": (1 if cid in deck_card_ids else 0) for cid in all_card_ids}
    
    return pd.DataFrame([one_hot])


def compute_deck_summary_features(
    deck_card_ids: List[int],
    metadata_df: pd.DataFrame,
    prefix: str = "player"
) -> Dict[str, float]:
    """
    Compute summary features for a deck (elixir, type distribution, etc.).
    
    Args:
        deck_card_ids: List of card IDs in the deck
        metadata_df: DataFrame with columns: card_id, elixir, type
        prefix: Prefix for output column names
        
    Returns:
        Dictionary of summary feature values
    """
    validate_deck_size(deck_card_ids, expected_size=8)
    
    # Get card details
    card_data = metadata_df[metadata_df['card_id'].isin(deck_card_ids)]
    
    features = {}
    
    # Elixir features
    if 'elixir' in metadata_df.columns:
        elixirs = card_data['elixir'].astype(float)
        features[f"{prefix}_avg_elixir"] = float(elixirs.mean())
        features[f"{prefix}_low_cost_cards"] = int((elixirs <= 3).sum())
        features[f"{prefix}_medium_cost_cards"] = int((elixirs == 4).sum())
        features[f"{prefix}_high_cost_cards"] = int((elixirs >= 5).sum())
        features[f"{prefix}_cycle_cards"] = int((elixirs <= 2).sum())
        features[f"{prefix}_cycle_ratio"] = float(features[f"{prefix}_cycle_cards"] / 8.0)
    
    # Card type features
    if 'type' in metadata_df.columns:
        types = card_data['type'].str.lower()
        features[f"{prefix}_troop_count"] = int((types == "troop").sum())
        features[f"{prefix}_spell_count"] = int((types == "spell").sum())
        features[f"{prefix}_building_count"] = int((types == "building").sum())
    
    return features


def align_to_schema(
    feature_df: pd.DataFrame,
    feature_schema: Union[List[str], Dict[str, any]]
) -> pd.DataFrame:
    """
    Align feature DataFrame to exact schema: drop extra columns, add missing columns with 0.
    
    Args:
        feature_df: DataFrame with constructed features
        feature_schema: List of column names or dict with schema definition
        
    Returns:
        DataFrame with exactly the columns in schema, in correct order
    """
    # Extract column names from schema
    if isinstance(feature_schema, list):
        schema_columns = feature_schema
    elif isinstance(feature_schema, dict) and "columns" in feature_schema:
        schema_columns = feature_schema["columns"]
    else:
        schema_columns = list(feature_schema.keys()) if isinstance(feature_schema, dict) else []
    
    # Add missing columns with 0
    for col in schema_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
    
    # Select and reorder to schema
    feature_df = feature_df[schema_columns]
    
    return feature_df


def build_feature_vector(
    deck_cards: Union[List[str], List[int]],
    metadata_df: pd.DataFrame,
    feature_schema: Union[List[str], Dict[str, any]],
    opponent_cards: Optional[Union[List[str], List[int]]] = None,
    prefix: str = "player"
) -> pd.DataFrame:
    """
    Build complete feature vector from deck input(s).
    
    Combines one-hot encoding and summary features into a single feature vector.
    
    Args:
        deck_cards: List of 8 card names or IDs
        metadata_df: Card metadata with columns: card_id, name, elixir, type
        feature_schema: List of expected feature column names or dict with schema
        opponent_cards: Optional list of 8 opponent card names or IDs
        prefix: Prefix for player deck features ("player", "deck", etc.)
        
    Returns:
        DataFrame with shape (1, len(schema)) ready for model prediction
        
    Raises:
        ValueError: If deck sizes are wrong, cards not found, or invalid input
    """
    # Normalize inputs
    deck_card_ids = normalize_card_input(deck_cards, metadata_df)
    validate_deck_size(deck_card_ids, expected_size=8)
    
    # Get all card IDs for one-hot encoding
    all_card_ids = sorted(metadata_df['card_id'].unique().tolist())
    
    # Build features
    features = {}
    
    # One-hot encoding for player deck
    one_hot = build_deck_one_hot(deck_card_ids, all_card_ids, prefix=f"{prefix}_card")
    features.update(one_hot.iloc[0].to_dict())
    
    # Summary features for player deck
    summary = compute_deck_summary_features(deck_card_ids, metadata_df, prefix=prefix)
    features.update(summary)
    
    # Optional: opponent deck features
    if opponent_cards is not None:
        opp_card_ids = normalize_card_input(opponent_cards, metadata_df)
        validate_deck_size(opp_card_ids, expected_size=8)
        
        # One-hot for opponent
        opp_one_hot = build_deck_one_hot(opp_card_ids, all_card_ids, prefix="opp_card")
        features.update(opp_one_hot.iloc[0].to_dict())
        
        # Summary for opponent
        opp_summary = compute_deck_summary_features(opp_card_ids, metadata_df, prefix="opp")
        features.update(opp_summary)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame([features])
    
    # Align to schema
    feature_df = align_to_schema(feature_df, feature_schema)
    
    return feature_df
