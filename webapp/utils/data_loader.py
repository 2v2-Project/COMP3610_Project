"""
Data loading utilities for the Streamlit app.
Centralizes loading of datasets, metadata, and other data files with caching.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Optional, Union


@st.cache_data
def load_card_metadata(path: Union[str, Path] = "data/processed/card_metadata.csv") -> pd.DataFrame:
    """
    Load card metadata from CSV.
    
    Args:
        path: Path to card metadata CSV file
        
    Returns:
        DataFrame with columns: card_id, name, elixir, rarity, type, icon_url
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Card metadata not found at {path}")
    
    df = pd.read_csv(path)
    
    # Ensure critical columns exist
    required_cols = {"card_id", "name"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Card metadata must contain columns: {required_cols}")
    
    return df


@st.cache_data
def load_final_dataset(path: Union[str, Path] = "data/processed/final_ml_dataset.parquet") -> pd.DataFrame:
    """
    Load the final ML-ready dataset from parquet.
    
    Args:
        path: Path to final ML dataset parquet file
        
    Returns:
        DataFrame with all engineered features and target
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Final dataset not found at {path}")
    
    return pd.read_parquet(path)


@st.cache_data
def load_csv_if_exists(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV file if it exists.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame if file exists, None otherwise
    """
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_parquet_if_exists(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Safely load a parquet file if it exists.
    
    Args:
        path: Path to parquet file
        
    Returns:
        DataFrame if file exists, None otherwise
    """
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_archetype_stats(path: Union[str, Path] = "data/processed/archetype_stats.csv") -> Optional[pd.DataFrame]:
    """
    Load archetype statistics if available.
    
    Args:
        path: Path to archetype stats CSV
        
    Returns:
        DataFrame with archetype performance data, or None if not found
    """
    return load_csv_if_exists(path)


@st.cache_data
def load_popular_decks(path: Union[str, Path] = "data/processed/popular_decks.csv") -> Optional[pd.DataFrame]:
    """
    Load popular decks data if available.
    
    Args:
        path: Path to popular decks CSV
        
    Returns:
        DataFrame with popular decks and stats, or None if not found
    """
    return load_csv_if_exists(path)


@st.cache_data
def load_historical_trends(path: Union[str, Path] = "data/processed/historical_trends.csv") -> Optional[pd.DataFrame]:
    """
    Load historical trends data if available.
    
    Args:
        path: Path to historical trends CSV
        
    Returns:
        DataFrame with time-series trend data, or None if not found
    """
    return load_csv_if_exists(path)
