"""
Data loading utilities for the Streamlit app.
Centralizes loading of datasets, metadata, and other data files with caching.
"""

import logging
import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Optional, Union
import duckdb
import requests

logger = logging.getLogger(__name__)

# ── HuggingFace dataset config ───────────────────────────────────
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/lillyem/clash-royale-data"
    "/resolve/main/clash_royale_clean.parquet"
)
CLEAN_PARQUET_PATH = Path("data/processed/clash_royale_clean.parquet")


def _ensure_httpfs() -> None:
    """Install and load DuckDB httpfs extension (idempotent)."""
    duckdb.sql("INSTALL httpfs; LOAD httpfs;")


def get_clean_parquet_source() -> str:
    """
    Return the path/URL for clash_royale_clean.parquet.
    Uses the local file if present, otherwise returns the HuggingFace URL
    so DuckDB / Polars can read it directly over HTTPS.
    """
    if CLEAN_PARQUET_PATH.exists():
        return str(CLEAN_PARQUET_PATH)
    _ensure_httpfs()
    return HF_PARQUET_URL


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


@st.cache_data
def load_card_rankings(path: Union[str, Path] = "data/processed/card_rankings.parquet") -> Optional[pd.DataFrame]:
    """Load StatsRoyale card rankings from parquet, with CSV fallback."""
    parquet = Path(path)
    csv_fallback = parquet.with_suffix(".csv")

    if parquet.exists():
        df = pd.read_parquet(parquet)
        logger.info("Loaded card rankings: %d cards from %s", len(df), parquet)
        return df

    if csv_fallback.exists():
        df = pd.read_csv(csv_fallback)
        logger.info("Loaded card rankings (CSV fallback): %d cards from %s", len(df), csv_fallback)
        return df

    logger.warning("Card rankings not found at %s or %s", parquet, csv_fallback)
    return None
