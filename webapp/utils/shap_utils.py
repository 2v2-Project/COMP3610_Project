"""
SHAP Explanation Utilities

Provides local SHAP value extraction for model predictions.
Used by the explanation engine to compute feature-level contributions.
"""

from __future__ import annotations

import logging
from typing import Any, List, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_shap_explainer(model: Any):
    """
    Create a SHAP explainer for the supplied model.

    Supports tree-based models directly. Falls back to generic shap.Explainer
    when possible.
    """
    try:
        import shap
    except ImportError as exc:
        logger.error("SHAP is not installed. Run: pip install shap")
        raise ImportError("SHAP is not installed. Run: pip install shap") from exc

    model_type = type(model).__name__.lower()

    # Fast path for tree-based models
    if any(name in model_type for name in ["xgb", "xgboost", "forest", "tree"]):
        try:
            logger.debug(f"Creating TreeExplainer for model type: {model_type}")
            return shap.TreeExplainer(model)
        except Exception as e:
            logger.debug(f"TreeExplainer failed, falling back to generic: {e}")
            pass

    # Generic fallback
    try:
        logger.debug(f"Creating generic Explainer for model type: {model_type}")
        return shap.Explainer(model)
    except Exception as exc:
        logger.error(f"Could not create SHAP explainer for model type: {type(model).__name__}")
        raise ValueError(
            f"Could not create a SHAP explainer for model type: {type(model).__name__}"
        ) from exc


def _normalize_shap_output(shap_values: Any) -> np.ndarray:
    """
    Normalize SHAP output to shape (n_samples, n_features).

    Handles:
    - list outputs for binary classifiers
    - Explanation objects
    - 3D arrays from some classifiers
    """
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    if isinstance(shap_values, list):
        # Binary classification commonly returns [class0, class1]
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values, dtype=float)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    if shap_values.ndim == 3:
        # e.g. (samples, features, classes)
        shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]

    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

    return shap_values


def compute_shap_values(model: Any, feature_df: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for a single-row or multi-row feature DataFrame.
    
    Raises:
        ValueError: If feature_df is empty
        Exceptions from SHAP library if explainer fails
    """
    if feature_df.empty:
        raise ValueError("feature_df is empty")

    logger.debug(f"Computing SHAP values for feature_df shape: {feature_df.shape}")

    explainer = get_shap_explainer(model)

    try:
        raw_values = explainer(feature_df)
        logger.debug(f"SHAP explainer call succeeded, raw output shape: {np.asarray(raw_values).shape}")
    except Exception as e:
        logger.debug(f"Primary SHAP call failed, trying fallback method: {e}")
        raw_values = explainer.shap_values(feature_df)
        logger.debug(f"Fallback SHAP call succeeded, output shape: {np.asarray(raw_values).shape}")

    normalized = _normalize_shap_output(raw_values)
    logger.debug(f"Normalized SHAP values shape: {normalized.shape}")
    
    return normalized


def get_top_shap_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> List[tuple[str, float]]:
    """
    Return top features by mean absolute SHAP value.
    """
    if shap_values.ndim != 2:
        raise ValueError("shap_values must be 2D")

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP values have {shap_values.shape[1]} columns but feature_names has {len(feature_names)}"
        )

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    return [(feature_names[i], float(mean_abs[i])) for i in top_idx]


def get_local_shap_explanation(
    model: Any,
    feature_df: pd.DataFrame,
    top_n: int = 4,
    min_abs_value: float = 1e-4,
) -> List[Dict[str, float | str]]:
    """
    Return the strongest local SHAP contributions for the first row.
    
    This is the core function used by the explanation engine to extract
    which features most strongly influenced the model's prediction for
    this specific data point.
    
    Args:
        model: Trained ML model
        feature_df: Feature vector (1 row)
        top_n: Number of top features to return
        min_abs_value: Minimum absolute SHAP value to include
        
    Returns:
        List of dicts with keys: feature, feature_value, shap_value, direction
        
    Raises:
        Various exceptions if SHAP computation fails (will be caught by caller)
    """
    logger.debug(f"Computing local SHAP explanation for top_n={top_n}")
    
    shap_values = compute_shap_values(model, feature_df)

    if shap_values.shape[0] == 0:
        logger.warning("SHAP values returned empty array")
        return []

    row_values = shap_values[0]
    ranked_idx = np.argsort(np.abs(row_values))[::-1]

    results: List[Dict[str, float | str]] = []

    for idx in ranked_idx:
        contribution = float(row_values[idx])
        if abs(contribution) < min_abs_value:
            continue

        feature_name = str(feature_df.columns[idx])
        direction = "increased" if contribution > 0 else "decreased"
        
        result_item = {
            "feature": feature_name,
            "feature_value": float(feature_df.iloc[0, idx]),
            "shap_value": contribution,
            "direction": direction,
        }
        results.append(result_item)
        logger.debug(f"Top feature: {feature_name} ({direction}), SHAP value: {contribution:.4f}")

        if len(results) >= top_n:
            break

    logger.debug(f"Found {len(results)} local SHAP features above threshold")
    return results