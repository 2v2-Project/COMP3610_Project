"""
SHAP explainability utilities for interpretable model predictions.
Provides functions to compute and visualize SHAP values.
"""

from typing import Any, Optional, List, Tuple
import numpy as np
import pandas as pd


def get_shap_explainer(model: Any):
    """
    Create a SHAP explainer for a model.
    
    Supports TreeExplainer for tree-based models (XGBoost, Random Forest).
    Falls back to generic KernelExplainer if needed.
    
    Args:
        model: Trained model object
        
    Returns:
        SHAP explainer object
        
    Raises:
        ImportError: If shap is not installed
        ValueError: If model type is not supported
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Run: pip install shap")
    
    # Check if model is tree-based
    model_type = type(model).__name__.lower()
    
    if "xgboost" in model_type or "randomforest" in model_type:
        try:
            explainer = shap.TreeExplainer(model)
            return explainer
        except Exception:
            pass
    
    # Fall back to generic explainer
    raise ValueError(f"Model type {type(model).__name__} not supported for SHAP explanation")


def compute_shap_values(model: Any, feature_df: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for predictions.
    
    Args:
        model: Trained model
        feature_df: Feature DataFrame
        
    Returns:
        Array of SHAP values with shape (n_samples, n_features)
        
    Raises:
        ImportError: If shap is not installed
        ValueError: If SHAP computation fails
    """
    try:
        explainer = get_shap_explainer(model)
        shap_values = explainer.shap_values(feature_df)
        
        # Handle classifier output (may return array for each class)
        if isinstance(shap_values, list):
            # Binary classification: use positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        return shap_values
    
    except Exception as e:
        raise ValueError(f"Failed to compute SHAP values: {str(e)}")


def get_top_shap_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Extract top N most important features based on mean absolute SHAP value.
    
    Args:
        shap_values: SHAP values array, shape (n_samples, n_features)
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        List of (feature_name, mean_abs_shap_value) tuples, sorted descending
        
    Raises:
        ValueError: If dimensions don't match
    """
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP values have {shap_values.shape[1]} features but "
            f"{len(feature_names)} feature names provided"
        )
    
    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get top-n indices
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    
    # Return as list of tuples
    result = [
        (feature_names[i], float(mean_abs_shap[i]))
        for i in top_indices
    ]
    
    return result
