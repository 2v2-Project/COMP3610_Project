"""
Model loading utilities for the Streamlit app.
Centralizes loading of trained models and feature schemas with caching.
"""

import streamlit as st
from pathlib import Path
import joblib
import json
from typing import Dict, Tuple, Optional, Union, Any, List


@st.cache_resource
def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a trained model from disk using joblib.
    
    Args:
        model_path: Path to the model .joblib file
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file does not exist
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return joblib.load(model_path)


@st.cache_resource
def load_logistic_regression_model() -> Any:
    """
    Load the logistic regression model.
    
    Returns:
        Loaded logistic regression model
        
    Raises:
        FileNotFoundError: If model not found
    """
    path = Path("models/logistic_regression.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Logistic regression model not found at {path}")
    return load_model(path)


@st.cache_resource
def load_random_forest_model() -> Any:
    """
    Load the random forest model.
    
    Returns:
        Loaded random forest model
        
    Raises:
        FileNotFoundError: If model not found
    """
    path = Path("models/random_forest.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Random forest model not found at {path}")
    return load_model(path)


@st.cache_resource
def load_xgboost_model() -> Any:
    """
    Load the XGBoost model.
    
    Returns:
        Loaded XGBoost model
        
    Raises:
        FileNotFoundError: If model not found
    """
    path = Path("models/xgboost_model.joblib")
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {path}")
    return load_model(path)


@st.cache_resource
def load_best_model(
    preferred_order: Tuple[str, ...] = ("xgboost", "random_forest", "logistic_regression")
) -> Tuple[Any, str]:
    """
    Load the best available model based on preference order.
    
    Args:
        preferred_order: Tuple of model names in preferred order.
                        Options: "xgboost", "random_forest", "logistic_regression"
    
    Returns:
        Tuple of (loaded_model, model_name)
        
    Raises:
        FileNotFoundError: If no models are available
    """
    model_loaders = {
        "xgboost": load_xgboost_model,
        "random_forest": load_random_forest_model,
        "logistic_regression": load_logistic_regression_model,
    }
    
    for model_name in preferred_order:
        if model_name not in model_loaders:
            continue
        
        try:
            model = model_loaders[model_name]()
            return model, model_name
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError(
        f"No models found in preferred order: {preferred_order}. "
        "Please ensure at least one model file exists in the models/ directory."
    )


def load_feature_schema(path: Union[str, Path] = "models/feature_schema.json") -> Union[List[str], Dict[str, Any]]:
    """
    Load feature schema from JSON file.
    
    Args:
        path: Path to feature schema JSON file
        
    Returns:
        Feature schema (list of column names or dict with schema definition)
        
    Raises:
        FileNotFoundError: If schema file does not exist
        json.JSONDecodeError: If JSON is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature schema not found at {path}")
    
    with open(path, "r") as f:
        schema = json.load(f)
    
    return schema
