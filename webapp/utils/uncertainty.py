"""
Model prediction with uncertainty estimation.
Combines predictions from multiple models with uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    best_model_name: str
    final_probability: float
    predicted_class: int
    uncertainty_std: float
    uncertainty_range_low: float
    uncertainty_range_high: float
    confidence_label: str
    model_probabilities: Dict[str, float]


@dataclass
class HybridConfidenceResult:
    label: str
    score: float
    historical_score: float
    model_score: float


def select_best_model(model_scores: dict[str, dict[str, float]], metric: str = "roc_auc") -> str:
    """
    Select the best model based on a specified metric.
    
    Example model_scores:
    {
        "Logistic Regression": {"accuracy": 0.55, "f1": 0.59, "roc_auc": 0.57},
        "Random Forest": {"accuracy": 0.56, "f1": 0.56, "roc_auc": 0.59},
        "XGBoost": {"accuracy": 0.58, "f1": 0.60, "roc_auc": 0.62},
    }
    """
    return max(model_scores, key=lambda name: model_scores[name][metric])


def get_confidence_label(uncertainty_std: float) -> str:
    """
    Convert uncertainty (std of model probabilities) into a readable label.
    Adjust thresholds if needed after observing your model outputs.
    """
    if uncertainty_std < 0.03:
        return "High"
    if uncertainty_std < 0.08:
        return "Medium"
    return "Low"


def _normalize_std_to_confidence(uncertainty_std: float) -> float:
    """
    Convert model disagreement (std) to a 0-1 confidence score.

    A std around 0.25 is treated as very uncertain for binary probabilities.
    """
    return max(0.0, 1.0 - min(uncertainty_std / 0.25, 1.0))


def _label_to_score(label: str) -> float:
    label_key = str(label).strip().lower()
    if label_key in {"very high", "high"}:
        return 1.0
    if label_key == "medium":
        return 0.6
    if label_key in {"moderate"}:
        return 0.5
    return 0.2


def _score_to_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def confidence_from_match_count(
    matches_played: int,
    high_threshold: int = 1000,
    medium_threshold: int = 200,
) -> str:
    """
    Convert historical match support into a confidence label.

    Thresholds are configurable so pages can preserve their existing behavior
    while sharing one central confidence implementation.
    """
    if matches_played >= high_threshold:
        return "High"
    if matches_played >= medium_threshold:
        return "Medium"
    return "Low"


def confidence_from_similar_decks(similar_count: int, total_matches: int) -> str:
    """
    Confidence label for estimates built from similar historical decks.
    """
    if similar_count >= 8 and total_matches >= 3000:
        return "High"
    if similar_count >= 4 and total_matches >= 1000:
        return "Medium"
    return "Low"


def get_model_predictions_safe(
    deck_cards: list[int],
    metadata_df: pd.DataFrame,
    feature_schema: Optional[list[str] | Dict[str, Any]] = None,
) -> Optional[Dict[str, float]]:
    """
    Safely load available models and get their predictions on a deck.
    
    Gracefully skips missing models. Returns None if no models are available/loadable.
    
    Args:
        deck_cards: List of 8 card IDs
        metadata_df: Card metadata
        feature_schema: Feature schema (loaded if not provided)
        
    Returns:
        dict mapping model names to predicted probabilities, or None if no models available
    """
    try:
        from .preprocess import build_feature_vector
        from .model_loader import (
            load_random_forest_model,
            load_logistic_regression_model,
            load_xgboost_model,
            load_feature_schema,
        )
    except ImportError:
        return None

    if feature_schema is None:
        try:
            feature_schema = load_feature_schema()
        except Exception:
            return None

    model_loaders = {
        "Random Forest": load_random_forest_model,
        "Logistic Regression": load_logistic_regression_model,
        "XGBoost": load_xgboost_model,
    }

    model_probabilities: Dict[str, float] = {}

    try:
        feature_df = build_feature_vector(
            deck_cards=deck_cards,
            metadata_df=metadata_df,
            feature_schema=feature_schema,
        )
    except Exception:
        return None

    for model_name, loader in model_loaders.items():
        try:
            model = loader()
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(feature_df)[0][1])
            else:
                pred = float(model.predict(feature_df)[0])
                prob = pred if 0 <= pred <= 1 else 0.5
            model_probabilities[model_name] = prob
        except Exception:
            continue

    return model_probabilities if len(model_probabilities) > 0 else None


def combine_confidence_signals(
    probability: float,
    historical_confidence_label: str,
    model_probabilities: Optional[Dict[str, float]] = None,
    model_weight: float = 0.6,
    historical_weight: float = 0.4,
) -> HybridConfidenceResult:
    """
    Combine model-based confidence and historical support confidence.

    Model signal priority:
    1) If 2+ model probabilities are available, use disagreement (std) across models.
    2) Otherwise, fall back to distance-from-0.5 confidence for the given probability.

    Returns a normalized score and a final High/Medium/Low label for UI badges.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")

    total_weight = model_weight + historical_weight
    if total_weight <= 0:
        raise ValueError("model_weight + historical_weight must be > 0")

    model_w = model_weight / total_weight
    historical_w = historical_weight / total_weight

    historical_score = _label_to_score(historical_confidence_label)

    model_score: float
    if model_probabilities and len(model_probabilities) >= 2:
        probs = np.array(list(model_probabilities.values()), dtype=float)
        uncertainty_std = float(np.std(probs))
        model_score = _normalize_std_to_confidence(uncertainty_std)
    else:
        model_score = compute_prediction_confidence(probability)

    combined_score = (model_w * model_score) + (historical_w * historical_score)
    return HybridConfidenceResult(
        label=_score_to_label(combined_score),
        score=combined_score,
        historical_score=historical_score,
        model_score=model_score,
    )


def ensure_dataframe(
    X_input: pd.DataFrame | pd.Series | dict,
    feature_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Ensure the input is a single-row DataFrame.
    """
    if isinstance(X_input, pd.DataFrame):
        return X_input.copy()

    if isinstance(X_input, pd.Series):
        return X_input.to_frame().T

    if isinstance(X_input, dict):
        df = pd.DataFrame([X_input])
        if feature_names is not None:
            missing = [col for col in feature_names if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            df = df[feature_names]
        return df

    raise TypeError("X_input must be a DataFrame, Series, or dict.")


def predict_with_uncertainty(
    X_input: pd.DataFrame | pd.Series | dict,
    models: Dict[str, Any],
    best_model_name: str,
    feature_names: Optional[list[str]] = None,
) -> PredictionResult:
    """
    Generate a final prediction using the best model and estimate uncertainty
    using disagreement across all trained models.

    Parameters
    ----------
    X_input : DataFrame | Series | dict
        A single example to predict.
    models : dict
        Example:
        {
            "Logistic Regression": log_reg_model,
            "Random Forest": rf_model,
            "XGBoost": xgb_model
        }
    best_model_name : str
        Name of the model to use for final prediction.
    feature_names : list[str], optional
        Feature order expected by the models.

    Returns
    -------
    PredictionResult
    """
    if best_model_name not in models:
        raise ValueError(f"'{best_model_name}' is not in models: {list(models.keys())}")

    X_df = ensure_dataframe(X_input, feature_names=feature_names)

    # Collect probabilities from all models
    model_probabilities: Dict[str, float] = {}
    for model_name, model in models.items():
        prob = float(model.predict_proba(X_df)[0][1])
        model_probabilities[model_name] = prob

    # Final prediction comes from the best model only
    final_probability = model_probabilities[best_model_name]
    predicted_class = int(final_probability >= 0.5)

    # Uncertainty comes from disagreement among all models
    probs = np.array(list(model_probabilities.values()), dtype=float)
    uncertainty_std = float(np.std(probs))

    # Simple uncertainty interval around the best model's prediction
    low = max(0.0, final_probability - uncertainty_std)
    high = min(1.0, final_probability + uncertainty_std)

    confidence_label = get_confidence_label(uncertainty_std)

    return PredictionResult(
        best_model_name=best_model_name,
        final_probability=final_probability,
        predicted_class=predicted_class,
        uncertainty_std=uncertainty_std,
        uncertainty_range_low=low,
        uncertainty_range_high=high,
        confidence_label=confidence_label,
        model_probabilities=model_probabilities,
    )


# Fallback functions from original implementation
def compute_prediction_confidence(probability: float) -> float:
    """
    Compute confidence in a prediction based on distance from 0.5.
    
    This is a simple heuristic: predictions close to 0.5 are uncertain,
    while predictions close to 0 or 1 are confident.
    
    Note: Not a calibrated uncertainty estimate, but useful for user-facing indication.
    
    Args:
        probability: Prediction probability (0-1)
        
    Returns:
        Confidence score (0-1)
        - 0.5 = most uncertain
        - 1.0 = maximum confidence
        
    Raises:
        ValueError: If probability not in [0, 1]
    """
    if not (0 <= probability <= 1):
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")
    
    # Distance from 0.5
    distance = abs(probability - 0.5)
    
    # Scale to 0-1 range where 0.5 = confidence 0 and 0/1 = confidence 1
    confidence = distance * 2.0  # distance ranges 0-0.5, so multiply by 2
    
    return min(confidence, 1.0)  # Ensure <= 1


def compute_uncertainty(probability: float) -> float:
    """
    Compute uncertainty in a prediction (inverse of confidence).
    
    Args:
        probability: Prediction probability (0-1)
        
    Returns:
        Uncertainty score (0-1)
        - 0 = certain
        - 1.0 = maximum uncertainty
    """
    confidence = compute_prediction_confidence(probability)
    return 1.0 - confidence


def confidence_label(confidence: float) -> str:
    """
    Get a human-readable label for confidence level.
    
    DEPRECATED: Use get_confidence_label() instead for uncertainty-based labels.
    This function is kept for backward compatibility with the original implementation.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Confidence label string
    """
    if confidence >= 0.8:
        return "Very High"
    elif confidence >= 0.6:
        return "High"
    elif confidence >= 0.4:
        return "Moderate"
    else:
        return "Low"
