"""
Simple uncertainty/confidence estimation for predictions.
Uses distance from 0.5 as a heuristic for prediction certainty.
"""

from typing import Union


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
