"""
Streamlit app utilities package.
Provides modular utilities for data loading, model inference, preprocessing, and UI components.
"""

from .data_loader import (
    get_clean_parquet_source,
    get_archetype_parquet_source,
    get_elixir_parquet_source,
    get_final_ml_parquet_source,
    load_card_metadata,
    load_final_dataset,
    load_csv_if_exists,
    load_parquet_if_exists,
    load_archetype_stats,
    load_popular_decks,
    load_historical_trends,
    load_card_rankings,
)

from .model_loader import (
    load_model,
    load_logistic_regression_model,
    load_random_forest_model,
    load_xgboost_model,
    load_best_model,
    load_feature_schema,
)

from .preprocess import (
    normalize_card_input,
    validate_deck_size,
    build_deck_one_hot,
    compute_deck_summary_features,
    align_to_schema,
    build_feature_vector,
)

from .prediction import (
    predict_win_probability,
    predict_matchup,
    format_probability,
)

from .ui_helpers import (
    render_page_header,
    deck_selector,
    display_prediction_result,
    display_deck,
    comparison_metric,
)

from .uncertainty import (
    compute_prediction_confidence,
    compute_uncertainty,
    confidence_label,
)

from .recommendation import (
    rank_candidate_decks,
    recommend_best_decks,
)

from .explanation_engine import build_prediction_explanations

__all__ = [
    # data_loader
    "load_card_metadata",
    "load_final_dataset",
    "load_csv_if_exists",
    "load_parquet_if_exists",
    "load_archetype_stats",
    "load_popular_decks",
    "load_historical_trends",
    # model_loader
    "load_model",
    "load_logistic_regression_model",
    "load_random_forest_model",
    "load_xgboost_model",
    "load_best_model",
    "load_feature_schema",
    # preprocess
    "normalize_card_input",
    "validate_deck_size",
    "build_deck_one_hot",
    "compute_deck_summary_features",
    "align_to_schema",
    "build_feature_vector",
    # prediction
    "predict_win_probability",
    "predict_matchup",
    "format_probability",
    # ui_helpers
    "render_page_header",
    "deck_selector",
    "display_prediction_result",
    "display_deck",
    "comparison_metric",
    # uncertainty
    "compute_prediction_confidence",
    "compute_uncertainty",
    "confidence_label",
    # recommendation
    "rank_candidate_decks",
    "recommend_best_decks",
    #explanation_engine
    "build_prediction_explanations"
]
