"""
Deck recommendation utilities.
Ranks candidate decks based on predicted win probability and
generates card-swap suggestions from historical data and XGBoost predictions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .preprocess import build_feature_vector
from .prediction import predict_win_probability


# ── Model-based ranking ────────────────────────────────────────────


def rank_candidate_decks(
    model: Any,
    candidate_decks: List[List[Union[str, int]]],
    metadata_df: pd.DataFrame,
    feature_schema: Union[List[str], Dict[str, Any]],
    top_k: int = 5,
) -> pd.DataFrame:
    """Rank candidate decks by predicted win probability."""
    results = []
    for idx, deck in enumerate(candidate_decks):
        try:
            feature_df = build_feature_vector(
                deck_cards=deck,
                metadata_df=metadata_df,
                feature_schema=feature_schema,
                prefix="player",
            )
            win_prob = predict_win_probability(model, feature_df)
            results.append(
                {"deck_index": idx, "cards": tuple(deck), "win_probability": win_prob}
            )
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("win_probability", ascending=False).reset_index(drop=True)
    df_results["rank"] = range(1, len(df_results) + 1)
    return df_results.head(top_k)


def recommend_best_decks(
    model: Any,
    candidate_decks: List[List[Union[str, int]]],
    metadata_df: pd.DataFrame,
    feature_schema: Union[List[str], Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return top ranked decks as a list of dicts for Streamlit display."""
    ranked_df = rank_candidate_decks(
        model, candidate_decks, metadata_df, feature_schema, top_k
    )
    if ranked_df.empty:
        return []
    recommendations = []
    for _, row in ranked_df.iterrows():
        recommendations.append(
            {
                "rank": int(row["rank"]),
                "cards": list(row["cards"]),
                "win_probability": float(row["win_probability"]),
                "formatted_probability": f"{row['win_probability'] * 100:.1f}%",
            }
        )
    return recommendations


# ── Data-driven swap recommendations ──────────────────────────────


def generate_swap_candidates(
    deck: list[int],
    pool: list[int],
) -> list[tuple[int, int, list[int]]]:
    """
    Generate single-card swap variants of *deck*.

    Returns list of (removed_card_id, added_card_id, new_deck) tuples.
    Only cards in *pool* that are not already in the deck are considered.
    """
    deck_set = set(deck)
    available = [c for c in pool if c not in deck_set]
    candidates: list[tuple[int, int, list[int]]] = []
    for idx, old_card in enumerate(deck):
        for new_card in available:
            new_deck = list(deck)
            new_deck[idx] = new_card
            candidates.append((old_card, new_card, sorted(new_deck)))
    return candidates


def score_swaps_with_model(
    deck: list[int],
    pool: list[int],
    metadata_df: pd.DataFrame,
    feature_schema: Union[List[str], Dict[str, Any]],
    base_prob: float,
    top_k: int = 10,
) -> list[dict]:
    """
    Score every single-card swap using XGBoost and return top-k improvements.

    Each swap is scored via ``predict_probability_with_xgboost`` from uncertainty.py
    which handles single-deck inference (no opponent needed).
    """
    from .uncertainty import predict_probability_with_xgboost

    candidates = generate_swap_candidates(deck, pool)
    results: list[dict] = []

    seen_keys: set[str] = set()
    for old_card, new_card, new_deck in candidates:
        key = ",".join(str(c) for c in new_deck)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        prob = predict_probability_with_xgboost(
            deck_cards=new_deck,
            metadata_df=metadata_df,
            feature_schema=feature_schema,
        )
        if prob is None:
            continue
        delta = prob - base_prob
        results.append(
            {
                "removed": old_card,
                "added": new_card,
                "new_deck": new_deck,
                "win_prob": prob,
                "delta": delta,
            }
        )

    results.sort(key=lambda r: r["delta"], reverse=True)
    return results[:top_k]


def find_top_historical_decks(
    deck_lookup_df: pd.DataFrame,
    name_map: dict[int, str],
    elixir_map: dict[int, int],
    type_map: dict[int, str],
    archetype_filter: Optional[str] = None,
    min_matches: int = 50,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Return the highest win-rate decks from a pre-computed deck lookup DataFrame.

    *deck_lookup_df* is expected to have columns produced by ``enrich_deck_record``:
        deck_key, card_ids, matches_played, wins, win_rate, archetype, …

    If *archetype_filter* is given (and not "All"), only matching rows are kept.
    """
    df = deck_lookup_df[deck_lookup_df["matches_played"] >= min_matches].copy()
    if archetype_filter and archetype_filter != "All":
        df = df[df["archetype"] == archetype_filter]
    df = df.sort_values("win_rate", ascending=False).head(top_k).reset_index(drop=True)
    return df


def find_similar_decks(
    deck: list[int],
    deck_lookup_df: pd.DataFrame,
    min_overlap: int = 5,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Find decks in the lookup that share at least *min_overlap* cards with *deck*.
    Results are sorted by win-rate descending.
    """
    target = set(deck)
    rows: list[dict] = []
    for _, r in deck_lookup_df.iterrows():
        overlap = len(target & set(r["card_ids"]))
        if overlap >= min_overlap:
            rows.append({**r.to_dict(), "shared_cards": overlap})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["shared_cards", "win_rate"], ascending=[False, False])
    return df.head(top_k).reset_index(drop=True)
