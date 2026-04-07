"""
Win Predictor Page
==================
Input: player deck (8 cards)
Output: estimated win probability, explanation, confidence level
"""

from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd
import polars as pl
import streamlit as st

from utils.metadata import (
    get_card_names,
    get_card_types,
    get_elixir_costs,
    get_icon_urls,
    get_card_metadata,
)
from utils.deck_helpers import (
    build_deck_key,
    compute_avg_elixir,
    compute_cycle_cost,
    count_card_types,
    detect_archetype,
    enrich_deck_record,
)
from utils.uncertainty import (
    combine_confidence_signals,
    confidence_from_match_count,
    confidence_from_similar_decks,
    predict_probability_with_xgboost,
)
from utils.model_loader import load_best_model, load_feature_schema, load_xgboost_model
from utils.preprocess import build_feature_vector
from utils.explanation_engine import build_prediction_explanations
from utils.ui_helpers import inject_fonts
from utils.data_loader import load_card_rankings, get_clean_parquet_source

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Win Predictor", layout="wide")
inject_fonts()

st.markdown(
    """
    <style>
    section.main > div {
        max-width: 1100px;
        margin: auto;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d0dbe8;
        padding: 10px 14px;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(26, 86, 219, 0.08);
    }

    div[data-testid="stMetric"] label {
        color: #6b7fa3 !important;
        font-size: 12px !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1a3a6e !important;
        font-size: 18px !important;
    }

    .conf-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 13px;
        color: white;
        margin-top: 6px;
    }

    .conf-high   { background: #16a34a; }
    .conf-medium { background: #f59e0b; }
    .conf-low    { background: #ef4444; }

    .note-box {
        background: #ffffff;
        border: 1px solid #d0dbe8;
        border-radius: 10px;
        padding: 14px 16px;
        color: #3b536e;
        font-size: 14px;
        box-shadow: 0 1px 3px rgba(26, 86, 219, 0.06);
    }

    .section-label {
        color: #6b7fa3;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .slot-card {
        border: 1px dashed #b0c4de;
        border-radius: 10px;
        min-height: 78px;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6b7fa3;
        font-size: 13px;
        text-align: center;
        padding: 8px;
    }

    .explanation-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #e8f2ff 100%);
        border: 1px solid #c0d8f0;
        border-left: 4px solid #1a5490;
        border-radius: 10px;
        padding: 18px 20px;
        margin: 12px 0;
        color: #1a3a6e;
        font-size: 14px;
        line-height: 1.7;
    }

    .explanation-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        font-weight: 700;
        font-size: 15px;
        color: #1a3a6e;
    }

    .explanation-bullet {
        margin: 10px 0;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 8px;
        border-left: 3px solid #4a90d9;
        font-size: 14px;
        line-height: 1.6;
    }

    .deck-card-wrap {
        position: relative;
        display: inline-block;
        width: 100%;
    }

    .deck-card-wrap img {
        border-radius: 10px;
        width: 100%;
    }

    .chooser-section {
        background: #ffffff;
        border: 1px solid #d0dbe8;
        border-radius: 12px;
        padding: 12px 14px;
    }

    .elixir-badge {
        display: inline-flex;
        align-items: center;
        gap: 2px;
        background: #e8f0fe;
        border-radius: 10px;
        padding: 1px 6px;
        font-size: 11px;
        font-weight: 700;
        color: #1a3a6e;
    }

    .elixir-badge img {
        width: 13px;
        height: 13px;
    }

    /* Make Streamlit top header/deploy bar transparent */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_SIZE = 200_000

PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
PLAYER_CROWNS_COL = "player1.crowns"
OPPONENT_CROWNS_COL = "player2.crowns"

SIMILAR_DECK_OVERLAP = 6
TOP_SIMILAR = 25
MIN_MATCHES = 20
MAX_ENRICHED_DECKS = 1500


@st.cache_data(show_spinner=False, ttl=3600)
def load_card_assets():
    """Load and cache all card data."""
    name_map = get_card_names(force_refresh=False)
    type_map = get_card_types(force_refresh=False)
    elixir_map = get_elixir_costs(force_refresh=False)
    icon_map = get_icon_urls(force_refresh=False)

    card_df = pd.DataFrame(
        {
            "card_id": list(name_map.keys()),
            "name": [name_map[cid] for cid in name_map.keys()],
        }
    )

    banned_keywords = [
        "super ",
        "santa ",
        "terry",
        "party",
        "evolved ",
        "evolution",
        "mirror mode",
        "placeholder",
        "raging",
    ]

    mask = ~card_df["name"].str.lower().apply(
        lambda x: any(keyword in x for keyword in banned_keywords)
    )

    card_df = card_df[mask].copy()
    card_df = card_df.sort_values("name").reset_index(drop=True)
    valid_ids = set(card_df["card_id"].tolist())

    name_map = {cid: name for cid, name in name_map.items() if cid in valid_ids}
    type_map = {cid: ctype for cid, ctype in type_map.items() if cid in valid_ids}
    elixir_map = {cid: cost for cid, cost in elixir_map.items() if cid in valid_ids}
    icon_map = {cid: url for cid, url in icon_map.items() if cid in valid_ids}

    return card_df, name_map, type_map, elixir_map, icon_map


@st.cache_data(show_spinner=False, ttl=3600)
def load_metadata_df() -> pd.DataFrame:
    return get_card_metadata(force_refresh=False)


@st.cache_data(show_spinner=True, ttl=3600)
def load_match_data() -> pl.DataFrame:
    needed = PLAYER_CARD_COLS + [PLAYER_CROWNS_COL, OPPONENT_CROWNS_COL]
    df = pl.scan_parquet(get_clean_parquet_source()).select(needed).collect()
    if df.height > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, seed=42)
    return df


@st.cache_data(show_spinner=True, ttl=3600)
def build_all_popular_decks_table() -> pd.DataFrame:
    """Build and cache the popular decks table."""
    df = load_match_data()

    df = df.with_columns(
        pl.concat_list(PLAYER_CARD_COLS).alias("deck_cards")
    ).with_columns(
        pl.col("deck_cards")
        .list.eval(pl.element().cast(pl.Int64))
        .list.sort()
        .list.eval(pl.element().cast(pl.Utf8))
        .list.join(",")
        .alias("deck_key"),
        (pl.col(PLAYER_CROWNS_COL) > pl.col(OPPONENT_CROWNS_COL)).cast(pl.Int8).alias("player_win"),
    )

    grouped = (
        df.group_by("deck_key")
        .agg(
            pl.len().alias("matches_played"),
            pl.col("player_win").sum().alias("wins"),
        )
        .with_columns(
            (pl.col("matches_played") - pl.col("wins")).alias("losses"),
            (pl.col("wins") / pl.col("matches_played") * 100).alias("win_rate"),
        )
        .sort(["matches_played", "win_rate"], descending=[True, True])
        .head(MAX_ENRICHED_DECKS)
    )

    grouped_pd = grouped.to_pandas()
    _, name_map, type_map, elixir_map, _ = load_card_assets()

    records = []
    for _, row in grouped_pd.iterrows():
        enriched = enrich_deck_record(
            deck_key=row["deck_key"],
            matches_played=int(row["matches_played"]),
            wins=int(row["wins"]),
            name_map=name_map,
            elixir_map=elixir_map,
            type_map=type_map,
        )
        enriched["losses"] = int(row["losses"])
        enriched["win_rate"] = float(row["win_rate"])
        records.append(enriched)

    result = pd.DataFrame(records)
    return (
        result.sort_values(["matches_played", "win_rate"], ascending=[False, False]).reset_index(drop=True)
        if not result.empty
        else result
    )


def estimate_deck_win_rate(cards: list[int], decks_df: pd.DataFrame) -> dict:
    """Estimate win rate for a deck."""
    dk = build_deck_key(cards)
    exact = decks_df[decks_df["deck_key"] == dk]
    if not exact.empty:
        row = exact.iloc[0]
        return {
            "win_rate": float(row["win_rate"]),
            "matches": int(row["matches_played"]),
            "source": "exact",
            "confidence": confidence_from_match_count(int(row["matches_played"])),
        }

    target = set(cards)
    work_df = decks_df[["card_ids", "matches_played", "win_rate"]].copy()
    work_df["overlap"] = work_df["card_ids"].apply(
        lambda vals: len(target.intersection(set(int(x) for x in vals)))
    )
    work_df = work_df[work_df["overlap"] >= SIMILAR_DECK_OVERLAP].copy()

    if work_df.empty:
        return {
            "win_rate": 50.0,
            "matches": 0,
            "source": "fallback",
            "confidence": confidence_from_match_count(matches_played=0),
        }

    work_df["sim"] = work_df["overlap"] / 8.0
    work_df = work_df.sort_values(["sim", "matches_played"], ascending=[False, False]).head(TOP_SIMILAR)

    weights = work_df["sim"] * work_df["matches_played"]
    wr = float((work_df["win_rate"] * weights).sum() / weights.sum())
    tm = int(work_df["matches_played"].sum())

    return {
        "win_rate": round(wr, 2),
        "matches": tm,
        "source": f"~{len(work_df)} similar decks",
        "confidence": confidence_from_similar_decks(len(work_df), tm),
    }


@st.cache_resource(show_spinner=False)
def load_model_resources():
    """Cache model and feature schema."""
    try:
        model = load_xgboost_model()
        model_name = "XGBoost"
    except Exception:
        model, model_name = load_best_model()
    feature_schema = load_feature_schema()
    return model, model_name, feature_schema


def render_card_image(card_id: int, icon_map: dict, name_map: dict) -> None:
    """Render a card image or fallback."""
    url = icon_map.get(int(card_id))
    name = name_map.get(int(card_id), str(card_id))
    if url and isinstance(url, str) and url.strip():
        st.image(url, use_container_width=True)
    else:
        st.markdown(
            f"<div style='border:1px solid #c5d5ea;border-radius:10px;padding:8px;"
            f"text-align:center;min-height:90px;display:flex;align-items:center;"
            f"justify-content:center;font-size:12px;color:#6b7fa3;background:#fff;'>"
            f"{name}</div>",
            unsafe_allow_html=True,
        )


def render_empty_slot(n: int):
    st.markdown(f"<div class='slot-card'>Slot {n}</div>", unsafe_allow_html=True)


def render_confidence(conf: str):
    cls = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}.get(conf, "conf-medium")
    st.markdown(
        f"<span class='conf-badge {cls}'>{conf} Confidence</span>",
        unsafe_allow_html=True,
    )


def init_deck_builder_state():
    if "deck_slots" not in st.session_state:
        st.session_state.deck_slots = [None] * 8


def _add(idx: int, cid: int):
    if cid not in st.session_state.deck_slots:
        st.session_state.deck_slots[idx] = int(cid)


def _remove(idx: int):
    st.session_state.deck_slots[idx] = None


def _clear():
    st.session_state.deck_slots = [None] * 8


def _filled() -> list[int]:
    return [c for c in st.session_state.deck_slots if c is not None]


def render_deck_builder(
    card_df: pd.DataFrame,
    name_map: dict,
    type_map: dict,
    elixir_map: dict,
    icon_map: dict,
):
    """Render the deck builder UI with image-based card browser."""
    st.markdown("<div class='section-label'>Your Deck</div>", unsafe_allow_html=True)
    filled = _filled()

    # Show selected deck cards in 2 rows of 4
    for row_start in (0, 4):
        cols = st.columns(4, gap="small")
        for i in range(4):
            idx = row_start + i
            with cols[i]:
                cid = st.session_state.deck_slots[idx]
                if cid is None:
                    render_empty_slot(idx + 1)
                else:
                    url = icon_map.get(int(cid), "")
                    if url:
                        st.markdown(
                            f"<div class='deck-card-wrap'><img src='{url}'/></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        render_card_image(cid, icon_map, name_map)
                    if st.button("\u2715", key=f"rm_{idx}", use_container_width=True):
                        _remove(idx)
                        st.rerun()

    if filled:
        avg_e = compute_avg_elixir(filled, elixir_map)
        cyc = compute_cycle_cost(filled, elixir_map) if len(filled) >= 4 else 0
        tc = count_card_types(filled, type_map)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg Elixir", f"{avg_e:.1f}")
        c2.metric("Cycle", cyc if cyc else "\u2014")
        c3.metric("Troops", tc["troop_count"])
        c4.metric("Spells", tc["spell_count"])
        c5.metric("Buildings", tc["building_count"])

    st.caption(f"{len(filled)}/8 cards selected")

    if st.button("Clear Deck", use_container_width=True):
        _clear()
        st.rerun()

    if len(filled) >= 8:
        return

    already_selected = set(filled)

    # Render all available cards as a single HTML grid per category (fast, no st.image)
    elixir_icon = "https://cdn.royaleapi.com/static/img/ui/elixir.png"
    category_map = [("Troops", "troop"), ("Spells", "spell"), ("Buildings", "building")]

    for cat_label, cat_key in category_map:
        section = card_df[card_df["card_id"].map(type_map).fillna("") == cat_key]
        section = section[~section["card_id"].isin(already_selected)].reset_index(drop=True)
        if section.empty:
            continue

        st.markdown(
            f"<div style='background:#dce6f5;padding:6px 14px;border-radius:6px;"
            f"color:#1a3a6e;font-weight:700;font-size:14px;margin:12px 0 8px;'>"
            f"{cat_label} ({len(section)} cards)</div>",
            unsafe_allow_html=True,
        )

        per_row = 8
        for start in range(0, len(section), per_row):
            chunk = section.iloc[start:start + per_row]
            cols = st.columns(per_row, gap="small")
            for ci, (_, row) in enumerate(chunk.iterrows()):
                with cols[ci]:
                    cid = int(row["card_id"])
                    url = icon_map.get(cid)
                    cname = name_map.get(cid, str(cid))
                    ec = elixir_map.get(cid, "")

                    html = ""
                    if url:
                        html += f"<img src='{url}' style='width:100%;border-radius:10px;'/>"
                    else:
                        html += f"<div style='border:1px solid #c5d5ea;border-radius:10px;padding:8px;text-align:center;min-height:70px;display:flex;align-items:center;justify-content:center;font-size:11px;color:#6b7fa3;background:#fff;'>{cname}</div>"
                    if ec:
                        html += (
                            f"<div style='text-align:center'>"
                            f"<span class='elixir-badge'>"
                            f"<img src='{elixir_icon}'/>{ec}</span></div>"
                        )
                    st.markdown(html, unsafe_allow_html=True)

                    if st.button("Add", key=f"add_{cid}", use_container_width=True):
                        next_slot = next((i for i, v in enumerate(st.session_state.deck_slots) if v is None), None)
                        if next_slot is not None:
                            _add(next_slot, cid)
                            st.rerun()


def get_prediction_explanation_bullets(
    selected_cards: list[int],
    metadata_df: pd.DataFrame,
) -> list[str]:
    """
    Get explanation bullets using the enhanced explanation engine.
    Always returns bullet-style output; never returns a paragraph string.
    """
    try:
        model, _model_name, feature_schema = load_model_resources()

        feature_df = build_feature_vector(
            deck_cards=selected_cards,
            metadata_df=metadata_df,
            feature_schema=feature_schema,
        )

        if feature_df is None or feature_df.empty:
            logger.warning("Feature vector for explanation engine is empty.")
            return []

        explanation_bullets = build_prediction_explanations(
            model=model,
            feature_df=feature_df,
            metadata_df=metadata_df,
            player_cards=[int(c) for c in selected_cards],
            opponent_cards=None,
            max_bullets=4,
            debug=False,
        )

        if explanation_bullets:
            return [str(b) for b in explanation_bullets[:4] if str(b).strip()]

    except Exception as e:
        logger.exception("Explanation engine failed")

    return []


def build_simple_fallback_bullets(
    archetype: str,
    avg_elixir: float,
    cycle_cost: int,
    troop_count: int,
    spell_count: int,
    building_count: int,
    source_label: str,
) -> list[str]:
    """Simple bullet fallback if explanation engine fails."""
    bullets: list[str] = []

    if archetype != "Unknown":
        bullets.append(f"This deck most closely matches a {archetype} strategy.")
    else:
        bullets.append("This deck uses a mixed strategy without a strong single archetype.")

    if avg_elixir <= 3.0:
        bullets.append(f"Average elixir is {avg_elixir:.1f}, so it should play at a fast pace.")
    elif avg_elixir >= 4.3:
        bullets.append(f"Average elixir is {avg_elixir:.1f}, so it plays more like a heavy push deck.")
    else:
        bullets.append(f"Average elixir is {avg_elixir:.1f}, which gives it a balanced pace.")

    bullets.append(
        f"Cycle cost is {cycle_cost}, with {troop_count} troops, {spell_count} spells, and {building_count} buildings."
    )

    bullets.append(f"This estimate is based on {source_label} historical support.")

    return bullets[:4]


def main():
    st.title("🎯 Win Predictor")
    st.markdown(
        "<div style='margin-bottom:10px;color:#5a7394;font-size:15px;'>"
        "Select your 8-card deck, then <strong>Analyze</strong> to see your estimated win rate, "
        "confidence level, and how the model thinks about your deck."
        "</div>",
        unsafe_allow_html=True,
    )

    init_deck_builder_state()
    card_df, name_map, type_map, elixir_map, icon_map = load_card_assets()
    metadata_df = load_metadata_df()

    render_deck_builder(card_df, name_map, type_map, elixir_map, icon_map)

    selected_cards = _filled()
    ready = len(selected_cards) == 8

    analyze = st.button(
        "🎯 Analyze Deck",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    if not ready:
        st.info("Select exactly 8 cards to analyze.")
        return

    if not analyze:
        return

    decks_df = build_all_popular_decks_table()
    p_est = estimate_deck_win_rate(selected_cards, decks_df)
    archetype = detect_archetype(selected_cards, name_map, elixir_map)
    avg_elixir = compute_avg_elixir(selected_cards, elixir_map)
    cycle_cost = compute_cycle_cost(selected_cards, elixir_map)
    tc = count_card_types(selected_cards, type_map)

    model = None
    model_name = None
    feature_schema = None
    model_probability = None

    try:
        model, model_name, feature_schema = load_model_resources()
        model_probability = predict_probability_with_xgboost(
            deck_cards=selected_cards,
            metadata_df=metadata_df,
            feature_schema=feature_schema,
        )
    except FileNotFoundError:
        logger.info("No trained model found; using historical data only.")
    except Exception as e:
        logger.warning("Model loading failed: %s: %s", type(e).__name__, str(e)[:150])

    if model_probability is not None:
        player_win_prob = round(model_probability * 100.0, 2)
        probability_source = f"{model_name} model"
    else:
        player_win_prob = p_est["win_rate"]
        probability_source = p_est["source"]

    historical_conf = p_est["confidence"]

    try:
        conf_result = combine_confidence_signals(
            probability=player_win_prob / 100.0,
            historical_confidence_label=historical_conf,
            model_probabilities={model_name: model_probability}
            if model_probability is not None and model_name is not None
            else None,
            model_weight=0.6,
            historical_weight=0.4,
        )
        confidence = conf_result.label
    except Exception as e:
        logger.warning("Confidence calculation fallback: %s: %s", type(e).__name__, str(e)[:150])
        confidence = historical_conf

    if abs(player_win_prob - 50) < 1.0:
        banner_text = f"Predicted Win Rate: {player_win_prob:.1f}% (Evenly Matched)"
        banner_color = "#fef9c3"
        text_color = "#854d0e"
    elif player_win_prob > 50:
        banner_text = f"📈 Predicted Win Rate: {player_win_prob:.1f}% (Favorable)"
        banner_color = "#dcfce7"
        text_color = "#166534"
    else:
        banner_text = f"📉 Predicted Win Rate: {player_win_prob:.1f}% (Unfavorable)"
        banner_color = "#fee2e2"
        text_color = "#991b1b"

    st.markdown(
        f"<div style='text-align:center;padding:18px;border-radius:12px;font-size:20px;"
        f"font-weight:700;background:{banner_color};color:{text_color};margin:12px 0;'>"
        f"{banner_text}</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Win Probability", f"{player_win_prob:.1f}%")
    m2.metric("Sample Size", f"{p_est['matches']:,} matches")
    m3.metric("Confidence", confidence)

    render_confidence(confidence)
    st.caption(f"Based on: {probability_source}")

    # ── Meta card count badge ───────────────────────────────────────
    rankings_df = load_card_rankings()
    if rankings_df is not None and not rankings_df.empty:
        top20_names = set(rankings_df.head(20)["card_name"])
        selected_names = [name_map.get(int(c), "") for c in selected_cards]
        meta_count = sum(1 for n in selected_names if n in top20_names)
        st.markdown(
            f"<div class='note-box' style='margin-top:8px;'>"
            f"🔥 This deck contains <strong>{meta_count}</strong> top meta card(s) "
            f"(out of the 20 most-used cards in the current meta).</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Why This Prediction Makes Sense")

    explanation_bullets = get_prediction_explanation_bullets(
        selected_cards=selected_cards,
        metadata_df=metadata_df,
    )

    if not explanation_bullets:
        explanation_bullets = build_simple_fallback_bullets(
            archetype=archetype,
            avg_elixir=avg_elixir,
            cycle_cost=cycle_cost,
            troop_count=tc["troop_count"],
            spell_count=tc["spell_count"],
            building_count=tc["building_count"],
            source_label=p_est["source"],
        )

    bullets_html = "".join(
        f"<div class='explanation-bullet'>{bullet}</div>"
        for bullet in explanation_bullets
    )
    st.markdown(
        f"<div class='explanation-box'>"
        f"<div class='explanation-header'>🧠 Deck Analysis</div>"
        f"{bullets_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Deck Overview")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("<div class='section-label'>Your Cards</div>", unsafe_allow_html=True)
        for row_start in (0, 4):
            img_cols = st.columns(4, gap="small")
            for i in range(4):
                with img_cols[i]:
                    render_card_image(selected_cards[row_start + i], icon_map, name_map)

        card_names = ", ".join(name_map.get(int(c), str(c)) for c in selected_cards)
        st.caption(card_names)

    with right:
        st.markdown("<div class='section-label'>Statistics</div>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("Archetype", archetype)
        s2.metric("Avg Elixir", f"{avg_elixir:.2f}")
        s3.metric("Cycle Cost", cycle_cost)

        s4, s5, s6 = st.columns(3)
        s4.metric("Troops", tc["troop_count"])
        s5.metric("Spells", tc["spell_count"])
        s6.metric("Buildings", tc["building_count"])


if __name__ == "__main__":
    main()