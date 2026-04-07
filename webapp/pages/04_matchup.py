"""
Matchup Analysis Page
=====================
Input: player + opponent deck.
Output: win probability, predicted winner, matchup stats.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from utils.metadata import (
    get_card_names,
    get_card_types,
    get_elixir_costs,
    get_icon_urls,
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
    confidence_from_match_count,
    combine_confidence_signals,
    predict_probability_with_xgboost,
)
from utils.metadata import get_card_metadata
from utils.model_loader import load_feature_schema, load_best_model, load_xgboost_model
from utils.explanation_engine import build_prediction_explanations
from utils.preprocess import build_feature_vector

st.set_page_config(page_title="Matchup Analysis", layout="wide")

from utils.ui_helpers import inject_fonts
from utils.data_loader import load_card_rankings, get_clean_parquet_source

inject_fonts()

# ── additional CSS ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    section.main > div { max-width: 1200px; margin: auto; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d0dbe8;
        padding: 10px 14px;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(26,86,219,0.08);
    }
    div[data-testid="stMetric"] label { color: #6b7fa3 !important; font-size: 12px !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1a3a6e !important; font-size: 18px !important; }
    .section-label { color: #6b7fa3; font-size: 13px; font-weight: 700;
        margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em; }
    .slot-card { border: 1px dashed #b0c4de; border-radius: 10px; min-height: 78px;
        background: #ffffff; display: flex; align-items: center; justify-content: center;
        color: #6b7fa3; font-size: 13px; text-align: center; padding: 8px; }
    .note-box { background: #ffffff; border: 1px solid #d0dbe8; border-radius: 10px;
        padding: 14px 16px; color: #3b536e; font-size: 14px;
        box-shadow: 0 1px 3px rgba(26,86,219,0.06); }
    .conf-badge { display: inline-block; padding: 6px 14px; border-radius: 999px;
        font-weight: 700; font-size: 13px; color: white; margin-top: 6px; }
    .conf-high   { background: #16a34a; }
    .conf-medium { background: #f59e0b; }
    .conf-low    { background: #ef4444; }
    .winner-banner { text-align: center; padding: 18px; border-radius: 12px;
        font-size: 22px; font-weight: 700; margin: 12px 0; }
    .winner-player { background: #dcfce7; color: #166534; border: 2px solid #16a34a; }
    .winner-opponent { background: #fee2e2; color: #991b1b; border: 2px solid #ef4444; }
    .winner-draw { background: #fef9c3; color: #854d0e; border: 2px solid #f59e0b; }
    .elixir-badge { display: inline-flex; align-items: center; gap: 2px;
        background: #e8f0fe; border-radius: 10px; padding: 1px 6px;
        font-size: 11px; font-weight: 700; color: #1a3a6e; }
    .elixir-badge img { width: 13px; height: 13px; }
    .deck-card-wrap { position: relative; display: inline-block; width: 100%; }
    .deck-card-wrap img { border-radius: 10px; width: 100%; }
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
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_SIZE = 200_000

PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
OPPONENT_CARD_COLS = [f"player2.card{i}" for i in range(1, 9)]
PLAYER_CROWNS_COL = "player1.crowns"
OPPONENT_CROWNS_COL = "player2.crowns"

SIMILAR_DECK_OVERLAP = 6
TOP_SIMILAR = 25
MIN_MATCHES = 20


# ── Helpers ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def load_card_assets():
    name_map = get_card_names(force_refresh=False)
    type_map = get_card_types(force_refresh=False)
    elixir_map = get_elixir_costs(force_refresh=False)
    icon_map = get_icon_urls(force_refresh=False)

    card_df = pd.DataFrame({"card_id": list(name_map.keys()),
                            "name": [name_map[c] for c in name_map]})
    banned = ["super ", "santa ", "terry", "party", "evolved ", "evolution",
              "mirror mode", "placeholder"]
    mask = ~card_df["name"].str.lower().apply(lambda x: any(k in x for k in banned))
    card_df = card_df[mask].sort_values("name").reset_index(drop=True)
    valid = set(card_df["card_id"])
    name_map = {k: v for k, v in name_map.items() if k in valid}
    type_map = {k: v for k, v in type_map.items() if k in valid}
    elixir_map = {k: v for k, v in elixir_map.items() if k in valid}
    icon_map = {k: v for k, v in icon_map.items() if k in valid}
    return card_df, name_map, type_map, elixir_map, icon_map


@st.cache_data(show_spinner=False, ttl=3600)
def _load_metadata_df() -> pd.DataFrame:
    return get_card_metadata(force_refresh=False)


@st.cache_data(show_spinner=True, ttl=3600)
def load_match_data() -> pl.DataFrame:
    needed = PLAYER_CARD_COLS + OPPONENT_CARD_COLS + [PLAYER_CROWNS_COL, OPPONENT_CROWNS_COL]
    df = pl.scan_parquet(get_clean_parquet_source()).select(needed).collect()
    if df.height > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, seed=42)
    return df


def _build_deck_key_col(df: pl.DataFrame, card_cols: list[str], alias: str) -> pl.DataFrame:
    """Build a canonical deck key column using Polars vectorized ops."""
    return df.with_columns(
        pl.concat_list(card_cols)
        .list.eval(pl.element().cast(pl.Int64))
        .list.sort()
        .list.eval(pl.element().cast(pl.Utf8))
        .list.join(",")
        .alias(alias)
    )


@st.cache_data(show_spinner="Building deck key index \u2026", ttl=3600)
def _keyed_match_data() -> pd.DataFrame:
    """Compute deck keys ONCE using Polars (vectorized), then convert to pandas."""
    df = load_match_data()
    df = _build_deck_key_col(df, PLAYER_CARD_COLS, "p_key")
    df = _build_deck_key_col(df, OPPONENT_CARD_COLS, "o_key")
    df = df.with_columns(
        (pl.col(PLAYER_CROWNS_COL) > pl.col(OPPONENT_CROWNS_COL)).cast(pl.Int8).alias("p_win")
    )
    return df.select(["p_key", "o_key", "p_win"]).to_pandas()


@st.cache_data(show_spinner=True, ttl=3600)
def build_deck_lookup(min_matches: int) -> pd.DataFrame:
    """Build grouped deck stats from player-side data."""
    df = _keyed_match_data()
    grp = df.groupby("p_key", as_index=False).agg(
        matches_played=("p_win", "count"), wins=("p_win", "sum"))
    grp = grp.rename(columns={"p_key": "deck_key"})
    grp = grp[grp["matches_played"] >= min_matches].copy()
    _, nm, tm, em, _ = load_card_assets()
    records = [enrich_deck_record(r["deck_key"], int(r["matches_played"]),
               int(r["wins"]), nm, em, tm) for _, r in grp.iterrows()]
    return pd.DataFrame(records).sort_values(
        ["matches_played", "win_rate"], ascending=[False, False]).reset_index(drop=True)


@st.cache_data(show_spinner=True, ttl=3600)
def compute_head_to_head(player_key: str, opponent_key: str) -> dict | None:
    """Look up exact head-to-head stats for two deck keys (uses pre-computed keys)."""
    df = _keyed_match_data()

    # matches where player deck = player_key AND opponent deck = opponent_key
    fwd = df[(df["p_key"] == player_key) & (df["o_key"] == opponent_key)]

    # also check reverse (opponent as player1, player as player2)
    rev = df[(df["p_key"] == opponent_key) & (df["o_key"] == player_key)]

    fwd_wins = int(fwd["p_win"].sum())
    rev_wins = int((1 - rev["p_win"]).sum())  # reversed perspective
    total = len(fwd) + len(rev)

    if total == 0:
        return None

    wins = fwd_wins + rev_wins
    losses = total - wins
    return {
        "total_matches": total,
        "player_wins": wins,
        "opponent_wins": losses,
        "player_win_rate": round(wins / total * 100, 2),
    }


def estimate_deck_win_rate(cards: list[int], decks_df: pd.DataFrame) -> dict:
    """Estimate win-rate for a single deck from exact or similar historical decks."""
    dk = build_deck_key(cards)
    exact = decks_df[decks_df["deck_key"] == dk]
    if not exact.empty:
        row = exact.iloc[0]
        return {"win_rate": float(row["win_rate"]),
                "matches": int(row["matches_played"]),
                "source": "exact", "confidence": confidence_from_match_count(int(row["matches_played"]))}

    target = set(cards)
    rows = []
    for _, r in decks_df.iterrows():
        overlap = len(target & set(int(x) for x in r["card_ids"]))
        if overlap >= SIMILAR_DECK_OVERLAP:
            rows.append({"sim": overlap / 8.0, "mp": int(r["matches_played"]),
                         "wr": float(r["win_rate"])})
    if not rows:
        return {
            "win_rate": 50.0,
            "matches": 0,
            "source": "fallback",
            "confidence": confidence_from_match_count(matches_played=0),
        }

    sdf = pd.DataFrame(rows).sort_values(["sim", "mp"], ascending=False).head(TOP_SIMILAR)
    w = sdf["sim"] * sdf["mp"]
    wr = float((sdf["wr"] * w).sum() / w.sum())
    tm = int(sdf["mp"].sum())
    return {"win_rate": round(wr, 2), "matches": tm,
            "source": f"~{len(sdf)} similar decks", "confidence": confidence_from_match_count(tm)}


@st.cache_resource(show_spinner=False)
def _load_model_for_explanations():
    """Load model and feature schema for explanation generation."""
    try:
        model = load_xgboost_model()
    except Exception:
        model, _ = load_best_model()
    schema = load_feature_schema()
    return model, schema


def get_matchup_explanation_bullets(
    player_cards: list[int],
    opponent_cards: list[int],
    metadata_df: pd.DataFrame,
    player_win_prob: float | None = None,
) -> list[str]:
    """Generate explanation bullets for a matchup using the explanation engine."""
    try:
        model, feature_schema = _load_model_for_explanations()

        feature_df = build_feature_vector(
            deck_cards=player_cards,
            metadata_df=metadata_df,
            feature_schema=feature_schema,
            opponent_cards=opponent_cards,
        )

        if feature_df is None or feature_df.empty:
            return []

        return build_prediction_explanations(
            model=model,
            feature_df=feature_df,
            metadata_df=metadata_df,
            player_cards=[int(c) for c in player_cards],
            opponent_cards=[int(c) for c in opponent_cards],
            max_bullets=4,
            player_win_prob=player_win_prob,
        )
    except Exception:
        return []


def render_card_image(card_id: int, icon_map: dict, name_map: dict) -> None:
    url = icon_map.get(int(card_id))
    name = name_map.get(int(card_id), str(card_id))
    if url and isinstance(url, str) and url.strip():
        st.image(url, use_container_width=True)
    else:
        st.markdown(
            f"<div style='border:1px solid #c5d5ea;border-radius:10px;padding:8px;"
            f"text-align:center;min-height:90px;display:flex;align-items:center;"
            f"justify-content:center;font-size:12px;color:#6b7fa3;"
            f"background:#fff;'>{name}</div>", unsafe_allow_html=True)


def render_empty_slot(n: int):
    st.markdown(f"<div class='slot-card'>Slot {n}</div>", unsafe_allow_html=True)


def render_confidence(conf: str):
    cls = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}.get(conf, "conf-medium")
    st.markdown(f"<span class='conf-badge {cls}'>{conf} Confidence</span>",
                unsafe_allow_html=True)


# ── Deck builder state ─────────────────────────────────────────────
def _init_state():
    if "mu_player" not in st.session_state:
        st.session_state.mu_player = [None] * 8
    if "mu_opponent" not in st.session_state:
        st.session_state.mu_opponent = [None] * 8


def _add(side: str, cid: int):
    slots = st.session_state[side]
    if cid in slots:
        return
    for i in range(8):
        if slots[i] is None:
            slots[i] = int(cid)
            return


def _remove(side: str, idx: int):
    st.session_state[side][idx] = None


def _clear(side: str):
    st.session_state[side] = [None] * 8


def _filled(side: str) -> list[int]:
    return [c for c in st.session_state[side] if c is not None]


# ── Deck builder UI ────────────────────────────────────────────────
def render_deck_builder(label: str, side: str, card_df: pd.DataFrame,
                        name_map: dict, type_map: dict, elixir_map: dict,
                        icon_map: dict):
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)
    filled = _filled(side)

    # show 2 rows × 4 slots
    for row_start in (0, 4):
        cols = st.columns(4, gap="small")
        for i in range(4):
            idx = row_start + i
            with cols[i]:
                cid = st.session_state[side][idx]
                if cid is None:
                    render_empty_slot(idx + 1)
                else:
                    url = icon_map.get(int(cid), "")
                    if url:
                        st.markdown(f"<div class='deck-card-wrap'><img src='{url}'/></div>",
                                    unsafe_allow_html=True)
                    else:
                        render_card_image(cid, icon_map, name_map)
                    if st.button("✕", key=f"rm_{side}_{idx}", use_container_width=True):
                        _remove(side, idx)
                        st.rerun()

    # live stats row
    if filled:
        avg_e = compute_avg_elixir(filled, elixir_map)
        cyc = compute_cycle_cost(filled, elixir_map) if len(filled) >= 4 else 0
        tc = count_card_types(filled, type_map)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg Elixir", f"{avg_e:.1f}")
        c2.metric("Cycle", cyc if cyc else "—")
        c3.metric("Troops", tc["troop_count"])
        c4.metric("Spells", tc["spell_count"])
        c5.metric("Buildings", tc["building_count"])

    st.caption(f"{len(filled)}/8 cards selected")

    if st.button("Clear", key=f"clear_{side}", use_container_width=True):
        _clear(side)
        st.rerun()

    # card chooser (collapsed when full)
    if len(filled) < 8:
        search = st.text_input("Search cards", key=f"search_{side}",
                               placeholder="Type a card name…")
        avail = card_df.copy()
        if search.strip():
            avail = avail[avail["name"].str.contains(search.strip(), case=False, na=False)]
        avail = avail[~avail["card_id"].isin(set(filled))].reset_index(drop=True)
        avail["type"] = avail["card_id"].map(type_map).fillna("unknown")
        avail["elixir"] = avail["card_id"].map(elixir_map).fillna(0).astype(int)
        avail = avail.sort_values(["elixir", "name"]).reset_index(drop=True)

        ELIXIR_ICON = "https://cdn.royaleapi.com/static/img/ui/elixir.png"
        for cat_label, cat_key in [("Troops", "troop"), ("Spells", "spell"),
                                    ("Buildings", "building")]:
            section = avail[avail["type"] == cat_key].reset_index(drop=True)
            if section.empty:
                continue
            st.markdown(
                f"<div style='background:#dce6f5;padding:6px 14px;border-radius:6px;"
                f"color:#1a3a6e;font-weight:700;font-size:14px;margin:12px 0 8px;'>"
                f"{cat_label}</div>", unsafe_allow_html=True)
            per_row = 6
            for start in range(0, len(section), per_row):
                chunk = section.iloc[start:start + per_row]
                cols = st.columns(per_row, gap="small")
                for ci, (_, row) in enumerate(chunk.iterrows()):
                    with cols[ci]:
                        cid = int(row["card_id"])
                        url = icon_map.get(cid)
                        if url:
                            st.image(url, use_container_width=True)
                        ec = elixir_map.get(cid, "")
                        if ec:
                            st.markdown(
                                f"<div style='text-align:center'>"
                                f"<span class='elixir-badge'>"
                                f"<img src='{ELIXIR_ICON}'/>{ec}</span></div>",
                                unsafe_allow_html=True)
                        if st.button("Add", key=f"add_{side}_{cid}",
                                     use_container_width=True):
                            if len(filled) < 8:
                                _add(side, cid)
                                st.rerun()


# ── Main page ───────────────────────────────────────────────────────
def main():
    st.title("⚔️ Matchup Analysis")
    st.markdown(
        "<div style='margin-bottom:10px;color:#5a7394;font-size:15px;'>"
        "Build a <strong>Player</strong> deck and an <strong>Opponent</strong> deck, "
        "then click <em>Analyze Matchup</em> to see win probability, predicted winner, "
        "and head-to-head statistics."
        "</div>",
        unsafe_allow_html=True,
    )

    _init_state()
    card_df, name_map, type_map, elixir_map, icon_map = load_card_assets()

    # ── Two side-by-side deck builders ──────────────────────────────
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        render_deck_builder("Player Deck", "mu_player", card_df,
                            name_map, type_map, elixir_map, icon_map)
    with right_col:
        render_deck_builder("Opponent Deck", "mu_opponent", card_df,
                            name_map, type_map, elixir_map, icon_map)

    st.divider()

    p_cards = _filled("mu_player")
    o_cards = _filled("mu_opponent")
    ready = len(p_cards) == 8 and len(o_cards) == 8

    analyze = st.button("⚔️ Analyze Matchup", type="primary",
                        use_container_width=True, disabled=not ready)
    if not ready:
        st.info("Select exactly 8 cards for **both** the Player and Opponent decks.")
        return
    if not analyze:
        return

    # ── Compute stats ───────────────────────────────────────────────
    decks_df = build_deck_lookup(MIN_MATCHES)
    p_key = build_deck_key(p_cards)
    o_key = build_deck_key(o_cards)

    p_est = estimate_deck_win_rate(p_cards, decks_df)
    o_est = estimate_deck_win_rate(o_cards, decks_df)

    h2h = compute_head_to_head(p_key, o_key)

    # win probability: blend head-to-head (if available) with individual rates
    if h2h and h2h["total_matches"] >= 5:
        h2h_weight = min(h2h["total_matches"] / 100, 0.7)
        individual_prob = p_est["win_rate"] / (p_est["win_rate"] + o_est["win_rate"]) * 100 \
            if (p_est["win_rate"] + o_est["win_rate"]) > 0 else 50.0
        player_win_prob = round(
            h2h_weight * h2h["player_win_rate"] + (1 - h2h_weight) * individual_prob, 2)
    else:
        total = p_est["win_rate"] + o_est["win_rate"]
        player_win_prob = round(p_est["win_rate"] / total * 100, 2) if total > 0 else 50.0

    opponent_win_prob = round(100 - player_win_prob, 2)

    # Historical confidence
    combined_matches = p_est["matches"] + o_est["matches"]
    if h2h:
        combined_matches += h2h["total_matches"]
    historical_conf = confidence_from_match_count(combined_matches)

    # XGBoost-based probability estimate
    try:
        metadata_df = _load_metadata_df()
        feature_schema = load_feature_schema()
        player_model_prob = predict_probability_with_xgboost(
            deck_cards=p_cards,
            metadata_df=metadata_df,
            feature_schema=feature_schema,
        )
        opponent_model_prob = predict_probability_with_xgboost(
            deck_cards=o_cards,
            metadata_df=metadata_df,
            feature_schema=feature_schema,
        )

        if player_model_prob is not None and opponent_model_prob is not None:
            total_model_prob = player_model_prob + opponent_model_prob
            if total_model_prob > 0:
                player_win_prob = round((player_model_prob / total_model_prob) * 100.0, 2)
            else:
                player_win_prob = p_est["win_rate"]
        elif player_model_prob is not None:
            player_win_prob = round(player_model_prob * 100.0, 2)
        else:
            player_win_prob = p_est["win_rate"]

        conf_result = combine_confidence_signals(
            probability=player_win_prob / 100.0,
            historical_confidence_label=historical_conf,
            model_probabilities={"XGBoost": player_model_prob}
            if player_model_prob is not None
            else None,
            model_weight=0.6,
            historical_weight=0.4,
        )
        confidence = conf_result.label
    except Exception:
        # Fallback to historical confidence only if model loading fails
        confidence = historical_conf

    # archetypes
    p_arch = detect_archetype(p_cards, name_map, elixir_map)
    o_arch = detect_archetype(o_cards, name_map, elixir_map)

    # ── Predicted winner banner ─────────────────────────────────────
    if abs(player_win_prob - 50) < 1.0:
        banner_cls = "winner-draw"
        banner_text = "🤝 Even Matchup — Too Close to Call"
    elif player_win_prob > 50:
        banner_cls = "winner-player"
        banner_text = f"🏆 Predicted Winner: Player ({player_win_prob:.1f}%)"
    else:
        banner_cls = "winner-opponent"
        banner_text = f"🏆 Predicted Winner: Opponent ({opponent_win_prob:.1f}%)"

    st.markdown(f"<div class='winner-banner {banner_cls}'>{banner_text}</div>",
                unsafe_allow_html=True)

    # ── Win probability metrics ─────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Player Win Prob", f"{player_win_prob:.1f}%")
    m2.metric("Opponent Win Prob", f"{opponent_win_prob:.1f}%")
    m3.metric("Prediction Confidence", confidence)

    render_confidence(confidence)
    st.caption("Confidence reflects the amount of historical data supporting this matchup estimate.")

    # ── Meta card count badges ──────────────────────────────────────
    rankings_df = load_card_rankings()
    if rankings_df is not None and not rankings_df.empty:
        top20_names = set(rankings_df.head(20)["card_name"])
        p_names = [name_map.get(int(c), "") for c in p_cards]
        o_names = [name_map.get(int(c), "") for c in o_cards]
        p_meta = sum(1 for n in p_names if n in top20_names)
        o_meta = sum(1 for n in o_names if n in top20_names)
        mc1, mc2 = st.columns(2)
        mc1.markdown(
            f"<div class='note-box'>🔥 Player deck contains <strong>{p_meta}</strong> "
            f"top meta card(s)</div>",
            unsafe_allow_html=True,
        )
        mc2.markdown(
            f"<div class='note-box'>🔥 Opponent deck contains <strong>{o_meta}</strong> "
            f"top meta card(s)</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Side-by-side deck comparison ────────────────────────────────
    st.subheader("Deck Comparison")

    left, right = st.columns(2, gap="large")

    for col, cards, est, arch, label in [
        (left, p_cards, p_est, p_arch, "Player"),
        (right, o_cards, o_est, o_arch, "Opponent"),
    ]:
        with col:
            st.markdown(f"<div class='section-label'>{label} Deck</div>",
                        unsafe_allow_html=True)
            for row_start in (0, 4):
                img_cols = st.columns(4, gap="small")
                for i in range(4):
                    with img_cols[i]:
                        render_card_image(cards[row_start + i], icon_map, name_map)
            st.caption(" • ".join(name_map.get(int(c), str(c)) for c in cards))

            avg_e = compute_avg_elixir(cards, elixir_map)
            cyc = compute_cycle_cost(cards, elixir_map)
            tc = count_card_types(cards, type_map)

            s1, s2, s3 = st.columns(3)
            s1.metric("Archetype", arch)
            s2.metric("Avg Elixir", f"{avg_e:.2f}")
            s3.metric("Cycle Cost", cyc)

            s4, s5, s6, s7 = st.columns(4)
            s4.metric("Troops", tc["troop_count"])
            s5.metric("Spells", tc["spell_count"])
            s6.metric("Buildings", tc["building_count"])
            s7.metric("Meta Win Rate", f"{est['win_rate']:.1f}%")

            st.caption(f"Source: {est['source']} ({est['matches']:,} matches)")

    st.divider()

    # ── Matchup stats ───────────────────────────────────────────────
    st.subheader("Matchup Statistics")

    if h2h and h2h["total_matches"] > 0:
        st.success(f"Found **{h2h['total_matches']:,}** historical head-to-head matches "
                   f"between these exact decks.")
        h1, h2_, h3 = st.columns(3)
        h1.metric("Player Wins", h2h["player_wins"])
        h2_.metric("Opponent Wins", h2h["opponent_wins"])
        h3.metric("H2H Player Win Rate", f"{h2h['player_win_rate']:.1f}%")
    else:
        st.info("No direct head-to-head matches found between these exact decks. "
                "The prediction is based on each deck's individual historical performance "
                "against the overall meta.")

    # ── Advantage breakdown ─────────────────────────────────────────
    st.subheader("Advantage Breakdown")

    p_avg = compute_avg_elixir(p_cards, elixir_map)
    o_avg = compute_avg_elixir(o_cards, elixir_map)
    p_cyc = compute_cycle_cost(p_cards, elixir_map)
    o_cyc = compute_cycle_cost(o_cards, elixir_map)
    p_tc = count_card_types(p_cards, type_map)
    o_tc = count_card_types(o_cards, type_map)
    shared = set(p_cards) & set(o_cards)

    advantages: list[str] = []

    if p_avg < o_avg - 0.3:
        advantages.append("✅ **Player** has a lighter deck — faster cycle and chip pressure.")
    elif o_avg < p_avg - 0.3:
        advantages.append("✅ **Opponent** has a lighter deck — faster cycle and chip pressure.")

    if p_cyc < o_cyc - 1:
        advantages.append("✅ **Player** cycles cheaper (4-card cycle cost is lower).")
    elif o_cyc < p_cyc - 1:
        advantages.append("✅ **Opponent** cycles cheaper (4-card cycle cost is lower).")

    if p_tc["building_count"] > o_tc["building_count"]:
        advantages.append("✅ **Player** has more buildings — better defensive structure.")
    elif o_tc["building_count"] > p_tc["building_count"]:
        advantages.append("✅ **Opponent** has more buildings — better defensive structure.")

    if p_tc["spell_count"] > o_tc["spell_count"]:
        advantages.append("✅ **Player** has more spells — stronger spell cycle and removal.")
    elif o_tc["spell_count"] > p_tc["spell_count"]:
        advantages.append("✅ **Opponent** has more spells — stronger spell cycle and removal.")

    if shared:
        shared_names = [name_map.get(int(c), str(c)) for c in shared]
        advantages.append(f"🔄 **Shared cards** ({len(shared)}): {', '.join(shared_names)}")

    if not advantages:
        advantages.append("These decks are very evenly matched across all dimensions.")

    for line in advantages:
        st.markdown(line)

    st.divider()

    # ── Matchup Explanation ─────────────────────────────────────────
    st.subheader("Matchup Analysis")

    metadata_df_expl = _load_metadata_df()
    explanation_bullets = get_matchup_explanation_bullets(
        player_cards=p_cards,
        opponent_cards=o_cards,
        metadata_df=metadata_df_expl,
        player_win_prob=player_win_prob,
    )

    if explanation_bullets:
        bullets_html = "".join(
            f"<div class='explanation-bullet'>{bullet}</div>"
            for bullet in explanation_bullets
        )
        st.markdown(
            f"<div class='explanation-box'>"
            f"<div class='explanation-header'>⚔️ Matchup Breakdown</div>"
            f"{bullets_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='note-box'>"
            "Could not generate detailed matchup explanations. "
            "The prediction is based on each deck's historical performance and model estimates."
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
