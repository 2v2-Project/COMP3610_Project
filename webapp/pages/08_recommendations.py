"""
Deck Recommendations Page
=========================
Build a starting deck, get card-swap suggestions scored by XGBoost,
browse top-performing historical decks, and explore similar decks.
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
    confidence_from_match_count,
    predict_probability_with_xgboost,
)
from utils.model_loader import load_feature_schema
from utils.recommendation import (
    score_swaps_with_model,
    find_top_historical_decks,
    find_similar_decks,
)

st.set_page_config(page_title="Deck Recommendations", layout="wide")

from utils.ui_helpers import inject_fonts

inject_fonts()

# ── Custom CSS ──────────────────────────────────────────────────────
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
    .elixir-badge { display: inline-flex; align-items: center; gap: 2px;
        background: #e8f0fe; border-radius: 10px; padding: 1px 6px;
        font-size: 11px; font-weight: 700; color: #1a3a6e; }
    .elixir-badge img { width: 13px; height: 13px; }
    .deck-card-wrap { position: relative; display: inline-block; width: 100%; }
    .deck-card-wrap img { border-radius: 10px; width: 100%; }
    .swap-card {
        background: #ffffff; border: 1px solid #d0dbe8; border-radius: 12px;
        padding: 16px; margin: 8px 0;
        box-shadow: 0 1px 4px rgba(26,86,219,0.06);
    }
    .swap-positive { border-left: 4px solid #16a34a; }
    .swap-negative { border-left: 4px solid #ef4444; }
    .swap-neutral  { border-left: 4px solid #f59e0b; }
    .meta-deck-card {
        background: linear-gradient(135deg, #f0f7ff 0%, #e8f2ff 100%);
        border: 1px solid #c0d8f0; border-radius: 12px;
        padding: 16px 20px; margin: 10px 0;
    }
    .conf-badge { display: inline-block; padding: 4px 10px; border-radius: 999px;
        font-weight: 700; font-size: 12px; color: white; }
    .conf-high   { background: #16a34a; }
    .conf-medium { background: #f59e0b; }
    .conf-low    { background: #ef4444; }
    .rec-tab-header { font-size: 15px; font-weight: 600; color: #1a3a6e;
        margin-bottom: 4px; }
    .note-box { background: #ffffff; border: 1px solid #d0dbe8; border-radius: 10px;
        padding: 14px 16px; color: #3b536e; font-size: 14px;
        box-shadow: 0 1px 3px rgba(26,86,219,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

ELIXIR_ICON = "https://cdn.royaleapi.com/static/img/ui/elixir.png"

# ── Data paths ──────────────────────────────────────────────────────
DATA_PATHS = [
    Path("data/processed/clash_royale_clean.parquet"),
    Path("data/processed/final_ml_dataset.parquet"),
]
PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
OPPONENT_CARD_COLS = [f"player2.card{i}" for i in range(1, 9)]
PLAYER_CROWNS_COL = "player1.crowns"
OPPONENT_CROWNS_COL = "player2.crowns"

MIN_MATCHES = 20

# ── Cached loaders ─────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=3600)
def load_card_assets():
    name_map = get_card_names(force_refresh=False)
    type_map = get_card_types(force_refresh=False)
    elixir_map = get_elixir_costs(force_refresh=False)
    icon_map = get_icon_urls(force_refresh=False)

    card_df = pd.DataFrame(
        {"card_id": list(name_map.keys()), "name": [name_map[c] for c in name_map]}
    )
    banned = [
        "super ", "santa ", "terry", "party", "evolved ", "evolution",
        "mirror mode", "placeholder",
    ]
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
    for p in DATA_PATHS:
        if p.exists():
            df = pl.read_parquet(p)
            needed = set(
                PLAYER_CARD_COLS + OPPONENT_CARD_COLS
                + [PLAYER_CROWNS_COL, OPPONENT_CROWNS_COL]
            )
            if needed.issubset(set(df.columns)):
                return df.select(list(needed))
    raise FileNotFoundError("No parquet with player+opponent cards and crowns found.")


def _build_deck_key_col(df: pl.DataFrame, card_cols: list[str], alias: str) -> pl.DataFrame:
    return df.with_columns(
        pl.concat_list(card_cols)
        .list.eval(pl.element().cast(pl.Int64))
        .list.sort()
        .list.eval(pl.element().cast(pl.Utf8))
        .list.join(",")
        .alias(alias)
    )


@st.cache_data(show_spinner="Building deck index …", ttl=3600)
def _keyed_match_data() -> pd.DataFrame:
    df = load_match_data()
    df = _build_deck_key_col(df, PLAYER_CARD_COLS, "p_key")
    df = _build_deck_key_col(df, OPPONENT_CARD_COLS, "o_key")
    df = df.with_columns(
        (pl.col(PLAYER_CROWNS_COL) > pl.col(OPPONENT_CROWNS_COL))
        .cast(pl.Int8)
        .alias("p_win")
    )
    return df.select(["p_key", "o_key", "p_win"]).to_pandas()


@st.cache_data(show_spinner="Aggregating deck win rates …", ttl=3600)
def build_deck_lookup(min_matches: int) -> pd.DataFrame:
    df = _keyed_match_data()
    grp = df.groupby("p_key", as_index=False).agg(
        matches_played=("p_win", "count"), wins=("p_win", "sum")
    )
    grp = grp.rename(columns={"p_key": "deck_key"})
    grp = grp[grp["matches_played"] >= min_matches].copy()
    _, nm, tm, em, _ = load_card_assets()
    records = [
        enrich_deck_record(r["deck_key"], int(r["matches_played"]), int(r["wins"]), nm, em, tm)
        for _, r in grp.iterrows()
    ]
    return (
        pd.DataFrame(records)
        .sort_values(["matches_played", "win_rate"], ascending=[False, False])
        .reset_index(drop=True)
    )


# ── Session state helpers ──────────────────────────────────────────


def _init_state():
    if "rec_deck" not in st.session_state:
        st.session_state.rec_deck = [None] * 8


def _add(cid: int):
    slots = st.session_state.rec_deck
    if cid in slots:
        return
    for i in range(8):
        if slots[i] is None:
            slots[i] = int(cid)
            return


def _remove(idx: int):
    st.session_state.rec_deck[idx] = None


def _clear():
    st.session_state.rec_deck = [None] * 8


def _filled() -> list[int]:
    return [c for c in st.session_state.rec_deck if c is not None]


# ── UI components ──────────────────────────────────────────────────


def render_empty_slot(n: int):
    st.markdown(f"<div class='slot-card'>Slot {n}</div>", unsafe_allow_html=True)


def render_confidence(conf: str):
    cls = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}.get(
        conf, "conf-medium"
    )
    st.markdown(
        f"<span class='conf-badge {cls}'>{conf}</span>", unsafe_allow_html=True
    )


def render_deck_row(
    card_ids: list[int], icon_map: dict, name_map: dict, elixir_map: dict
):
    """Render a horizontal row of card images with elixir badges."""
    cols = st.columns(8, gap="small")
    for i, cid in enumerate(card_ids):
        with cols[i]:
            url = icon_map.get(int(cid), "")
            name = name_map.get(int(cid), str(cid))
            if url:
                st.markdown(
                    f"<div class='deck-card-wrap'><img src='{url}' title='{name}'/></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(name)
            ec = elixir_map.get(int(cid))
            if ec is not None:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<span class='elixir-badge'>"
                    f"<img src='{ELIXIR_ICON}'/>{ec}</span></div>",
                    unsafe_allow_html=True,
                )


def render_deck_builder(
    card_df: pd.DataFrame,
    name_map: dict,
    type_map: dict,
    elixir_map: dict,
    icon_map: dict,
):
    st.markdown(
        "<div class='section-label'>Your Deck</div>", unsafe_allow_html=True
    )
    filled = _filled()

    for row_start in (0, 4):
        cols = st.columns(4, gap="small")
        for i in range(4):
            idx = row_start + i
            with cols[i]:
                cid = st.session_state.rec_deck[idx]
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
                        st.caption(name_map.get(int(cid), str(cid)))
                    if st.button("✕", key=f"rec_rm_{idx}", use_container_width=True):
                        _remove(idx)
                        st.rerun()

    # live stats
    if filled:
        avg_e = compute_avg_elixir(filled, elixir_map)
        cyc = compute_cycle_cost(filled, elixir_map) if len(filled) >= 4 else 0
        tc = count_card_types(filled, type_map)
        arch = detect_archetype(filled, name_map, elixir_map)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Avg Elixir", f"{avg_e:.1f}")
        c2.metric("Cycle", cyc if cyc else "—")
        c3.metric("Troops", tc["troop_count"])
        c4.metric("Spells", tc["spell_count"])
        c5.metric("Buildings", tc["building_count"])
        c6.metric("Archetype", arch)

    st.caption(f"{len(filled)}/8 cards selected")

    if st.button("Clear Deck", key="rec_clear", use_container_width=True):
        _clear()
        st.rerun()

    # card chooser
    if len(filled) < 8:
        search = st.text_input(
            "Search cards", key="rec_search", placeholder="Type a card name…"
        )
        avail = card_df.copy()
        if search.strip():
            avail = avail[
                avail["name"].str.contains(search.strip(), case=False, na=False)
            ]
        avail = avail[~avail["card_id"].isin(set(filled))].reset_index(drop=True)
        avail["type"] = avail["card_id"].map(type_map).fillna("unknown")
        avail["elixir"] = avail["card_id"].map(elixir_map).fillna(0).astype(int)
        avail = avail.sort_values(["elixir", "name"]).reset_index(drop=True)

        for cat_label, cat_key in [
            ("Troops", "troop"),
            ("Spells", "spell"),
            ("Buildings", "building"),
        ]:
            section = avail[avail["type"] == cat_key].reset_index(drop=True)
            if section.empty:
                continue
            st.markdown(
                f"<div style='background:#dce6f5;padding:6px 14px;border-radius:6px;"
                f"color:#1a3a6e;font-weight:700;font-size:14px;margin:12px 0 8px;'>"
                f"{cat_label}</div>",
                unsafe_allow_html=True,
            )
            per_row = 6
            for start in range(0, len(section), per_row):
                chunk = section.iloc[start : start + per_row]
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
                                unsafe_allow_html=True,
                            )
                        if st.button(
                            "Add", key=f"rec_add_{cid}", use_container_width=True
                        ):
                            if len(_filled()) < 8:
                                _add(cid)
                                st.rerun()


# ── Main page ───────────────────────────────────────────────────────


def main():
    st.title("💡 Deck Recommendations")
    st.markdown(
        "<div style='margin-bottom:14px;color:#5a7394;font-size:15px;'>"
        "Build a deck to get <strong>card-swap suggestions</strong> powered by XGBoost, "
        "or explore the <strong>highest win-rate decks</strong> from the dataset."
        "</div>",
        unsafe_allow_html=True,
    )

    _init_state()
    card_df, name_map, type_map, elixir_map, icon_map = load_card_assets()

    # ── Deck builder (top section, full width) ──────────────────────
    with st.expander("🛠️ **Deck Builder** — select 8 cards", expanded=len(_filled()) < 8):
        render_deck_builder(card_df, name_map, type_map, elixir_map, icon_map)

    # ── Show current deck summary bar when 8 cards selected ─────────
    deck_cards = _filled()
    if len(deck_cards) == 8:
        st.markdown(
            "<div class='section-label' style='margin-top:4px;'>Your Current Deck</div>",
            unsafe_allow_html=True,
        )
        render_deck_row(deck_cards, icon_map, name_map, elixir_map)
        avg_e = compute_avg_elixir(deck_cards, elixir_map)
        arch = detect_archetype(deck_cards, name_map, elixir_map)
        cyc = compute_cycle_cost(deck_cards, elixir_map)
        tc = count_card_types(deck_cards, type_map)
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Archetype", arch)
        s2.metric("Avg Elixir", f"{avg_e:.1f}")
        s3.metric("Cycle", cyc if cyc else "—")
        s4.metric("Troops", tc["troop_count"])
        s5.metric("Spells", tc["spell_count"])
        s6.metric("Buildings", tc["building_count"])

    st.divider()

    # ── Results tabs (full width) ──────────────────────────────────
    tab_swap, tab_meta, tab_similar = st.tabs(
        ["🔄 Card Swap Suggestions", "🏆 Top Meta Decks", "🔍 Similar Decks"]
    )

    # ── Tab 1: Card swap suggestions ────────────────────────────
    with tab_swap:
        if len(deck_cards) != 8:
            st.info("Select exactly **8 cards** to get card-swap recommendations.")
        else:
            if st.button(
                "🔄 Analyze Swaps",
                type="primary",
                use_container_width=True,
                key="btn_swap",
            ):
                metadata_df = _load_metadata_df()
                feature_schema = load_feature_schema()

                with st.spinner("Scoring your deck …"):
                    base_prob = predict_probability_with_xgboost(
                        deck_cards=deck_cards,
                        metadata_df=metadata_df,
                        feature_schema=feature_schema,
                    )

                if base_prob is None:
                    st.error(
                        "Could not score the base deck. "
                        "Ensure the XGBoost model is available in `models/`."
                    )
                else:
                    st.markdown(
                        f"**Current deck win probability:** "
                        f"`{base_prob * 100:.1f}%`"
                    )

                    pool = list(card_df["card_id"])

                    with st.spinner("Evaluating card swaps (this may take a moment) …"):
                        swaps = score_swaps_with_model(
                            deck=deck_cards,
                            pool=pool,
                            metadata_df=metadata_df,
                            feature_schema=feature_schema,
                            base_prob=base_prob,
                            top_k=10,
                        )

                    if not swaps:
                        st.warning("No improving swaps found.")
                    else:
                        improvements = [s for s in swaps if s["delta"] > 0.001]
                        if not improvements:
                            st.success(
                                "Your deck is already well-optimised — "
                                "no single swap improves the predicted win rate."
                            )
                            improvements = swaps[:5]

                        for i, s in enumerate(improvements):
                            removed_name = name_map.get(s["removed"], str(s["removed"]))
                            added_name = name_map.get(s["added"], str(s["added"]))
                            delta_pct = s["delta"] * 100
                            new_prob_pct = s["win_prob"] * 100

                            if delta_pct > 0.1:
                                border_cls = "swap-positive"
                                arrow = "▲"
                                color = "#16a34a"
                            elif delta_pct < -0.1:
                                border_cls = "swap-negative"
                                arrow = "▼"
                                color = "#ef4444"
                            else:
                                border_cls = "swap-neutral"
                                arrow = "—"
                                color = "#f59e0b"

                            st.markdown(
                                f"<div class='swap-card {border_cls}'>"
                                f"<strong>#{i + 1}</strong> &nbsp; "
                                f"Remove <strong>{removed_name}</strong> → "
                                f"Add <strong>{added_name}</strong> &nbsp; "
                                f"<span style='color:{color};font-weight:700;'>"
                                f"{arrow} {delta_pct:+.2f}%</span> &nbsp; "
                                f"(new win prob: {new_prob_pct:.1f}%)"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                            with st.expander(f"Preview deck after swap #{i + 1}"):
                                render_deck_row(
                                    s["new_deck"], icon_map, name_map, elixir_map
                                )
                                new_avg = compute_avg_elixir(s["new_deck"], elixir_map)
                                new_arch = detect_archetype(
                                    s["new_deck"], name_map, elixir_map
                                )
                                mc1, mc2 = st.columns(2)
                                mc1.metric("Avg Elixir", f"{new_avg:.1f}")
                                mc2.metric("Archetype", new_arch)

    # ── Tab 2: Top meta decks ──────────────────────────────────
    with tab_meta:
        decks_df = build_deck_lookup(MIN_MATCHES)

        archetypes = ["All"] + sorted(
            decks_df["archetype"].dropna().unique().tolist()
        )
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            arch_filter = st.selectbox(
                "Filter by archetype", archetypes, key="meta_arch"
            )
        with col_f2:
            top_k = st.slider(
                "Number of decks", 5, 25, 10, key="meta_topk"
            )

        top_decks = find_top_historical_decks(
            deck_lookup_df=decks_df,
            name_map=name_map,
            elixir_map=elixir_map,
            type_map=type_map,
            archetype_filter=arch_filter,
            min_matches=MIN_MATCHES,
            top_k=top_k,
        )

        if top_decks.empty:
            st.warning("No decks found for the selected archetype filter.")
        else:
            for rank, (_, row) in enumerate(top_decks.iterrows(), start=1):
                card_ids = row["card_ids"]
                conf = row.get("confidence", "Low")

                st.markdown(
                    f"<div class='meta-deck-card'>"
                    f"<strong>#{rank}</strong> &nbsp; "
                    f"Win Rate: <strong>{row['win_rate']:.1f}%</strong> &nbsp; | &nbsp; "
                    f"Matches: <strong>{int(row['matches_played']):,}</strong> &nbsp; | &nbsp; "
                    f"Archetype: <strong>{row['archetype']}</strong> &nbsp; | &nbsp; "
                    f"Avg Elixir: <strong>{row['avg_elixir']:.1f}</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                render_deck_row(card_ids, icon_map, name_map, elixir_map)
                render_confidence(conf)
                st.markdown("---")

    # ── Tab 3: Similar decks ───────────────────────────────────
    with tab_similar:
        if len(deck_cards) != 8:
            st.info(
                "Select exactly **8 cards** in the deck builder "
                "to find similar historical decks."
            )
        else:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                overlap_min = st.slider(
                    "Minimum shared cards", 4, 7, 5, key="sim_overlap"
                )
            with col_s2:
                sim_topk = st.slider(
                    "Number of results", 5, 20, 10, key="sim_topk"
                )

            decks_df = build_deck_lookup(MIN_MATCHES)
            similar = find_similar_decks(
                deck=deck_cards,
                deck_lookup_df=decks_df,
                min_overlap=overlap_min,
                top_k=sim_topk,
            )

            if similar.empty:
                st.warning(
                    "No similar decks found in the dataset. "
                    "Try lowering the minimum shared cards."
                )
            else:
                st.success(f"Found **{len(similar)}** similar decks.")
                for rank, (_, row) in enumerate(similar.iterrows(), start=1):
                    card_ids = row["card_ids"]
                    shared = int(row["shared_cards"])
                    diff_cards = set(card_ids) - set(deck_cards)
                    diff_names = [
                        name_map.get(int(c), str(c)) for c in diff_cards
                    ]

                    st.markdown(
                        f"<div class='meta-deck-card'>"
                        f"<strong>#{rank}</strong> &nbsp; "
                        f"Win Rate: <strong>{row['win_rate']:.1f}%</strong> &nbsp; | &nbsp; "
                        f"Shared: <strong>{shared}/8</strong> cards &nbsp; | &nbsp; "
                        f"Matches: <strong>{int(row['matches_played']):,}</strong> &nbsp; | &nbsp; "
                        f"Archetype: <strong>{row['archetype']}</strong>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    render_deck_row(card_ids, icon_map, name_map, elixir_map)

                    if diff_names:
                        st.caption(
                            f"Different cards: **{', '.join(diff_names)}**"
                        )
                    render_confidence(row.get("confidence", "Low"))
                    st.markdown("---")


main()

