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
    enrich_deck_record,
)

st.set_page_config(page_title="Win Predictor", layout="wide")

st.markdown(
    """
    <style>
    section.main > div {
        max-width: 1100px;
        margin: auto;
    }

    div[data-testid="stMetric"] {
        background-color: #1e2a3a;
        border: 1px solid #2d3f54;
        padding: 10px 14px;
        border-radius: 8px;
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 12px !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
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
        background: #111827;
        border: 1px solid #253347;
        border-radius: 10px;
        padding: 14px 16px;
        color: #cbd5e1;
        font-size: 14px;
    }

    .section-label {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .slot-card {
        border: 1px dashed #3b4c63;
        border-radius: 10px;
        min-height: 78px;
        background: #111827;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        font-size: 13px;
        text-align: center;
        padding: 8px;
    }

    .deck-helper {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
    }

    .card-name {
        text-align: center;
        color: #cbd5e1;
        font-size: 12px;
        line-height: 1.2;
        min-height: 32px;
        margin-top: 4px;
        margin-bottom: 6px;
    }

    .deck-area {
        max-width: 860px;
        margin: 0 auto;
    }

    .chooser-area {
        max-width: 980px;
        margin: 0 auto;
    }

    .builder-gap {
        height: 14px;
    }

    .builder-buttons-gap {
        height: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATHS = [
    Path("data/processed/clash_royale_clean.parquet"),
    Path("data/processed/final_ml_dataset.parquet"),
]

PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
PLAYER_CROWNS_COL = "player1.crowns"
OPPONENT_CROWNS_COL = "player2.crowns"

SIMILAR_DECK_MATCH_THRESHOLD = 20
TOP_SIMILAR_DECKS = 25


def load_card_assets():
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
        "party"
        "evolved ",
        "evolution",
        "mirror mode",
        "placeholder",
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


@st.cache_data(show_spinner=True)
def load_match_data() -> pl.DataFrame:
    for path in DATA_PATHS:
        if path.exists():
            df = pl.read_parquet(path)
            required_cols = set(PLAYER_CARD_COLS + [PLAYER_CROWNS_COL, OPPONENT_CROWNS_COL])
            if required_cols.issubset(set(df.columns)):
                return df.select(PLAYER_CARD_COLS + [PLAYER_CROWNS_COL, OPPONENT_CROWNS_COL])

    raise FileNotFoundError(
        "Could not find a suitable parquet file with player deck cards and crown columns."
    )


@st.cache_data(show_spinner=True)
def build_popular_decks_table(min_matches: int) -> pd.DataFrame:
    df = load_match_data()
    pdf = df.to_pandas()

    pdf["deck_key"] = pdf[PLAYER_CARD_COLS].apply(
        lambda row: build_deck_key([int(x) for x in row.tolist()]),
        axis=1,
    )
    pdf["player_win"] = (pdf[PLAYER_CROWNS_COL] > pdf[OPPONENT_CROWNS_COL]).astype(int)

    grouped = (
        pdf.groupby("deck_key", as_index=False)
        .agg(
            matches_played=("player_win", "count"),
            wins=("player_win", "sum"),
        )
    )

    grouped = grouped[grouped["matches_played"] >= min_matches].copy()

    _, name_map, type_map, elixir_map, _ = load_card_assets()

    records = [
        enrich_deck_record(
            deck_key=row["deck_key"],
            matches_played=int(row["matches_played"]),
            wins=int(row["wins"]),
            name_map=name_map,
            elixir_map=elixir_map,
            type_map=type_map,
        )
        for _, row in grouped.iterrows()
    ]

    result = pd.DataFrame(records)
    result = result.sort_values(["matches_played", "win_rate"], ascending=[False, False]).reset_index(drop=True)
    return result


def render_confidence_badge(confidence: str):
    conf_cls = {
        "High": "conf-high",
        "Medium": "conf-medium",
        "Low": "conf-low",
    }.get(confidence, "conf-medium")

    st.markdown(
        f"<span class='conf-badge {conf_cls}'>{confidence} Confidence</span>",
        unsafe_allow_html=True,
    )


def render_card_image(card_id: int, icon_map: dict[int, str], name_map: dict[int, str], small: bool = False) -> None:
    icon_url = icon_map.get(int(card_id))
    card_name = name_map.get(int(card_id), str(card_id))

    if icon_url and isinstance(icon_url, str) and icon_url.strip():
        if small:
            st.image(icon_url, width=86)
        else:
            st.image(icon_url, use_container_width=True)
    else:
        min_height = "72px" if small else "90px"
        st.markdown(
            f"""
            <div style="
                border:1px solid #333;
                border-radius:10px;
                padding:8px 6px;
                text-align:center;
                min-height:{min_height};
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:12px;
                color:#bbb;
                background:#111827;">
                {card_name}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_empty_slot(slot_num: int):
    st.markdown(
        f"""
        <div class="slot-card">
            Empty Slot {slot_num}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_confidence_from_matches(matches: int) -> str:
    if matches >= 1000:
        return "High"
    if matches >= 200:
        return "Medium"
    return "Low"


def get_confidence_from_similar_decks(similar_count: int, total_matches: int) -> str:
    if similar_count >= 8 and total_matches >= 3000:
        return "High"
    if similar_count >= 4 and total_matches >= 1000:
        return "Medium"
    return "Low"


def estimate_from_similar_decks(selected_cards: list[int], decks_df: pd.DataFrame) -> dict | None:
    target_set = set(selected_cards)
    candidate_rows = []

    for _, row in decks_df.iterrows():
        deck_cards = set(int(x) for x in row["card_ids"])
        overlap = len(target_set.intersection(deck_cards))

        if overlap >= 6:
            similarity_score = overlap / 8.0
            candidate_rows.append(
                {
                    "similarity": similarity_score,
                    "matches_played": int(row["matches_played"]),
                    "win_rate": float(row["win_rate"]),
                    "archetype": row["archetype"],
                }
            )

    if not candidate_rows:
        return None

    sim_df = pd.DataFrame(candidate_rows)
    sim_df = sim_df.sort_values(
        ["similarity", "matches_played", "win_rate"],
        ascending=[False, False, False],
    ).head(TOP_SIMILAR_DECKS)

    weight = sim_df["similarity"] * sim_df["matches_played"]
    estimated_win_rate = float((sim_df["win_rate"] * weight).sum() / weight.sum())
    total_matches = int(sim_df["matches_played"].sum())
    similar_count = int(len(sim_df))
    archetype_mode = sim_df["archetype"].mode().iloc[0] if not sim_df.empty else "Unknown"

    return {
        "estimated_win_rate": round(estimated_win_rate, 2),
        "confidence": get_confidence_from_similar_decks(similar_count, total_matches),
        "historical_matches": total_matches,
        "similar_deck_count": similar_count,
        "archetype_guess": archetype_mode,
    }


def get_card_type_counts(selected_cards: list[int], type_map: dict[int, str]) -> tuple[int, int, int]:
    troop_count = 0
    spell_count = 0
    building_count = 0

    for cid in selected_cards:
        ctype = type_map.get(int(cid), "")
        if ctype == "troop":
            troop_count += 1
        elif ctype == "spell":
            spell_count += 1
        elif ctype == "building":
            building_count += 1

    return troop_count, spell_count, building_count


def compute_avg_elixir(selected_cards: list[int], elixir_map: dict[int, int]) -> float:
    costs = [elixir_map.get(int(cid), 0) for cid in selected_cards]
    valid_costs = [c for c in costs if c is not None]
    return round(sum(valid_costs) / len(valid_costs), 2) if valid_costs else 0.0


def compute_cycle_cost(selected_cards: list[int], elixir_map: dict[int, int]) -> int:
    costs = sorted(elixir_map.get(int(cid), 99) for cid in selected_cards)
    valid_costs = [c for c in costs if c != 99]
    return int(sum(valid_costs[:4])) if len(valid_costs) >= 4 else 0


def detect_simple_archetype(
    selected_cards: list[int],
    name_map: dict[int, str],
    elixir_map: dict[int, int],
) -> str:
    names = {name_map.get(int(cid), "").lower() for cid in selected_cards}
    avg_elixir = compute_avg_elixir(selected_cards, elixir_map)

    def has(*keywords: str) -> bool:
        return any(keyword.lower() in card_name for keyword in keywords for card_name in names)

    if avg_elixir <= 3.3 and (has("hog rider") or has("miner") or has("wall breakers")):
        return "Cycle"
    if avg_elixir >= 4.3 and (has("golem") or has("giant") or has("electro giant") or has("lava hound")):
        return "Beatdown"
    if has("x-bow") or has("mortar"):
        return "Siege"
    if has("goblin barrel") or has("princess"):
        return "Bait"
    if has("graveyard"):
        return "Graveyard"
    if has("balloon"):
        return "Loon"
    if has("sparky"):
        return "Sparky"
    if avg_elixir <= 3.6:
        return "Control"
    return "Unknown"


def build_explanations(
    exact_match: pd.Series | None,
    similar_result: dict | None,
    archetype: str,
    avg_elixir: float,
    cycle_cost: int,
    troop_count: int,
    spell_count: int,
    building_count: int,
) -> list[str]:
    bullets = []

    if exact_match is not None:
        bullets.append(
            f"This exact deck exists in the dataset, so the estimate is based on {int(exact_match['matches_played']):,} historical matches."
        )
    elif similar_result is not None:
        bullets.append(
            f"This estimate uses {similar_result['similar_deck_count']} similar historical decks with overlapping cards."
        )
    else:
        bullets.append("This deck has limited historical support, so the estimate is only a rough baseline.")

    if avg_elixir <= 3.0:
        bullets.append("This is a low-cost deck, which usually supports faster cycling and constant pressure.")
    elif avg_elixir >= 4.3:
        bullets.append("This is a heavier deck, which usually leans toward slower but stronger pushes.")
    else:
        bullets.append("This deck has a balanced elixir profile between fast cycle and heavy beatdown.")

    bullets.append(
        f"Cycle Cost is {cycle_cost}, which is the total elixir cost of the 4 cheapest cards in the deck."
    )
    bullets.append(
        f"Deck composition: {troop_count} troops, {spell_count} spells, and {building_count} buildings."
    )

    if archetype != "Unknown":
        bullets.append(f"This deck most closely resembles a {archetype} strategy.")

    return bullets[:5]


def init_deck_builder_state():
    if "deck_slots" not in st.session_state:
        st.session_state.deck_slots = [None] * 8


def add_card_to_deck(card_id: int):
    if card_id in st.session_state.deck_slots:
        return

    for i in range(8):
        if st.session_state.deck_slots[i] is None:
            st.session_state.deck_slots[i] = int(card_id)
            return


def remove_card_from_slot(slot_idx: int):
    st.session_state.deck_slots[slot_idx] = None


def clear_deck():
    st.session_state.deck_slots = [None] * 8


def main():
    st.title("🎯 Win Predictor")
    st.markdown(
        """
        <div style='margin-bottom:10px;color:#94a3b8;font-size:15px;'>
            Build a deck like you would in Clash Royale, then estimate its overall strength against
            the meta using historical deck performance and similar-deck signals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    init_deck_builder_state()

    card_df, name_map, type_map, elixir_map, icon_map = load_card_assets()
    decks_df = build_popular_decks_table(min_matches=SIMILAR_DECK_MATCH_THRESHOLD)

    st.markdown("<div class='deck-area'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Your Deck</div>", unsafe_allow_html=True)

    top_slots = st.columns(4, gap="small")
    for i in range(4):
        with top_slots[i]:
            card_id = st.session_state.deck_slots[i]
            if card_id is None:
                render_empty_slot(i + 1)
            else:
                render_card_image(card_id, icon_map, name_map, small=True)
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"remove_slot_{i}", use_container_width=True):
                    remove_card_from_slot(i)
                    st.rerun()

    st.markdown("<div class='builder-gap'></div>", unsafe_allow_html=True)

    bottom_slots = st.columns(4, gap="small")
    for i in range(4, 8):
        with bottom_slots[i - 4]:
            card_id = st.session_state.deck_slots[i]
            if card_id is None:
                render_empty_slot(i + 1)
            else:
                render_card_image(card_id, icon_map, name_map, small=True)
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"remove_slot_{i}", use_container_width=True):
                    remove_card_from_slot(i)
                    st.rerun()

    filled_cards = [cid for cid in st.session_state.deck_slots if cid is not None]

    st.markdown("<div class='builder-buttons-gap'></div>", unsafe_allow_html=True)

    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        if st.button("Clear Deck", use_container_width=True):
            clear_deck()
            st.rerun()
    with action_col2:
        analyze_clicked = st.button(
            "Analyze Deck",
            type="primary",
            use_container_width=True,
            disabled=len(filled_cards) != 8,
        )

    st.caption(f"{len(filled_cards)}/8 cards selected")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    with st.expander("Choose Cards", expanded=False):
        st.markdown("<div class='chooser-area'>", unsafe_allow_html=True)

        search_text = st.text_input("Search cards", placeholder="Type a card name...")

        available_df = card_df.copy()

        if search_text.strip():
            available_df = available_df[
                available_df["name"].str.contains(search_text.strip(), case=False, na=False)
            ]

        selected_set = set(filled_cards)
        available_df = available_df[~available_df["card_id"].isin(selected_set)].reset_index(drop=True)

        if available_df.empty:
            st.info("No cards match your search, or all matching cards are already in your deck.")
        else:
            for start in range(0, len(available_df), 6):
                row_df = available_df.iloc[start:start + 6]
                cols = st.columns(6, gap="small")

                for idx, (_, row) in enumerate(row_df.iterrows()):
                    with cols[idx]:
                        cid = int(row["card_id"])
                        render_card_image(cid, icon_map, name_map, small=True)
                        st.markdown(
                            f"<div class='card-name'>{row['name']}</div>",
                            unsafe_allow_html=True,
                        )
                        if st.button("Add", key=f"add_{cid}", use_container_width=True):
                            if len(filled_cards) < 8:
                                add_card_to_deck(cid)
                                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    if len(filled_cards) != 8:
        st.markdown(
            """
            <div class='note-box'>
                Select exactly 8 cards, then click <strong>Analyze Deck</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    selected_cards = filled_cards

    if not analyze_clicked:
        return

    avg_elixir = compute_avg_elixir(selected_cards, elixir_map)
    cycle_cost = compute_cycle_cost(selected_cards, elixir_map)
    troop_count, spell_count, building_count = get_card_type_counts(selected_cards, type_map)
    archetype = detect_simple_archetype(selected_cards, name_map, elixir_map)

    deck_key = build_deck_key(selected_cards)
    exact_rows = decks_df[decks_df["deck_key"] == deck_key]
    exact_match = exact_rows.iloc[0] if not exact_rows.empty else None
    similar_result = None

    if exact_match is not None:
        estimated_win_rate = float(exact_match["win_rate"])
        confidence = get_confidence_from_matches(int(exact_match["matches_played"]))
        historical_matches = int(exact_match["matches_played"])
        source_label = "Exact historical deck match"
        display_archetype = exact_match["archetype"]
    else:
        similar_result = estimate_from_similar_decks(selected_cards, decks_df)

        if similar_result is not None:
            estimated_win_rate = float(similar_result["estimated_win_rate"])
            confidence = similar_result["confidence"]
            historical_matches = int(similar_result["historical_matches"])
            source_label = f"Estimated from {similar_result['similar_deck_count']} similar decks"
            display_archetype = archetype if archetype != "Unknown" else similar_result["archetype_guess"]
        else:
            estimated_win_rate = 50.0
            confidence = "Low"
            historical_matches = 0
            source_label = "Fallback estimate"
            display_archetype = archetype

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Estimated Meta Win Rate", f"{estimated_win_rate:.2f}%")
    metric2.metric("Archetype", display_archetype)
    metric3.metric("Avg Elixir", f"{avg_elixir:.2f}")
    metric4.metric("Cycle Cost", cycle_cost)

    render_confidence_badge(confidence)
    st.caption("Confidence reflects how much historical support exists for this estimate.")

    st.divider()

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("<div class='section-label'>Deck Stats</div>", unsafe_allow_html=True)

        preview_top = st.columns(4, gap="small")
        for i in range(4):
            with preview_top[i]:
                render_card_image(selected_cards[i], icon_map, name_map)

        preview_bottom = st.columns(4, gap="small")
        for i in range(4, 8):
            with preview_bottom[i - 4]:
                render_card_image(selected_cards[i], icon_map, name_map)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric("Troops", troop_count)
        stat2.metric("Spells", spell_count)
        stat3.metric("Buildings", building_count)
        stat4.metric("Historical Matches", f"{historical_matches:,}")

        st.caption(" • ".join([name_map[int(cid)] for cid in selected_cards]))
        st.caption("Cycle Cost = total elixir of the 4 cheapest cards in the deck.")

    with right:
        st.markdown("<div class='section-label'>Prediction Notes</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='note-box'>
                <strong>Estimate source:</strong> {source_label}<br>
                <strong>Archetype:</strong> {display_archetype}<br>
                <strong>Historical matches used:</strong> {historical_matches:,}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        explanation_bullets = build_explanations(
            exact_match=exact_match,
            similar_result=similar_result,
            archetype=display_archetype,
            avg_elixir=avg_elixir,
            cycle_cost=cycle_cost,
            troop_count=troop_count,
            spell_count=spell_count,
            building_count=building_count,
        )

        for bullet in explanation_bullets:
            st.markdown(f"- {bullet}")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if exact_match is not None:
            st.success("This estimate uses the exact same deck from the historical dataset.")
        elif similar_result is not None:
            st.info("This exact deck was not found enough, so the estimate is based on similar historical decks.")
        else:
            st.warning("This deck has very little historical support, so the estimate is only a rough baseline.")


if __name__ == "__main__":
    main()