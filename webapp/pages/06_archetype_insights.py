"""
Archetype Insights Page
=======================
Task 33 — Archetype vs archetype win rates displayed as heatmap and table.
Task 35 — SHAP feature importance (global + summary) when a trained model exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from utils.deck_helpers import (
    compute_avg_elixir,
    count_card_types,
    detect_archetype,
)
from utils.metadata import get_card_names, get_card_types, get_elixir_costs, get_icon_urls
from utils.ui_helpers import inject_fonts

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Archetype Insights", layout="wide")
inject_fonts()

# ── page CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
section.main > div { max-width: 1140px; margin: auto; }

.arch-pill-row {
    display: flex; flex-wrap: wrap; gap: 10px;
    justify-content: center; margin-bottom: 24px;
}
.arch-pill {
    background: #ffffff; border: 1px solid #d0dbe8;
    border-radius: 999px; padding: 8px 18px;
    box-shadow: 0 1px 4px rgba(26,86,219,0.06);
    display: inline-flex; align-items: center; gap: 8px;
}
.arch-pill .ap-name { font-weight: 700; color: #1a3a6e; font-size: 14px; }
.arch-pill .ap-stat { color: #5a7394; font-size: 12px; }

.section-label {
    color: #6b7fa3; font-size: 13px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── constants ───────────────────────────────────────────────────────
DATA_PATHS = [
    Path("data/processed/clash_royale_clean.parquet"),
    Path("data/processed/final_ml_dataset.parquet"),
]
PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
OPPONENT_CARD_COLS = [f"player2.card{i}" for i in range(1, 9)]
PLAYER_CROWNS = "player1.crowns"
OPPONENT_CROWNS = "player2.crowns"

MIN_MATCHUPS = 20  # minimum games for a cell to be shown


# ── data loading ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def _load_card_maps():
    name_map = get_card_names(force_refresh=False)
    type_map = get_card_types(force_refresh=False)
    elixir_map = get_elixir_costs(force_refresh=False)
    icon_map = get_icon_urls(force_refresh=False)
    return name_map, type_map, elixir_map, icon_map


@st.cache_data(show_spinner=True, ttl=3600)
def _load_matches() -> pl.DataFrame:
    needed = set(PLAYER_CARD_COLS + OPPONENT_CARD_COLS + [PLAYER_CROWNS, OPPONENT_CROWNS])
    for p in DATA_PATHS:
        if p.exists():
            df = pl.read_parquet(p)
            if needed.issubset(df.columns):
                return df.select(list(needed))
    raise FileNotFoundError("No parquet with required columns found.")


def _vectorized_archetype(pdf: pd.DataFrame, card_cols: list[str],
                          name_map: dict, elixir_map: dict) -> np.ndarray:
    """Vectorised archetype detection — orders of magnitude faster than row-by-row apply."""
    # Pre-map every card column to its name (lowercase) and elixir cost
    name_arrays: list[pd.Series] = []
    elixir_arrays: list[pd.Series] = []
    for col in card_cols:
        ids = pdf[col].astype(int)
        name_arrays.append(ids.map(lambda x: name_map.get(x, "").lower()))
        elixir_arrays.append(ids.map(lambda x: elixir_map.get(x, 0)))

    avg_elixir = pd.concat(elixir_arrays, axis=1).mean(axis=1)

    def _has_any(keywords: list[str]) -> pd.Series:
        mask = pd.Series(False, index=pdf.index)
        for na in name_arrays:
            for kw in keywords:
                mask |= na.str.contains(kw, case=False, na=False)
        return mask

    # Rules in priority order (first match wins, same as detect_archetype)
    conditions = [
        (avg_elixir <= 3.3) & _has_any(["hog rider", "miner", "wall breakers"]),
        (avg_elixir >= 4.3) & _has_any(["golem", "giant", "electro giant", "lava hound"]),
        _has_any(["x-bow", "mortar"]),
        _has_any(["goblin barrel", "princess"]),
        _has_any(["royal giant", "pekka", "bandit", "battle ram"]),
        _has_any(["graveyard"]),
        _has_any(["balloon"]),
        _has_any(["sparky"]),
        _has_any(["miner"]) & (avg_elixir <= 3.8),
        (avg_elixir <= 3.6),
    ]
    choices = [
        "Cycle", "Beatdown", "Siege", "Bait", "Bridge Spam",
        "Graveyard", "Loon", "Sparky", "Miner Control", "Control",
    ]
    return np.select(conditions, choices, default="Unknown")


@st.cache_data(show_spinner="Computing archetype matchups...", ttl=3600)
def build_archetype_matchup_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (matchup_matrix, archetype_summary, per_match_df).
    matchup_matrix: rows = player archetype, cols = opponent archetype, values = player win rate.
    """
    name_map, _, elixir_map, _ = _load_card_maps()
    df = _load_matches()

    # cap sample size for speed (200 K is ample for stable archetype stats)
    if len(df) > 200_000:
        df = df.sample(n=200_000, seed=42)

    pdf = df.to_pandas()

    pdf["p1_arch"] = _vectorized_archetype(pdf, PLAYER_CARD_COLS, name_map, elixir_map)
    pdf["p2_arch"] = _vectorized_archetype(pdf, OPPONENT_CARD_COLS, name_map, elixir_map)
    pdf["p1_win"] = (pdf[PLAYER_CROWNS] > pdf[OPPONENT_CROWNS]).astype(int)

    # ── archetype summary ──
    arch_stats = (
        pdf.groupby("p1_arch")
        .agg(
            matches=("p1_win", "count"),
            wins=("p1_win", "sum"),
        )
        .assign(win_rate=lambda d: d["wins"] / d["matches"] * 100)
        .sort_values("matches", ascending=False)
        .reset_index()
        .rename(columns={"p1_arch": "archetype"})
    )

    # ── matchup matrix ──
    cross = (
        pdf.groupby(["p1_arch", "p2_arch"])
        .agg(
            matches=("p1_win", "count"),
            wins=("p1_win", "sum"),
        )
        .assign(win_rate=lambda d: d["wins"] / d["matches"] * 100)
        .reset_index()
    )

    # filter low-sample cells
    cross = cross[cross["matches"] >= MIN_MATCHUPS]

    # pivot to matrix
    matrix = cross.pivot_table(
        index="p1_arch", columns="p2_arch", values="win_rate"
    )

    # order by popularity
    order = arch_stats["archetype"].tolist()
    present = [a for a in order if a in matrix.index and a in matrix.columns]
    matrix = matrix.loc[present, present]

    return matrix, arch_stats, cross


# ── SHAP section ────────────────────────────────────────────────────
def _try_shap_analysis():
    """Attempt SHAP analysis. Returns (fig_importance, fig_summary, top_features, model_name) or None."""
    try:
        from utils.model_loader import load_xgboost_model, load_best_model, load_feature_schema
        from utils.preprocess import build_feature_vector
        from utils.shap_utils import get_shap_explainer, compute_shap_values, get_top_shap_features
        import shap
    except ImportError:
        return None

    # load model
    try:
        model = load_xgboost_model()
        model_name = "XGBoost"
    except FileNotFoundError:
        try:
            model, model_name = load_best_model()
        except FileNotFoundError:
            return None

    # load feature schema
    try:
        feature_schema = load_feature_schema()
    except FileNotFoundError:
        return None

    # build a sample feature matrix from popular decks
    name_map, type_map, elixir_map, _ = _load_card_maps()
    from utils.metadata import get_card_metadata
    metadata_df = get_card_metadata(force_refresh=False)

    df = _load_matches()
    if len(df) > 500_000:
        df = df.sample(n=500_000, seed=42)

    sample_rows = df.to_pandas().head(200)

    feature_rows = []
    for _, row in sample_rows.iterrows():
        cards = [int(row[c]) for c in PLAYER_CARD_COLS]
        try:
            fv = build_feature_vector(
                deck_cards=cards,
                metadata_df=metadata_df,
                feature_schema=feature_schema,
            )
            if fv is not None and not fv.empty:
                feature_rows.append(fv)
        except Exception:
            continue

    if len(feature_rows) < 20:
        return None

    X = pd.concat(feature_rows, ignore_index=True)

    # compute SHAP
    try:
        shap_values = compute_shap_values(model, X)
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None

    feature_names = list(X.columns)
    top_features = get_top_shap_features(shap_values, feature_names, top_n=15)

    # ── Global importance bar chart ──
    tf_df = pd.DataFrame(top_features, columns=["Feature", "Mean |SHAP|"])
    fig_importance = px.bar(
        tf_df.iloc[::-1],
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title=f"Global Feature Importance ({model_name})",
        color="Mean |SHAP|",
        color_continuous_scale="Blues",
    )
    fig_importance.update_layout(
        plot_bgcolor="#f8fbff",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#1a3a6e",
        showlegend=False,
        coloraxis_showscale=False,
        height=480,
    )

    # ── SHAP summary (beeswarm) ──
    try:
        shap_exp = shap.Explanation(
            values=shap_values,
            data=X.values,
            feature_names=feature_names,
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_summary, ax = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(shap_exp, show=False)
        fig_summary = plt.gcf()
    except Exception:
        fig_summary = None

    return fig_importance, fig_summary, top_features, model_name


# ── main ────────────────────────────────────────────────────────────
def main():
    st.title("🎨 Archetype Insights")
    st.markdown(
        "<div style='color:#5a7394;font-size:15px;margin-bottom:16px;'>"
        "Archetype vs archetype win rates, usage breakdowns, and model feature importance."
        "</div>",
        unsafe_allow_html=True,
    )

    matrix, arch_stats, cross = build_archetype_matchup_data()

    # ── Summary pills ──
    pills_html = '<div class="arch-pill-row">'
    for _, r in arch_stats.iterrows():
        pills_html += (
            f"<div class='arch-pill'>"
            f"<span class='ap-name'>{r['archetype']}</span>"
            f"<span class='ap-stat'>{int(r['matches']):,} games &middot; {r['win_rate']:.1f}% WR</span>"
            f"</div>"
        )
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)

    # ── Tabs ──
    tab_heatmap, tab_table, tab_shap = st.tabs(
        ["Matchup Heatmap", "Matchup Table", "Feature Importance (SHAP)"]
    )

    # ── Tab 1: Heatmap ──
    with tab_heatmap:
        st.markdown("<div class='section-label'>Archetype vs Archetype Win Rate</div>", unsafe_allow_html=True)
        st.caption("Cell value = Player (row) win rate vs Opponent (column). Green > 50 %, Red < 50 %.")

        heatmap_data = matrix.copy()

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.tolist(),
                y=heatmap_data.index.tolist(),
                colorscale=[
                    [0.0, "#ef4444"],
                    [0.5, "#fef9c3"],
                    [1.0, "#22c55e"],
                ],
                zmin=35,
                zmax=65,
                text=np.where(
                    np.isnan(heatmap_data.values),
                    "",
                    np.char.add(
                        np.char.mod("%.1f", np.nan_to_num(heatmap_data.values)),
                        np.full(heatmap_data.values.shape, "%"),
                    ),
                ),
                texttemplate="%{text}",
                textfont=dict(size=12),
                hovertemplate="Player: %{y}<br>Opponent: %{x}<br>Win Rate: %{z:.1f}%<extra></extra>",
                colorbar=dict(title="Win %", ticksuffix="%"),
            )
        )
        fig.update_layout(
            xaxis_title="Opponent Archetype",
            yaxis_title="Player Archetype",
            height=520,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#1a3a6e",
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Table ──
    with tab_table:
        st.markdown("<div class='section-label'>Detailed Matchup Table</div>", unsafe_allow_html=True)

        display_cross = cross.copy()
        display_cross = display_cross.rename(columns={
            "p1_arch": "Player Archetype",
            "p2_arch": "Opponent Archetype",
            "matches": "Matches",
            "wins": "Wins",
            "win_rate": "Win Rate %",
        })
        display_cross["Win Rate %"] = display_cross["Win Rate %"].round(1)
        display_cross = display_cross.sort_values(
            ["Player Archetype", "Win Rate %"], ascending=[True, False]
        )
        st.dataframe(
            display_cross,
            use_container_width=True,
            hide_index=True,
            height=500,
        )

    # ── Tab 3: SHAP ──
    with tab_shap:
        st.markdown("<div class='section-label'>SHAP Feature Importance</div>", unsafe_allow_html=True)
        st.caption(
            "Which features most influence the model's win prediction? "
            "Computed via SHAP on a sample of match feature vectors."
        )

        with st.spinner("Running SHAP analysis..."):
            shap_result = _try_shap_analysis()

        if shap_result is None:
            st.info(
                "No trained model found in `models/` directory. "
                "Run the training pipeline (scripts 09-13) to generate a model, "
                "then SHAP analysis will appear here automatically."
            )
        else:
            fig_imp, fig_summary, top_feats, model_name = shap_result

            st.plotly_chart(fig_imp, use_container_width=True)

            if fig_summary is not None:
                st.markdown("**SHAP Summary (Beeswarm)**")
                st.pyplot(fig_summary, clear_figure=True)

            st.markdown("**Top Features Interpretation**")
            interpretation_lines = []
            for feat, val in top_feats[:5]:
                interpretation_lines.append(f"- **{feat}** (mean |SHAP| = {val:.4f})")
            st.markdown("\n".join(interpretation_lines))
            st.caption(f"Model: {model_name} | Sample size: 200 decks")

    # ── Archetype usage bar chart ──
    st.divider()
    st.markdown("<div class='section-label'>Archetype Usage Distribution</div>", unsafe_allow_html=True)

    usage_fig = px.bar(
        arch_stats,
        x="archetype",
        y="matches",
        color="win_rate",
        color_continuous_scale="RdYlGn",
        range_color=[40, 60],
        labels={"archetype": "Archetype", "matches": "Total Matches", "win_rate": "Win Rate %"},
        text=arch_stats["matches"].apply(lambda x: f"{x:,}"),
    )
    usage_fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#1a3a6e",
        height=400,
        coloraxis_colorbar=dict(title="Win %", ticksuffix="%"),
    )
    usage_fig.update_traces(textposition="outside")
    st.plotly_chart(usage_fig, use_container_width=True)


if __name__ == "__main__":
    main()
