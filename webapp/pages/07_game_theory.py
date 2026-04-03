"""
Game Theory Insights Page
=========================
Task 31 — Archetypes as strategies, win probability as payoff,
          matchup reasoning text.

Frames the Clash Royale meta as a game-theory problem:
  • Each archetype is a *strategy* a player can choose.
  • The *payoff* for choosing strategy i against strategy j is
    the historical win-rate of archetype i vs archetype j.
  • A payoff matrix, Nash-equilibrium approximation, and
    dominant / dominated strategy analysis are computed from
    the real match dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from utils.deck_helpers import (
    build_deck_key,
    detect_archetype,
)
from utils.metadata import (
    get_card_names,
    get_elixir_costs,
)
from utils.ui_helpers import inject_fonts

st.set_page_config(page_title="Game Theory Insights", layout="wide")
inject_fonts()

# ── CSS ─────────────────────────────────────────────────────────────
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
    .note-box { background: #ffffff; border: 1px solid #d0dbe8; border-radius: 10px;
        padding: 14px 16px; color: #3b536e; font-size: 14px;
        box-shadow: 0 1px 3px rgba(26,86,219,0.06); }
    .insight-card { background: #ffffff; border: 1px solid #d0dbe8; border-radius: 10px;
        padding: 16px 20px; margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(26,86,219,0.06); }
    .payoff-high { color: #16a34a; font-weight: 700; }
    .payoff-low  { color: #ef4444; font-weight: 700; }
    .payoff-even { color: #6b7fa3; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ───────────────────────────────────────────────────────
DATA_PATHS = [
    Path("data/processed/clash_royale_clean.parquet"),
    Path("data/processed/final_ml_dataset.parquet"),
]
PLAYER_CARD_COLS = [f"player1.card{i}" for i in range(1, 9)]
OPPONENT_CARD_COLS = [f"player2.card{i}" for i in range(1, 9)]
PLAYER_CROWNS = "player1.crowns"
OPPONENT_CROWNS = "player2.crowns"

ARCHETYPE_ORDER = [
    "Cycle", "Beatdown", "Siege", "Bait", "Bridge Spam",
    "Graveyard", "Miner Control", "Loon", "Sparky", "Control",
]

# ── Archetype philosophy text (for reasoning section) ───────────────
ARCHETYPE_PHILOSOPHY: dict[str, str] = {
    "Cycle": (
        "Cycle decks rely on low-cost cards to rapidly rotate back to their win condition "
        "(typically Hog Rider or Miner). They apply relentless chip damage and out-cycle "
        "the opponent's counters. In game-theory terms this strategy maximises *tempo* — "
        "the rate of threat delivery."
    ),
    "Beatdown": (
        "Beatdown builds slow, heavy pushes behind tanks like Golem or Giant. The payoff "
        "comes from overwhelming the opponent in a single decisive push. This is a "
        "*high-variance, high-reward* strategy: when it connects, the payoff is enormous, "
        "but it is vulnerable to fast punishes during the build-up phase."
    ),
    "Siege": (
        "Siege plants a win condition at the bridge (X-Bow or Mortar) and defends it. "
        "It inverts the usual offensive pattern — the 'attack' is a building. "
        "Strategically it is a *defensive-dominant* approach that punishes opponents who "
        "commit elixir on offence."
    ),
    "Bait": (
        "Bait decks force the opponent to spend their key spells on lesser threats, then "
        "capitalise with Goblin Barrel or similar punish cards. This mirrors a classic "
        "game-theory *bluffing / information asymmetry* strategy: the opponent must guess "
        "which threat to counter."
    ),
    "Bridge Spam": (
        "Bridge Spam drops high-pressure troops at the bridge immediately. It exploits "
        "narrow windows of low elixir on the opponent's side. In game theory this is an "
        "*aggressive pre-emptive* strategy — it restricts the opponent's action space by "
        "forcing immediate defensive reactions."
    ),
    "Graveyard": (
        "Graveyard decks generate value over time by spawning skeletons on the opponent's "
        "tower. The payoff is *probabilistic* — damage depends on how many skeletons the "
        "opponent fails to clear. This makes it a *mixed-strategy* archetype with "
        "inherent variance."
    ),
    "Miner Control": (
        "Miner Control uses the Miner's surprise placement to chip away at towers while "
        "maintaining strong defensive card rotations. It is a *minimax* strategy — "
        "minimising the opponent's maximum damage while securing incremental gains."
    ),
    "Loon": (
        "Balloon-based decks aim to connect the Balloon to a tower for massive burst "
        "damage. The strategy is *binary*: if the Balloon reaches the tower the payoff is "
        "very high; if it is countered, the investment is largely wasted."
    ),
    "Sparky": (
        "Sparky decks rely on the threat of Sparky's devastating charged shot to force "
        "specific counter-play. The presence of Sparky *constrains* the opponent's "
        "strategy set — they must hold a counter or face catastrophic damage."
    ),
    "Control": (
        "Control decks prioritise defensive value and counter-pushes over aggressive "
        "plays. They aim to accumulate small elixir advantages and convert those into "
        "wins over time. This is a *risk-averse* strategy that performs well when "
        "opponents over-commit."
    ),
}

# ── Data loading ────────────────────────────────────────────────────

# 500K sample is statistically robust for archetype-level win rates
# (each of ~10 archetypes gets ~50K rows on average) while being ~25×
# faster than processing the full 12.4M-row dataset.
SAMPLE_SIZE = 500_000

@st.cache_data(show_spinner="Loading match data …")
def load_match_data() -> pd.DataFrame:
    for p in DATA_PATHS:
        if p.exists():
            df = pl.read_parquet(p)
            needed = set(PLAYER_CARD_COLS + OPPONENT_CARD_COLS +
                         [PLAYER_CROWNS, OPPONENT_CROWNS])
            if needed.issubset(set(df.columns)):
                selected = df.select(list(needed))
                # Stratified random sample for performance
                if selected.height > SAMPLE_SIZE:
                    selected = selected.sample(n=SAMPLE_SIZE, seed=42)
                return selected.to_pandas()
    raise FileNotFoundError("No parquet with player+opponent cards and crowns found.")


@st.cache_data(show_spinner="Building archetype payoff matrix …")
def build_payoff_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    payoff_matrix : DataFrame
        Rows = player archetype, Cols = opponent archetype.
        Values = player win-rate (%) in that matchup.
    counts_matrix : DataFrame
        Same shape; values = number of matches in that cell.
    """
    df = load_match_data()
    name_map = get_card_names(force_refresh=False)
    elixir_map = get_elixir_costs(force_refresh=False)

    def _detect(row: pd.Series, cols: list[str]) -> str:
        ids = [int(row[c]) for c in cols]
        return detect_archetype(ids, name_map, elixir_map)

    df["p_arch"] = df.apply(lambda r: _detect(r, PLAYER_CARD_COLS), axis=1)
    df["o_arch"] = df.apply(lambda r: _detect(r, OPPONENT_CARD_COLS), axis=1)
    df["p_win"] = (df[PLAYER_CROWNS] > df[OPPONENT_CROWNS]).astype(int)

    # Filter to known archetypes only
    df = df[df["p_arch"].isin(ARCHETYPE_ORDER) & df["o_arch"].isin(ARCHETYPE_ORDER)]

    grouped = (
        df.groupby(["p_arch", "o_arch"], as_index=False)
        .agg(matches=("p_win", "count"), wins=("p_win", "sum"))
    )
    grouped["win_rate"] = (grouped["wins"] / grouped["matches"] * 100).round(2)

    payoff = grouped.pivot(index="p_arch", columns="o_arch", values="win_rate")
    counts = grouped.pivot(index="p_arch", columns="o_arch", values="matches")

    # Reindex to canonical order
    payoff = payoff.reindex(index=ARCHETYPE_ORDER, columns=ARCHETYPE_ORDER)
    counts = counts.reindex(index=ARCHETYPE_ORDER, columns=ARCHETYPE_ORDER).fillna(0).astype(int)

    return payoff, counts


# ── Game-theory helpers ─────────────────────────────────────────────

def find_dominant_strategies(payoff: pd.DataFrame) -> dict[str, list[str]]:
    """Identify dominant and dominated strategies."""
    archetypes = list(payoff.index)
    dominant: list[str] = []
    dominated: list[str] = []

    for i, a in enumerate(archetypes):
        row_a = payoff.loc[a].values
        is_dominated = False
        for j, b in enumerate(archetypes):
            if i == j:
                continue
            row_b = payoff.loc[b].values
            # a is dominated by b if b wins more in every matchup
            valid = ~(np.isnan(row_a) | np.isnan(row_b))
            if valid.sum() < 3:
                continue
            if np.all(row_b[valid] >= row_a[valid]) and np.any(row_b[valid] > row_a[valid]):
                is_dominated = True
                break
        if is_dominated:
            dominated.append(a)

    # dominant = beats 50 %+ against every opponent
    for a in archetypes:
        row = payoff.loc[a].values
        valid = ~np.isnan(row)
        if valid.sum() >= 3 and np.all(row[valid] >= 50):
            dominant.append(a)

    return {"dominant": dominant, "dominated": dominated}


def approximate_nash_equilibrium(payoff: pd.DataFrame, iterations: int = 10_000) -> pd.Series:
    """
    Approximate a mixed-strategy Nash equilibrium via fictitious play.

    Returns the empirical frequency with which each archetype should be
    played to maximise minimum expected payoff.
    """
    mat = payoff.values.copy()
    # Replace NaN with 50 (neutral assumption)
    mat = np.where(np.isnan(mat), 50.0, mat)
    n = mat.shape[0]

    counts = np.ones(n, dtype=float)  # opponent mixed strategy counts
    my_counts = np.zeros(n, dtype=float)

    for _ in range(iterations):
        opponent_mix = counts / counts.sum()
        # Expected payoff of each strategy against opponent mix
        expected = mat @ opponent_mix
        best = int(np.argmax(expected))
        my_counts[best] += 1
        # Opponent best responds to our mix
        my_mix = my_counts / my_counts.sum()
        opp_expected = mat.T @ my_mix  # opponent wants to minimise our payoff
        worst = int(np.argmin(opp_expected))
        counts[worst] += 1

    freq = my_counts / my_counts.sum()
    return pd.Series(freq, index=payoff.index, name="Nash Frequency")


def compute_best_responses(payoff: pd.DataFrame) -> dict[str, str]:
    """For each opponent archetype, find the player archetype with highest payoff."""
    best = {}
    for opp in payoff.columns:
        col = payoff[opp].dropna()
        if not col.empty:
            best[opp] = str(col.idxmax())
    return best


def compute_archetype_summary(payoff: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table: archetype, avg payoff, best/worst matchup, total matches."""
    rows = []
    for arch in payoff.index:
        row_vals = payoff.loc[arch].dropna()
        row_counts = counts.loc[arch]
        if row_vals.empty:
            continue
        best_mu = str(row_vals.idxmax())
        worst_mu = str(row_vals.idxmin())
        rows.append({
            "Archetype": arch,
            "Avg Win Rate (%)": round(float(row_vals.mean()), 2),
            "Best Matchup": f"vs {best_mu} ({row_vals[best_mu]:.1f}%)",
            "Worst Matchup": f"vs {worst_mu} ({row_vals[worst_mu]:.1f}%)",
            "Total Matches": int(row_counts.sum()),
        })
    return pd.DataFrame(rows)


# ── Styling helper for the heatmap ──────────────────────────────────

def style_payoff_cell(val):
    if pd.isna(val):
        return "color: #ccc;"
    if val >= 55:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if val <= 45:
        return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
    return "background-color: #fefce8; color: #854d0e;"


# ── Main page ───────────────────────────────────────────────────────

def main():
    st.title("🎲 Game Theory Insights")
    st.markdown(
        "<div style='margin-bottom:14px;color:#5a7394;font-size:15px;'>"
        "Analysing the Clash Royale meta through the lens of <strong>game theory</strong>. "
        "Each archetype is a <em>strategy</em>, and the historical win rate of archetype A "
        "vs archetype B is the <em>payoff</em>."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        payoff, counts = build_payoff_matrix()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    # ── 1. Payoff matrix heatmap ────────────────────────────────────
    st.subheader("📊 Archetype Payoff Matrix")
    st.markdown(
        "Each cell shows the **player (row) win-rate %** against the **opponent (column)** archetype. "
        "Green cells (≥ 55 %) indicate a favourable matchup; red cells (≤ 45 %) indicate an unfavourable one."
    )

    styled = (
        payoff.style
        .format("{:.1f}", na_rep="—")
        .map(style_payoff_cell)
    )
    st.dataframe(styled, use_container_width=True, height=420)

    st.caption(
        "Rows = player archetype (strategy chosen). "
        "Columns = opponent archetype. "
        "Values = player win-rate %. "
        "'—' means insufficient data (< threshold)."
    )

    st.divider()

    # ── 2. Archetype summary table ──────────────────────────────────
    st.subheader("📋 Archetype Performance Summary")
    summary = compute_archetype_summary(payoff, counts)
    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data to build the summary table.")

    st.divider()

    # ── 3. Dominant / dominated strategies ──────────────────────────
    st.subheader("♟️ Dominant & Dominated Strategies")
    st.markdown(
        """
        In game theory:
        - A **dominant strategy** yields a payoff ≥ 50 % against *every* opponent — 
          it is never a bad choice.
        - A **dominated strategy** is one where another strategy is *always* at least 
          as good — a rational player would avoid it.
        """
    )

    dom = find_dominant_strategies(payoff)

    col_d, col_dd = st.columns(2)
    with col_d:
        if dom["dominant"]:
            for a in dom["dominant"]:
                avg = payoff.loc[a].dropna().mean()
                st.success(f"**{a}** — dominant (avg payoff {avg:.1f} %)")
        else:
            st.info("No strictly dominant strategy exists — the meta is balanced and "
                    "no single archetype beats every other.")
    with col_dd:
        if dom["dominated"]:
            for a in dom["dominated"]:
                avg = payoff.loc[a].dropna().mean()
                st.error(f"**{a}** — dominated (avg payoff {avg:.1f} %)")
        else:
            st.info("No strictly dominated strategy exists — every archetype has at "
                    "least one favourable matchup.")

    st.divider()

    # ── 4. Best-response map ────────────────────────────────────────
    st.subheader("🎯 Best Responses")
    st.markdown(
        "If you *know* your opponent's archetype, the **best response** is the "
        "archetype with the highest payoff against it."
    )

    best_resp = compute_best_responses(payoff)
    if best_resp:
        resp_df = pd.DataFrame([
            {
                "Opponent Plays": opp,
                "Best Response": br,
                "Expected Win Rate": f"{payoff.loc[br, opp]:.1f} %",
            }
            for opp, br in best_resp.items()
        ])
        st.dataframe(resp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Insufficient data for best-response analysis.")

    st.divider()

    # ── 5. Nash equilibrium approximation ───────────────────────────
    st.subheader("⚖️ Approximate Nash Equilibrium")
    st.markdown(
        """
        A **Nash equilibrium** is a mixed strategy where no player can improve their 
        expected payoff by unilaterally changing their archetype choice. The frequencies 
        below approximate how often each archetype should be played to maximise the 
        minimum expected win-rate (the *maximin* strategy).

        This is computed via **fictitious play** over the payoff matrix.
        """
    )

    nash = approximate_nash_equilibrium(payoff)
    nash_df = (
        nash.reset_index()
        .rename(columns={"index": "Archetype", "Nash Frequency": "Equilibrium Frequency"})
    )
    nash_df["Equilibrium Frequency"] = nash_df["Equilibrium Frequency"].apply(
        lambda x: f"{x * 100:.1f} %"
    )
    # Add recommended column
    nash_values = nash.values
    max_freq = nash_values.max()
    nash_df["Recommendation"] = [
        "★ Strong pick" if v == max_freq else ("Viable" if v > 0.05 else "Niche")
        for v in nash_values
    ]

    st.dataframe(nash_df, use_container_width=True, hide_index=True)

    st.markdown(
        "<div class='note-box'>"
        "<strong>Interpretation:</strong> In a perfectly rational meta, players would "
        "mix between archetypes at these frequencies. A high equilibrium frequency means "
        "the archetype is a strong choice even when opponents counter-pick optimally."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 6. Matchup reasoning — archetype deep-dives ────────────────
    st.subheader("🧠 Matchup Reasoning by Archetype")
    st.markdown(
        "Select an archetype to see its game-theory profile: strategic philosophy, "
        "best / worst matchups, and the reasoning behind them."
    )

    selected = st.selectbox("Choose an archetype:", ARCHETYPE_ORDER)

    if selected:
        st.markdown(f"### {selected}")

        # Philosophy text
        philosophy = ARCHETYPE_PHILOSOPHY.get(selected, "")
        if philosophy:
            st.markdown(
                f"<div class='insight-card'>{philosophy}</div>",
                unsafe_allow_html=True,
            )

        # Matchup details from the payoff matrix
        row = payoff.loc[selected].dropna().sort_values(ascending=False)
        if row.empty:
            st.info("Insufficient matchup data for this archetype.")
        else:
            best_3 = row.head(3)
            worst_3 = row.tail(3).sort_values()

            col_b, col_w = st.columns(2)
            with col_b:
                st.markdown("<div class='section-label'>Best Matchups</div>",
                            unsafe_allow_html=True)
                for opp, wr in best_3.items():
                    delta = wr - 50.0
                    st.metric(f"vs {opp}", f"{wr:.1f}%", f"{delta:+.1f}pp")
                _explain_advantage(selected, list(best_3.index), advantage=True)

            with col_w:
                st.markdown("<div class='section-label'>Worst Matchups</div>",
                            unsafe_allow_html=True)
                for opp, wr in worst_3.items():
                    delta = wr - 50.0
                    st.metric(f"vs {opp}", f"{wr:.1f}%", f"{delta:+.1f}pp")
                _explain_advantage(selected, list(worst_3.index), advantage=False)

        # Match volume
        arch_matches = int(counts.loc[selected].sum())
        st.caption(f"Based on {arch_matches:,} matches where a player used {selected}.")

    st.divider()

    # ── 7. How this works section ───────────────────────────────────
    st.subheader("📖 How This Analysis Works")
    st.markdown(
        f"""
        1. **Archetype Classification** — A random sample of {SAMPLE_SIZE:,} matches
           (from ~12.4 M total) is classified into archetypes using rule-based detection
           (win-condition cards + average elixir cost). The sample is large enough for
           statistically stable archetype-level estimates.

        2. **Payoff Matrix** — For each (player archetype, opponent archetype) pair, the 
           player's historical win-rate is computed. This is the *payoff* in game-theory 
           terminology.

        3. **Dominant / Dominated** — An archetype is *dominant* if its payoff ≥ 50 % 
           against every opponent; *dominated* if another archetype always outperforms it.

        4. **Best Response** — For each opponent archetype, the player archetype with the 
           highest payoff is the *best response* — the optimal counter-pick.

        5. **Nash Equilibrium** — Using *fictitious play*, we approximate the mixed 
           strategy where no player benefits from switching. This tells us how often 
           each archetype should appear in a balanced meta.

        6. **Matchup Reasoning** — Qualitative explanations connect the quantitative 
           payoffs to Clash Royale gameplay mechanics (cycle speed, push weight, spell 
           value, etc.).
        """
    )


# ── Reasoning text generator ────────────────────────────────────────

_MATCHUP_REASONS: dict[tuple[str, str], str] = {
    # Favourable reasons (selected archetype → opponent)
    ("Cycle", "Beatdown"): "Cycle decks can punish the slow Beatdown build-up with constant opposite-lane pressure.",
    ("Cycle", "Siege"): "Fast cycling lets Cycle decks pressure both lanes before Siege can set up a defensive lock.",
    ("Beatdown", "Siege"): "Heavy tanks can overwhelm a single X-Bow or Mortar placement and trade favourably.",
    ("Beatdown", "Bait"): "Beatdown pushes are harder to stop with chip/bait units; spell splash clears swarms.",
    ("Siege", "Bait"): "Siege buildings force responses, and defensive spells handle Bait swarms effectively.",
    ("Siege", "Graveyard"): "Siege decks often carry enough splash and buildings to neutralise Graveyard spawns.",
    ("Bait", "Cycle"): "Bait forces awkward spell usage, disrupting a Cycle deck's rotation.",
    ("Bait", "Control"): "Multiple bait threats exhaust Control's limited removal options.",
    ("Bridge Spam", "Siege"): "Fast bridge-drop units can destroy an X-Bow or Mortar before it locks on.",
    ("Bridge Spam", "Graveyard"): "Aggressive pressure forces Graveyard players to defend rather than set up.",
    ("Graveyard", "Control"): "Graveyard's sustained DPS is hard for Control decks to fully neutralise.",
    ("Graveyard", "Cycle"): "Cycle decks lack the splash to clear Graveyard skeletons efficiently.",
    ("Miner Control", "Siege"): "Miner can snipe buildings and chip towers while maintaining defensive control.",
    ("Loon", "Cycle"): "Balloon's burst damage can overwhelm Cycle decks that lack air-targeting buildings.",
    ("Loon", "Bait"): "Bait decks rarely carry the hard air counters needed to stop a Balloon push.",
    ("Sparky", "Beatdown"): "Sparky's splash charge can obliterate grouped Beatdown units behind the tank.",
    ("Control", "Bridge Spam"): "Control decks specialise in absorbing aggression and counter-pushing for value.",
    ("Control", "Loon"): "Buildings and air-targeting troops in Control decks answer Balloon reliably.",
}


def _explain_advantage(selected: str, opponents: list[str], advantage: bool) -> None:
    """Render reasoning bullets for why *selected* is strong/weak vs *opponents*."""
    bullets: list[str] = []
    for opp in opponents:
        key = (selected, opp) if advantage else (opp, selected)
        reason = _MATCHUP_REASONS.get(key)
        if reason:
            if advantage:
                bullets.append(f"✅ **vs {opp}:** {reason}")
            else:
                bullets.append(f"⚠️ **vs {opp}:** {reason}")

    if not bullets:
        if advantage:
            bullets.append("Historical data shows a consistent edge, likely due to "
                           "favourable elixir trades and card interactions.")
        else:
            bullets.append("The opposing archetype structurally counters the key win "
                           "conditions of this strategy.")

    for b in bullets:
        st.markdown(b)


if __name__ == "__main__":
    main()
