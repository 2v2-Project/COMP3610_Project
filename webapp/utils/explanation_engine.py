"""
Task 32: Explainable AI Prediction Engine

Combines SHAP-based model explanations with rule-based domain knowledge.
Produces 2-4 explanation bullets per prediction that read like a Clash Royale analyst,
not a technical model output.

Features:
  - Local SHAP feature contributions with human-readable explanations
  - Rule-based signals (archetype, elixir, synergies, matchup)
  - Matchup interaction detection (counters, weaknesses)
  - Human-readable strategy explanations
  - Seamless single-deck and matchup mode support

Used by both Win Predictor and Matchup Analysis pages.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict

import pandas as pd

try:
    from .deck_helpers import (
        compute_avg_elixir,
        compute_cycle_cost,
        count_card_types,
        detect_archetype,
    )
    from .shap_utils import get_local_shap_explanation
    
except ImportError:
    from utils.deck_helpers import (
        compute_avg_elixir,
        compute_cycle_cost,
        count_card_types,
        detect_archetype,
    )
    from utils.shap_utils import get_local_shap_explanation

logger = logging.getLogger(__name__)


def _build_maps(metadata_df: pd.DataFrame) -> tuple[dict[int, str], dict[int, int], dict[int, str]]:
    """Build name, elixir, and type maps from card metadata."""
    required = {"card_id", "name"}
    if not required.issubset(metadata_df.columns):
        raise ValueError(f"metadata_df must contain at least {required}")

    name_map = {
        int(row["card_id"]): str(row["name"])
        for _, row in metadata_df.iterrows()
        if pd.notna(row["card_id"]) and pd.notna(row["name"])
    }

    elixir_map: dict[int, int] = {}
    if "elixir" in metadata_df.columns:
        for _, row in metadata_df.iterrows():
            try:
                if pd.notna(row["elixir"]):
                    elixir_map[int(row["card_id"])] = int(row["elixir"])
            except Exception:
                continue

    type_map: dict[int, str] = {}
    if "type" in metadata_df.columns:
        for _, row in metadata_df.iterrows():
            card_type = str(row["type"]).strip().lower() if pd.notna(row["type"]) else ""
            if card_type in {"troop", "spell", "building"}:
                type_map[int(row["card_id"])] = card_type

    return name_map, elixir_map, type_map


def _has_cards(card_ids: List[int], name_map: dict[int, str], *keywords: str) -> bool:
    """Check if any card in deck matches any of the keywords (exact name match)."""
    names = [name_map.get(cid, "").lower() for cid in card_ids]
    return any(keyword.lower() == card_name for keyword in keywords for card_name in names)


def _detect_synergy_bullets(card_ids: List[int], name_map: dict[int, str]) -> List[str]:
    """
    Detect meaningful synergies in deck.

    Covers realistic card interactions across categories:
    - Win condition + support
    - Tank + support
    - Bait + bait support
    - Spell synergy
    - Control + cycle
    """
    bullets: List[str] = []

    # Win condition + support
    if _has_cards(card_ids, name_map, "hog rider") and _has_cards(card_ids, name_map, "earthquake"):
        bullets.append("Hog Rider + Earthquake creates strong pressure and helps break through defensive buildings.")

    if _has_cards(card_ids, name_map, "hog rider") and _has_cards(card_ids, name_map, "ice spirit"):
        bullets.append("Hog Rider + Ice Spirit supports fast-cycle pressure and efficient damage trades.")

    if _has_cards(card_ids, name_map, "hog rider") and _has_cards(card_ids, name_map, "skeletons"):
        bullets.append("Hog Rider + Skeletons creates cheap offensive rotations that keep pressure constant.")

    if _has_cards(card_ids, name_map, "balloon") and _has_cards(card_ids, name_map, "freeze"):
        bullets.append("Balloon + Freeze gives this deck a dangerous finishing combo in high-pressure moments.")

    if _has_cards(card_ids, name_map, "balloon") and _has_cards(card_ids, name_map, "lava hound"):
        bullets.append("Lava Hound + Balloon stacks aerial pressure and forces heavy defensive commitments.")

    if _has_cards(card_ids, name_map, "miner") and _has_cards(card_ids, name_map, "wall breakers"):
        bullets.append("Miner + Wall Breakers creates split-lane pressure and consistent chip damage.")

    if _has_cards(card_ids, name_map, "miner") and _has_cards(card_ids, name_map, "poison"):
        bullets.append("Miner + Poison supports a reliable chip strategy with strong spell-backed pressure.")

    if _has_cards(card_ids, name_map, "graveyard") and _has_cards(card_ids, name_map, "poison"):
        bullets.append("Graveyard + Poison creates a classic control win condition with strong area denial.")

    if _has_cards(card_ids, name_map, "graveyard") and _has_cards(card_ids, name_map, "mirror"):
        bullets.append("Graveyard + Mirror doubles down on a spell-supported win condition and can overwhelm defense.")

    # Tank + support
    if _has_cards(card_ids, name_map, "golem") and _has_cards(card_ids, name_map, "night witch"):
        bullets.append("Golem + Night Witch is a classic beatdown core that builds threatening late pushes.")

    if _has_cards(card_ids, name_map, "golem") and _has_cards(card_ids, name_map, "baby dragon"):
        bullets.append("Golem + Baby Dragon creates durable pushes with splash support behind the tank.")

    if _has_cards(card_ids, name_map, "giant") and _has_cards(card_ids, name_map, "inferno dragon"):
        bullets.append("Giant + Inferno Dragon combines tank pressure with strong single-target support.")

    if _has_cards(card_ids, name_map, "pekka") and _has_cards(card_ids, name_map, "battle ram"):
        bullets.append("PEKKA + Battle Ram points to bridge-spam pressure backed by strong tank-killing defense.")

    if _has_cards(card_ids, name_map, "royal giant") and _has_cards(card_ids, name_map, "fisherman"):
        bullets.append("Royal Giant + Fisherman supports ranged pressure while improving defensive control.")

    # Bait
    if _has_cards(card_ids, name_map, "goblin barrel") and _has_cards(card_ids, name_map, "princess"):
        bullets.append("Goblin Barrel + Princess is a classic bait shell that pressures opponents to overspend spells.")

    if _has_cards(card_ids, name_map, "goblin barrel") and _has_cards(card_ids, name_map, "goblins"):
        bullets.append("Goblin Barrel + Goblins reinforces small-unit bait pressure and chip damage.")

    if _has_cards(card_ids, name_map, "princess") and _has_cards(card_ids, name_map, "inferno tower"):
        bullets.append("Princess + Inferno Tower gives the deck a strong bait-and-defense backbone.")

    # Siege
    if _has_cards(card_ids, name_map, "x-bow") and _has_cards(card_ids, name_map, "tesla"):
        bullets.append("X-Bow + Tesla creates a reliable siege setup with defensive counterpush potential.")

    if _has_cards(card_ids, name_map, "mortar") and _has_cards(card_ids, name_map, "tornado"):
        bullets.append("Mortar + Tornado combines siege pressure with positional control.")

    # Spell synergy
    if _has_cards(card_ids, name_map, "fireball") and _has_cards(card_ids, name_map, "tornado"):
        bullets.append("Fireball + Tornado gives the deck strong spell control against grouped units.")

    if _has_cards(card_ids, name_map, "lightning") and _has_cards(card_ids, name_map, "pekka"):
        bullets.append("Lightning + PEKKA helps remove key defensive pieces before committing heavy pushes.")

    # Control / defense
    if _has_cards(card_ids, name_map, "inferno dragon") and _has_cards(card_ids, name_map, "tornado"):
        bullets.append("Inferno Dragon + Tornado creates strong reactive defense against high-health units.")

    if _has_cards(card_ids, name_map, "cannon") and _has_cards(card_ids, name_map, "archers"):
        bullets.append("Cannon + Archers forms a low-cost defensive cycle core.")

    if _has_cards(card_ids, name_map, "inferno tower") and _has_cards(card_ids, name_map, "hog rider"):
        bullets.append("Inferno Tower + Hog Rider balances sturdy defense with fast offensive pressure.")

    if _has_cards(card_ids, name_map, "tesla") and _has_cards(card_ids, name_map, "miner"):
        bullets.append("Tesla + Miner pairs defensive coverage with consistent chip damage.")

    # Electro Giant combos
    if _has_cards(card_ids, name_map, "electro giant") and _has_cards(card_ids, name_map, "tornado"):
        bullets.append("Electro Giant + Tornado pulls units into the Giant's reflect damage for devastating defensive value.")

    if _has_cards(card_ids, name_map, "three musketeers") and _has_cards(card_ids, name_map, "elixir collector"):
        bullets.append("Three Musketeers + Elixir Collector enables expensive split-lane pushes once elixir advantage builds up.")

    if _has_cards(card_ids, name_map, "pekka") and _has_cards(card_ids, name_map, "electro wizard"):
        bullets.append("PEKKA + Electro Wizard creates a counterpush that stuns and melts through opposing tanks and support.")

    if _has_cards(card_ids, name_map, "mega knight") and _has_cards(card_ids, name_map, "inferno dragon"):
        bullets.append("Mega Knight + Inferno Dragon forms a strong defensive-to-counterpush that punishes overcommitment.")

    if _has_cards(card_ids, name_map, "royal hogs") and _has_cards(card_ids, name_map, "earthquake"):
        bullets.append("Royal Hogs + Earthquake clears defensive buildings and punishes grouped defense for consistent split-lane damage.")

    if _has_cards(card_ids, name_map, "prince") and _has_cards(card_ids, name_map, "dark prince"):
        bullets.append("Double Prince creates dual-lane charge pressure that's very hard to defend on both sides efficiently.")

    if _has_cards(card_ids, name_map, "sparky") and _has_cards(card_ids, name_map, "goblin giant"):
        bullets.append("Goblin Giant + Sparky tanks for the Sparky while Spear Goblins add chip — a classic push combo.")

    if _has_cards(card_ids, name_map, "miner") and _has_cards(card_ids, name_map, "bats"):
        bullets.append("Miner + Bats is an efficient chip combination that forces spell responses or deals punishing tower damage.")

    if _has_cards(card_ids, name_map, "ram rider") and _has_cards(card_ids, name_map, "bandit"):
        bullets.append("Ram Rider + Bandit punish opponents at the bridge with fast, aggressive pressure and snare control.")

    if _has_cards(card_ids, name_map, "goblin barrel") and _has_cards(card_ids, name_map, "rocket"):
        bullets.append("Goblin Barrel + Rocket combines bait pressure with a heavy spell finisher for closing out tight matches.")

    if _has_cards(card_ids, name_map, "lava hound") and _has_cards(card_ids, name_map, "skeleton dragons"):
        bullets.append("Lava Hound + Skeleton Dragons stacks aerial splash behind the tank, cleaning up ground swarms on defense.")

    if _has_cards(card_ids, name_map, "elite barbarians") and _has_cards(card_ids, name_map, "rage"):
        bullets.append("Elite Barbarians + Rage creates sudden burst pressure that can catch opponents off guard at the bridge.")

    if _has_cards(card_ids, name_map, "giant") and _has_cards(card_ids, name_map, "sparky"):
        bullets.append("Giant + Sparky forces opponents to hold a reset card or risk massive damage from the charged shot.")

    return bullets


def _detect_matchup_interactions(
    player_cards: List[int],
    opponent_cards: List[int],
    name_map: dict[int, str],
) -> List[str]:
    """
    Detect matchup interactions: direct counters, weaknesses, and strategic advantages.

    Returns up to 2 bullets.
    """
    bullets: List[str] = []

    player_names = {name_map.get(cid, "").lower() for cid in player_cards if cid in name_map}
    opponent_names = {name_map.get(cid, "").lower() for cid in opponent_cards if cid in name_map}

    if "inferno tower" in opponent_names and any(card in player_names for card in ["golem", "giant", "electro giant", "lava hound"]):
        bullets.append("Opponent's Inferno Tower directly threatens your tank-based pushes and can force awkward commitments.")

    if "poison" in opponent_names and any(card in player_names for card in ["barbarians", "night witch", "goblin gang", "skeleton army", "minion horde"]):
        bullets.append("Opponent's Poison reduces the value of your swarm cards, which can weaken your pressure.")

    if "graveyard" in player_names and "poison" in opponent_names:
        bullets.append("Opponent's Poison makes it harder for your Graveyard to generate sustained damage.")

    player_has_air_threat = any(card in player_names for card in ["balloon", "lava hound", "baby dragon", "inferno dragon"])
    opponent_has_air_threat = any(card in opponent_names for card in ["balloon", "lava hound", "baby dragon", "inferno dragon"])

    player_has_air_defense = any(card in player_names for card in ["archers", "hunter", "tesla", "inferno tower", "inferno dragon", "baby dragon", "wizard"])
    opponent_has_air_defense = any(card in opponent_names for card in ["archers", "hunter", "tesla", "inferno tower", "inferno dragon", "baby dragon", "wizard"])

    if player_has_air_threat and not opponent_has_air_defense:
        bullets.append("Opponent appears light on air defense, so your air threat should be harder to contain.")

    if opponent_has_air_threat and not player_has_air_defense:
        bullets.append("You appear light on air defense, so opponent air threats may be difficult to stop cleanly.")

    player_has_small_units = any(card in player_names for card in ["goblin barrel", "goblins", "skeletons", "goblin gang", "princess", "dart goblin", "minions"])
    opponent_has_splash = any(card in opponent_names for card in ["bomber", "baby dragon", "valkyrie", "bowler", "wizard", "executioner", "tornado", "fireball"])

    if player_has_small_units and opponent_has_splash:
        bullets.append("Opponent has good splash control, which can make your lighter support units less reliable.")

    opponent_has_swarms = any(card in opponent_names for card in ["barbarians", "night witch", "goblin gang", "skeleton army", "minion horde", "goblins"])
    player_has_splash = any(card in player_names for card in ["bomber", "baby dragon", "valkyrie", "bowler", "wizard", "executioner", "fireball", "tornado", "rocket"])

    if opponent_has_swarms and player_has_splash:
        bullets.append("Your splash options line up well into the opponent's swarm-based support cards.")

    if _has_cards(player_cards, name_map, "hog rider") and _has_cards(opponent_cards, name_map, "hog rider"):
        bullets.append("Both decks rely on Hog Rider pressure, so support cards and cycle timing will likely decide the matchup.")

    if _has_cards(player_cards, name_map, "balloon") and _has_cards(opponent_cards, name_map, "balloon"):
        bullets.append("This Balloon mirror matchup will depend heavily on air defense and spell timing.")

    # Reset mechanics vs Inferno
    if any(card in opponent_names for card in ["inferno tower", "inferno dragon"]) and \
       any(card in player_names for card in ["zap", "electro wizard", "electro spirit", "electro dragon", "lightning"]):
        bullets.append("You have reset tools against the opponent's Inferno, which reduces its threat to your pushes.")

    # PEKKA vs opponent tanks
    if "pekka" in player_names and any(card in opponent_names for card in ["golem", "giant", "electro giant", "royal giant", "mega knight"]):
        bullets.append("Your PEKKA directly counters the opponent's tank, giving you a strong defensive anchor in this matchup.")

    if "pekka" in opponent_names and any(card in player_names for card in ["golem", "giant", "electro giant"]):
        bullets.append("Opponent's PEKKA can shut down your tank — you may need to split push or outplay with timing.")

    # Earthquake vs buildings
    if "earthquake" in player_names and any(card in opponent_names for card in ["inferno tower", "tesla", "cannon", "bomb tower", "x-bow", "mortar"]):
        bullets.append("Your Earthquake directly threatens the opponent's buildings, removing key defensive structures.")

    # Royal Giant vs Siege
    if "royal giant" in player_names and any(card in opponent_names for card in ["x-bow", "mortar"]):
        bullets.append("Royal Giant outranges siege buildings, giving you a natural advantage in this matchup.")

    if any(card in player_names for card in ["x-bow", "mortar"]) and "royal giant" in opponent_names:
        bullets.append("Royal Giant outranges your siege building — this is a traditionally difficult matchup you'll need to outplay.")

    # Log vs Goblin Barrel
    if any(card in player_names for card in ["the log", "barbarian barrel"]) and "goblin barrel" in opponent_names:
        bullets.append("Your log-type spell cleanly counters Goblin Barrel, which weakens their primary bait threat.")

    if "goblin barrel" in player_names and not any(card in opponent_names for card in ["the log", "barbarian barrel", "tornado"]):
        bullets.append("Opponent lacks a clean answer to Goblin Barrel, which gives your bait strategy extra value.")

    # Win condition vs no building
    player_wc = any(card in player_names for card in ["hog rider", "ram rider", "battle ram", "royal hogs"])
    opp_has_building = any(card in opponent_names for card in ["inferno tower", "tesla", "cannon", "bomb tower", "goblin cage", "tombstone"])
    if player_wc and not opp_has_building:
        bullets.append("Opponent has no building to pull your win condition, which should allow more direct tower connections.")

    # Rocket vs high-value targets
    if "rocket" in player_names and any(card in opponent_names for card in ["x-bow", "sparky", "elixir collector", "three musketeers"]):
        bullets.append("Your Rocket can eliminate key high-value targets in the opponent's deck for strong elixir trades.")

    return bullets[:3]


# ── Archetype matchup dynamics ──────────────────────────────────────

_ARCHETYPE_DYNAMICS: Dict[tuple[str, str], str] = {
    ("Beatdown", "Siege"): "Beatdown typically overwhelms Siege — heavy tanks can absorb siege damage while support troops break through.",
    ("Siege", "Beatdown"): "Siege often struggles against Beatdown since heavy tanks absorb building damage and support troops push through.",
    ("Cycle", "Beatdown"): "Cycle decks can exploit Beatdown's slow buildup by pressuring opposite lane and chipping before big pushes develop.",
    ("Beatdown", "Cycle"): "Beatdown needs to survive early chip pressure and build a decisive push in double elixir to overwhelm Cycle defense.",
    ("Bait", "Control"): "Bait tests Control's spell discipline — if they waste their small spell, Goblin Barrel gets full value.",
    ("Control", "Bait"): "Control needs perfect spell management against Bait — saving the right spell for Goblin Barrel is critical.",
    ("Bridge Spam", "Beatdown"): "Bridge Spam can punish Beatdown's heavy elixir investments by rushing the opposite lane before pushes build.",
    ("Beatdown", "Bridge Spam"): "Beatdown must defend Bridge Spam pressure patiently and find windows to build up in double elixir.",
    ("Cycle", "Control"): "Cycle outpaces Control through fast rotations, but Control's defensive answers can absorb repeated chip pressure.",
    ("Control", "Cycle"): "Control should focus on maximum spell value and avoid being outcycled by the opponent's faster rotation.",
    ("Loon", "Cycle"): "Balloon decks need to catch Cycle out of rotation — relying on spell support or freeze timing to connect.",
    ("Cycle", "Loon"): "Cycle decks can often defend Balloon with fast rotations and punish with opposite-lane counterpressure.",
    ("Graveyard", "Beatdown"): "Graveyard's chip damage races against Beatdown's massive pushes — it's a tempo-dependent matchup.",
    ("Bait", "Bridge Spam"): "Both decks apply constant pressure — elixir management and defensive timing usually decide the outcome.",
    ("Siege", "Cycle"): "Siege vs Cycle is a battle of patience — Cycle can outcycle Siege defenses, but Siege chips from a distance.",
    ("Beatdown", "Graveyard"): "Beatdown's heavy pushes can overwhelm the defensive tools Graveyard decks rely on to stay alive.",
    ("Bridge Spam", "Siege"): "Bridge Spam can punish Siege setups with fast pressure before the defensive structure locks in.",
    ("Siege", "Bridge Spam"): "Siege must defend bridge pressure efficiently and find windows to lock an X-Bow or Mortar at the bridge.",
    ("Loon", "Bait"): "Balloon decks often carry enough splash to handle Bait's swarms, while Bait may lack strong anti-air.",
    ("Bait", "Loon"): "Bait decks may struggle to defend Balloon consistently without dedicated air-targeting units.",
    ("Miner Control", "Beatdown"): "Miner Control chips away while defending Beatdown's big pushes — patience and defensive timing are key.",
    ("Beatdown", "Miner Control"): "Beatdown must build overwhelming pushes to break through Miner Control's defensive framework.",
    ("Sparky", "Cycle"): "Sparky decks can be outcycled, as fast rotation lets the opponent always have a counter ready for the charged shot.",
    ("Cycle", "Sparky"): "Cycle's fast rotation helps ensure you always have a reset or distraction ready for the opponent's Sparky.",
}


def _archetype_matchup_bullet(player_arch: str, opponent_arch: str) -> Optional[str]:
    """Return a strategic insight about the archetype matchup, if known."""
    if player_arch == "Unknown" or opponent_arch == "Unknown":
        return None
    return _ARCHETYPE_DYNAMICS.get((player_arch, opponent_arch))


def _detect_vulnerabilities(
    card_ids: List[int],
    name_map: dict[int, str],
    elixir_map: dict[int, int],
    type_map: dict[int, str],
) -> List[str]:
    """
    Detect structural weaknesses in a deck.

    Returns up to 2 vulnerability bullets.
    """
    bullets: List[str] = []
    names = {name_map.get(cid, "").lower() for cid in card_ids if cid in name_map}

    # No win condition
    win_conditions = {"hog rider", "miner", "balloon", "golem", "giant", "royal giant",
                      "electro giant", "lava hound", "x-bow", "mortar", "graveyard",
                      "goblin barrel", "wall breakers", "ram rider", "battle ram",
                      "royal hogs", "three musketeers", "sparky", "mega knight",
                      "pekka", "prince", "elite barbarians"}
    if not names & win_conditions:
        bullets.append("This deck may lack a clear win condition, which can make it difficult to consistently deal tower damage.")

    # No big spell
    big_spells = {"fireball", "poison", "rocket", "lightning"}
    if not names & big_spells:
        bullets.append("No heavy spell means limited ability to finish damaged towers or remove medium-health support troops.")

    # No air defense
    air_defense = {"archers", "musketeer", "hunter", "wizard", "executioner",
                   "baby dragon", "inferno dragon", "electro dragon", "mega minion",
                   "minions", "minion horde", "tesla", "inferno tower", "bats",
                   "skeleton dragons", "firecracker", "electro wizard", "dart goblin"}
    if not names & air_defense:
        bullets.append("This deck appears light on air defense, which leaves it vulnerable to Balloon, Lava Hound, and other air threats.")

    # No building
    buildings = {"inferno tower", "tesla", "cannon", "bomb tower", "goblin cage",
                 "tombstone", "furnace", "elixir collector", "barbarian hut"}
    has_building = bool(names & buildings)
    # Only flag if deck uses ground win conditions that benefit from buildings for pulling
    if not has_building and not names & {"x-bow", "mortar"}:
        ground_threats_exist = bool(names & {"hog rider", "ram rider", "battle ram", "royal hogs"})
        if not ground_threats_exist:
            # deck doesn't need to pull things, but no building still matters
            pass  # skip - not every deck needs a building

    # Very heavy with no pump
    avg = compute_avg_elixir(card_ids, elixir_map)
    if avg >= 4.5 and "elixir collector" not in names:
        bullets.append("Very high average elixir without Elixir Collector means slower starts and risk of being overwhelmed early.")

    return bullets[:2]


def _get_elixir_profile(avg_elixir: float) -> str:
    """Classify deck by elixir cost profile."""
    if avg_elixir <= 2.8:
        return "very fast cycle"
    if avg_elixir <= 3.2:
        return "fast cycle"
    if avg_elixir <= 3.8:
        return "balanced"
    if avg_elixir <= 4.2:
        return "heavy control"
    return "ultra-heavy beatdown"


def _get_composition_balance(types: Dict[str, int]) -> str:
    """Describe deck composition balance."""
    troops = types.get("troop_count", 0)
    spells = types.get("spell_count", 0)
    buildings = types.get("building_count", 0)

    if troops >= 5:
        return "troop-heavy, which usually supports constant board presence"
    if spells >= 4:
        return "spell-heavy, which usually improves control and cleanup"
    if buildings >= 3:
        return "defensive, which usually improves structure-based control"
    return "balanced, with a mix of offense and defense"


def _rule_based_bullets(
    player_cards: List[int],
    metadata_df: pd.DataFrame,
    opponent_cards: Optional[List[int]] = None,
) -> List[str]:
    """
    Build human-readable deck explanations from game features.
    """
    name_map, elixir_map, type_map = _build_maps(metadata_df)

    player_avg = compute_avg_elixir(player_cards, elixir_map)
    player_cycle = compute_cycle_cost(player_cards, elixir_map)
    player_types = count_card_types(player_cards, type_map)
    player_archetype = detect_archetype(player_cards, name_map, elixir_map)

    bullets: List[str] = []

    if opponent_cards is None:
        elixir_desc = _get_elixir_profile(player_avg)
        comp_desc = _get_composition_balance(player_types)

        if player_archetype != "Unknown":
            bullets.append(
                f"This {player_archetype} deck plays with a {elixir_desc} profile, which shapes how it applies pressure."
            )
        else:
            bullets.append(
                f"This deck has a {elixir_desc} profile and a {comp_desc} structure."
            )

        if player_cycle > 0:
            if player_cycle <= 6:
                bullets.append(
                    f"Low cycle cost ({player_cycle}) means you can return to key cards quickly and keep pressure consistent."
                )
            elif player_cycle <= 10:
                bullets.append(
                    f"Moderate cycle cost ({player_cycle}) gives a balance between pressure and defensive flexibility."
                )
            else:
                bullets.append(
                    f"High cycle cost ({player_cycle}) means stronger individual cards, but slower rotations."
                )

        synergies = _detect_synergy_bullets(player_cards, name_map)
        if synergies:
            bullets.append(synergies[0])

        # Vulnerability detection for single-deck analysis
        vulnerabilities = _detect_vulnerabilities(player_cards, name_map, elixir_map, type_map)
        if vulnerabilities:
            bullets.append(vulnerabilities[0])

    else:
        opp_avg = compute_avg_elixir(opponent_cards, elixir_map)
        opp_cycle = compute_cycle_cost(opponent_cards, elixir_map)
        opp_archetype = detect_archetype(opponent_cards, name_map, elixir_map)

        elixir_diff = round(player_avg - opp_avg, 2)

        if player_archetype != "Unknown" and opp_archetype != "Unknown":
            arch_insight = _archetype_matchup_bullet(player_archetype, opp_archetype)
            if arch_insight:
                bullets.append(arch_insight)
            else:
                bullets.append(
                    f"{player_archetype} versus {opp_archetype} changes the matchup because these strategies pressure each other differently."
                )

        if abs(elixir_diff) >= 0.3:
            if elixir_diff < 0:
                bullets.append(
                    f"Your deck is lighter on average, so it should cycle faster and create more pressure windows."
                )
            else:
                bullets.append(
                    f"Your deck is heavier on average, so it leans more on stronger pushes than fast rotation."
                )

        cycle_diff = player_cycle - opp_cycle
        if abs(cycle_diff) >= 2:
            if cycle_diff < 0:
                bullets.append(
                    f"Your 4-card cycle is faster, so you should reach key defensive or offensive cards sooner."
                )
            else:
                bullets.append(
                    f"Opponent cycles faster, which can make repeated defensive answers easier for them."
                )

        synergies = _detect_synergy_bullets(player_cards, name_map)
        if synergies:
            bullets.append(synergies[0])

    return bullets


def _humanize_shap_feature(feature_name: str, metadata_df: pd.DataFrame) -> tuple[str, str]:
    """
    Convert technical SHAP feature names to human-readable strategy explanations.
    Returns (feature_description, gameplay_context)
    """
    name_map, _, _ = _build_maps(metadata_df)

    if feature_name.startswith("player_card_"):
        try:
            card_id = int(feature_name.split("_")[-1])
            card_name = name_map.get(card_id, f"card {card_id}")
            card_lower = card_name.lower()

            if any(x in card_lower for x in ["hog", "miner", "wall breakers", "balloon", "graveyard", "ram rider", "royal hogs"]):
                context = "it serves as the deck's primary win condition for dealing tower damage"
            elif any(x in card_lower for x in ["golem", "giant", "lava hound", "electro giant"]):
                context = "it acts as the deck's tank, absorbing damage and enabling support troops to deal damage"
            elif any(x in card_lower for x in ["inferno", "tesla", "cannon", "bomb tower"]):
                context = "it anchors the deck's defense against enemy pushes"
            elif any(x in card_lower for x in ["poison", "fireball", "tornado", "rocket", "lightning"]):
                context = "it provides crucial spell control and area denial"
            elif any(x in card_lower for x in ["zap", "the log", "snowball", "barbarian barrel"]):
                context = "it's a key small spell for resetting, finishing, or countering swarms"
            elif any(x in card_lower for x in ["skeletons", "ice spirit", "electro spirit", "bats"]):
                context = "it's a cheap cycle card that keeps rotation fast and efficient"
            elif any(x in card_lower for x in ["mega knight", "pekka", "sparky"]):
                context = "it creates high-threat counterpushes that demand an immediate response"
            elif any(x in card_lower for x in ["musketeer", "wizard", "executioner", "hunter"]):
                context = "it provides ranged support and helps control air and ground threats"
            elif any(x in card_lower for x in ["valkyrie", "knight", "ice golem"]):
                context = "it acts as a mini-tank that absorbs damage and supports offensive pushes"
            elif any(x in card_lower for x in ["princess", "dart goblin", "firecracker"]):
                context = "it applies chip pressure at range and baits out spell responses"
            else:
                context = "it contributes meaningfully to the deck's overall strategy"

            return (card_name, context)
        except Exception:
            return (feature_name, "it matters in the model's evaluation")

    if feature_name.startswith("opp_card_"):
        try:
            card_id = int(feature_name.split("_")[-1])
            card_name = name_map.get(card_id, f"card {card_id}")
            return (f"Opponent having {card_name}", "it changes how your deck can attack or defend")
        except Exception:
            return (feature_name, "it changes the matchup context")

    if "avg_elixir" in feature_name:
        if "player_" in feature_name:
            return ("Your average elixir profile", "it changes the deck's pace and rotation speed")
        return ("Opponent's average elixir profile", "it changes the pace of the matchup")

    if "cycle" in feature_name.lower():
        if "player_" in feature_name:
            return ("Your cycle speed", "it affects how quickly you reach key cards")
        return ("Opponent's cycle speed", "it affects how quickly they can respond")

    if "troop_count" in feature_name:
        if "player_" in feature_name:
            return ("Your troop count", "it changes board presence and pressure consistency")
        return ("Opponent's troop count", "it changes their pressure consistency")

    if "spell_count" in feature_name:
        if "player_" in feature_name:
            return ("Your spell count", "it changes your control and cleanup options")
        return ("Opponent's spell count", "it changes their control options")

    if "building_count" in feature_name:
        if "player_" in feature_name:
            return ("Your building count", "it changes your defensive structure")
        return ("Opponent's building count", "it changes their defensive structure")

    pretty = feature_name.replace("player_", "your ").replace("opp_", "opponent ")
    pretty = pretty.replace("_", " ").strip()
    return (pretty, "it influences the model's evaluation")


def _shap_bullets(
    model,
    feature_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    max_bullets: int = 2,
) -> List[str]:
    """
    Convert local SHAP values into human-readable strategy explanations.
    """
    bullets: List[str] = []

    try:
        top_features = get_local_shap_explanation(model, feature_df, top_n=max_bullets)

        if not top_features:
            logger.debug("SHAP returned no top features")
            return []

        for item in top_features:
            feature_name = str(item["feature"])
            direction = str(item.get("direction", "")).strip().lower()
            shap_value = float(item.get("shap_value", 0))

            if abs(shap_value) < 1e-4:
                continue

            feature_desc, gameplay_context = _humanize_shap_feature(feature_name, metadata_df)

            if direction == "increased":
                bullets.append(
                    f"{feature_desc} increases your win chances because {gameplay_context}."
                )
            elif direction == "decreased":
                bullets.append(
                    f"{feature_desc} lowers your win chances because {gameplay_context}."
                )

    except Exception as e:
        logger.warning("SHAP explanation failed: %s: %s", type(e).__name__, str(e)[:120])
        return []

    return bullets


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicate bullets while preserving order."""
    seen = set()
    output = []
    for item in items:
        key = item.strip().lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def build_prediction_explanations(
    model,
    feature_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    player_cards: List[int],
    opponent_cards: Optional[List[int]] = None,
    max_bullets: int = 4,
    player_win_prob: Optional[float] = None,
    debug: bool = False,
) -> List[str]:
    """
    Build 2-4 explanation bullets blending SHAP, rule-based signals, and matchup interactions.

    When player_win_prob is provided for matchup mode, explanations are framed
    around whether the player is favored or at a disadvantage.
    """
    if max_bullets < 2 or max_bullets > 4:
        raise ValueError("max_bullets must be between 2 and 4")

    shap_part = _shap_bullets(
        model=model,
        feature_df=feature_df,
        metadata_df=metadata_df,
        max_bullets=2,
    )

    if debug:
        logger.debug("SHAP bullets (%s): %s", len(shap_part), shap_part)

    rules_part = _rule_based_bullets(
        player_cards=player_cards,
        metadata_df=metadata_df,
        opponent_cards=opponent_cards,
    )

    if debug:
        logger.debug("Rule bullets (%s): %s", len(rules_part), rules_part)

    matchup_interactions: List[str] = []
    if opponent_cards:
        name_map, _, _ = _build_maps(metadata_df)
        matchup_interactions = _detect_matchup_interactions(
            player_cards=player_cards,
            opponent_cards=opponent_cards,
            name_map=name_map,
        )

        if debug:
            logger.debug("Matchup interaction bullets (%s): %s", len(matchup_interactions), matchup_interactions)

    merged = _dedupe_preserve_order(rules_part + matchup_interactions + shap_part)
    result = merged[:max_bullets]

    if len(result) < 2:
        if len(result) == 0:
            result.append("This deck's card choices and elixir profile strongly influence its expected performance.")
        if len(result) < 2:
            result.append("The prediction also depends on how well the deck's structure supports its main strategy.")

    # Add loss/win framing for matchup mode when win probability is known
    if player_win_prob is not None and opponent_cards:
        name_map, elixir_map, _ = _build_maps(metadata_df)
        player_archetype = detect_archetype(player_cards, name_map, elixir_map)
        opp_archetype = detect_archetype(opponent_cards, name_map, elixir_map)

        if player_win_prob < 40:
            if opp_archetype != "Unknown":
                framing = (
                    f"Overall, the opponent's {opp_archetype} deck holds a structural advantage "
                    f"in this matchup — outplays and smart elixir management would be needed to win."
                )
            else:
                framing = (
                    "Overall, the opponent's deck holds a structural advantage here — "
                    "you'd need to outplay on key interactions and manage elixir carefully to win."
                )
            result.append(framing)
        elif player_win_prob > 60:
            if player_archetype != "Unknown":
                framing = (
                    f"Overall, your {player_archetype} deck is well-positioned to win this matchup "
                    f"if you play to its strengths."
                )
            else:
                framing = "Overall, your deck has a clear structural edge in this matchup."
            result.append(framing)

    if debug:
        logger.debug("Final bullets (%s): %s", len(result), result)

    return result[:max_bullets]