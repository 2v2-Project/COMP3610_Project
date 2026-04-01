from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd


def build_deck_key(card_ids: list[int]) -> str:
    """Create a canonical deck key so the same deck always groups together."""
    return ",".join(str(card_id) for card_id in sorted(card_ids))


def parse_deck_key(deck_key: str) -> list[int]:
    """Convert deck key string back into a list of card ids."""
    if not deck_key:
        return []
    return [int(x) for x in deck_key.split(",") if x.strip()]


def compute_avg_elixir(card_ids: list[int], elixir_map: dict[int, int]) -> float:
    costs = [elixir_map.get(card_id) for card_id in card_ids if elixir_map.get(card_id) is not None]
    if not costs:
        return 0.0
    return round(sum(costs) / len(costs), 2)


def compute_cycle_cost(card_ids: list[int], elixir_map: dict[int, int]) -> int:
    """
    Approximate cycle cost = sum of cheapest 4 cards.
    This is a common lightweight Clash Royale cycle indicator.
    """
    costs = sorted(elixir_map.get(card_id, 99) for card_id in card_ids)
    valid_costs = [c for c in costs if c != 99]
    if len(valid_costs) < 4:
        return 0
    return int(sum(valid_costs[:4]))


def count_card_types(card_ids: list[int], type_map: dict[int, str]) -> dict[str, int]:
    counts = Counter(type_map.get(card_id, "unknown") for card_id in card_ids)
    return {
        "troop_count": counts.get("troop", 0),
        "spell_count": counts.get("spell", 0),
        "building_count": counts.get("building", 0),
    }


def detect_archetype(
    card_ids: list[int],
    name_map: dict[int, str],
    elixir_map: dict[int, int],
) -> str:
    """
    Rule-based archetype detector.
    This is intentionally simple and app-friendly.
    """
    names = {name_map.get(card_id, "").lower() for card_id in card_ids}
    avg_elixir = compute_avg_elixir(card_ids, elixir_map)

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

    if has("royal giant") or has("pekka") or has("bandit") or has("battle ram"):
        return "Bridge Spam"

    if has("graveyard"):
        return "Graveyard"

    if has("balloon"):
        return "Loon"

    if has("sparky"):
        return "Sparky"

    if has("miner") and avg_elixir <= 3.8:
        return "Miner Control"

    if avg_elixir <= 3.6:
        return "Control"

    return "Unknown"


def get_confidence_label_from_matches(matches_played: int) -> str:
    if matches_played >= 500:
        return "High"
    if matches_played >= 100:
        return "Medium"
    return "Low"


def enrich_deck_record(
    deck_key: str,
    matches_played: int,
    wins: int,
    name_map: dict[int, str],
    elixir_map: dict[int, int],
    type_map: dict[int, str],
) -> dict:
    card_ids = parse_deck_key(deck_key)
    avg_elixir = compute_avg_elixir(card_ids, elixir_map)
    cycle_cost = compute_cycle_cost(card_ids, elixir_map)
    type_counts = count_card_types(card_ids, type_map)
    archetype = detect_archetype(card_ids, name_map, elixir_map)
    win_rate = round((wins / matches_played) * 100, 2) if matches_played > 0 else 0.0

    return {
        "deck_key": deck_key,
        "card_ids": card_ids,
        "matches_played": matches_played,
        "wins": wins,
        "win_rate": win_rate,
        "confidence": get_confidence_label_from_matches(matches_played),
        "archetype": archetype,
        "avg_elixir": avg_elixir,
        "cycle_cost": cycle_cost,
        **type_counts,
    }