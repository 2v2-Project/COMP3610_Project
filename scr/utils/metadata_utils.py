from pathlib import Path
import json
import pandas as pd
import requests

CACHE_PATH = Path("data/processed/card_metadata.csv")
CACHE_JSON_PATH = Path("data/processed/card_metadata_raw.json")
CARDS_URL = "https://royaleapi.github.io/cr-api-data/json/cards.json"


def _fetch_raw_cards_from_api() -> list[dict]:
    """
    Fetch raw card metadata from Royale API.
    Returns a list of card dictionaries or empty list on error.
    """
    try:
        response = requests.get(CARDS_URL, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        print(f"Could not fetch card metadata from API: {error}")
        return []


def _load_raw_cards_cached(force_refresh=False) -> list[dict]:
    """
    Load raw card metadata from cache, fetching from API if needed.
    """
    if CACHE_JSON_PATH.exists() and not force_refresh:
        print("Loading raw card metadata from cache...")
        with open(CACHE_JSON_PATH, "r") as f:
            return json.load(f)

    print("Downloading card metadata from API...")
    cards = _fetch_raw_cards_from_api()

    if cards:
        CACHE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_JSON_PATH, "w") as f:
            json.dump(cards, f, indent=2)

    return cards


def get_raw_cards(force_refresh=False) -> list[dict]:
    """
    Get raw card metadata list from cache or API.
    Returns a list of card dictionaries.
    """
    return _load_raw_cards_cached(force_refresh=force_refresh)


def get_card_names(force_refresh=False) -> dict[int, str]:
    """
    Get a mapping of card_id -> card_name.
    Uses cached data when available.
    """
    cards = get_raw_cards(force_refresh=force_refresh)
    card_names = {}

    for card in cards:
        try:
            card_id = int(card.get("id"))
            name = card.get("name")
            if name:
                card_names[card_id] = name
        except (TypeError, ValueError):
            continue

    return card_names


def get_elixir_costs(force_refresh=False) -> dict[int, int]:
    """
    Get a mapping of card_id -> elixir_cost.
    Uses cached data when available.
    """
    cards = get_raw_cards(force_refresh=force_refresh)
    elixir_costs = {}

    for card in cards:
        try:
            card_id = int(card.get("id"))
            elixir = int(card.get("elixir")) if card.get("elixir") is not None else None
            if elixir is not None:
                elixir_costs[card_id] = elixir
        except (TypeError, ValueError):
            continue

    return elixir_costs


def get_card_types(force_refresh=False) -> dict[int, str]:
    """
    Get a mapping of card_id -> card_type.
    Normalized types: 'troop', 'spell', 'building'.
    Uses cached data when available.
    """
    cards = get_raw_cards(force_refresh=force_refresh)
    card_types = {}

    for card in cards:
        try:
            card_id = int(card.get("id"))
            card_type = str(card.get("type", "")).strip().lower()
            if card_type in {"troop", "spell", "building"}:
                card_types[card_id] = card_type
        except (TypeError, ValueError):
            continue

    return card_types


def get_card_metadata(force_refresh=False):
    """
    Get card metadata as a pandas DataFrame with full details.
    Includes: card_id, name, elixir, rarity, type, icon_url.
    """
    if CACHE_PATH.exists() and not force_refresh:
        print("Loading card metadata from cache...")
        return pd.read_csv(CACHE_PATH)

    print("Downloading card metadata from API...")
    cards = _fetch_raw_cards_from_api()

    rows = []
    for card in cards:
        rows.append({
            "card_id": card.get("id"),
            "name": card.get("name"),
            "elixir": card.get("elixir"),
            "rarity": card.get("rarity"),
            "type": card.get("type"),
            "icon_url": card.get("iconUrls", {}).get("medium")
        })

    df = pd.DataFrame(rows)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def main():
    df = get_card_metadata()
    print(df.head())

    df_2 = get_card_metadata(force_refresh=True)
    print(df_2.head())


if __name__ == "__main__":
    main() 