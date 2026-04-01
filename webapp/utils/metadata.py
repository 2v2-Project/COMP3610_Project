from pathlib import Path
import json
import pandas as pd
import requests

CACHE_PATH = Path("data/processed/card_metadata.csv")
CACHE_JSON_PATH = Path("data/processed/card_metadata_raw.json")
CARDS_URL = "https://royaleapi.github.io/cr-api-data/json/cards.json"


def _fetch_raw_cards_from_api() -> list[dict]:
    try:
        response = requests.get(CARDS_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            return data

        print("Unexpected card metadata format from API.")
        return []
    except Exception as error:
        print(f"Could not fetch card metadata from API: {error}")
        return []


def _load_raw_cards_cached(force_refresh: bool = False) -> list[dict]:
    if CACHE_JSON_PATH.exists() and not force_refresh:
        try:
            with open(CACHE_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as error:
            print(f"Could not read cached raw card metadata: {error}")

    cards = _fetch_raw_cards_from_api()

    if cards:
        CACHE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2)

    return cards


def get_raw_cards(force_refresh: bool = False) -> list[dict]:
    return _load_raw_cards_cached(force_refresh=force_refresh)


def _extract_icon_url(card: dict) -> str | None:
    icon_data = card.get("iconUrls")

    if not isinstance(icon_data, dict):
        return None

    icon_url = (
        icon_data.get("medium")
        or icon_data.get("evolutionMedium")
        or icon_data.get("base")
        or icon_data.get("evolution")
    )

    if isinstance(icon_url, str) and icon_url.strip():
        return icon_url.strip()

    return None


def get_card_names(force_refresh: bool = False) -> dict[int, str]:
    cards = get_raw_cards(force_refresh=force_refresh)
    card_names: dict[int, str] = {}

    for card in cards:
        try:
            card_id = int(card.get("id"))
            name = card.get("name")
            if name:
                card_names[card_id] = str(name)
        except (TypeError, ValueError):
            continue

    return card_names


def get_elixir_costs(force_refresh: bool = False) -> dict[int, int]:
    cards = get_raw_cards(force_refresh=force_refresh)
    elixir_costs: dict[int, int] = {}

    for card in cards:
        try:
            card_id = int(card.get("id"))
            elixir = card.get("elixir")
            if elixir is not None:
                elixir_costs[card_id] = int(elixir)
        except (TypeError, ValueError):
            continue

    return elixir_costs


def get_card_types(force_refresh: bool = False) -> dict[int, str]:
    cards = get_raw_cards(force_refresh=force_refresh)
    card_types: dict[int, str] = {}

    for card in cards:
        try:
            card_id = int(card.get("id"))
            card_type = str(card.get("type", "")).strip().lower()
            if card_type in {"troop", "spell", "building"}:
                card_types[card_id] = card_type
        except (TypeError, ValueError):
            continue

    return card_types


def get_icon_urls(force_refresh: bool = False) -> dict[int, str]:
    cards = get_raw_cards(force_refresh=force_refresh)
    icon_urls: dict[int, str] = {}

    for card in cards:
        try:
            card_id = int(card.get("id"))
        except (TypeError, ValueError):
            continue

        icon_url = _extract_icon_url(card)
        if icon_url:
            icon_urls[int(card_id)] = icon_url

    return icon_urls


def get_card_metadata(force_refresh: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force_refresh:
        try:
            df = pd.read_csv(CACHE_PATH)

            if "card_id" in df.columns:
                df["card_id"] = pd.to_numeric(df["card_id"], errors="coerce")
                df = df.dropna(subset=["card_id"]).copy()
                df["card_id"] = df["card_id"].astype(int)

            return df
        except Exception as error:
            print(f"Could not read cached card metadata CSV: {error}")

    cards = get_raw_cards(force_refresh=force_refresh)

    rows = []
    for card in cards:
        try:
            card_id = int(card.get("id"))
        except (TypeError, ValueError):
            continue

        rows.append(
            {
                "card_id": int(card_id),
                "name": card.get("name"),
                "elixir": card.get("elixir"),
                "rarity": card.get("rarity"),
                "type": str(card.get("type", "")).strip().lower() if card.get("type") else None,
                "icon_url": _extract_icon_url(card),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["card_id"] = pd.to_numeric(df["card_id"], errors="coerce")
        df = df.dropna(subset=["card_id"]).copy()
        df["card_id"] = df["card_id"].astype(int)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def main():
    df = get_card_metadata(force_refresh=True)
    print(df.head())
    print("Sample icon urls:", list(get_icon_urls(force_refresh=True).items())[:5])


if __name__ == "__main__":
    main()