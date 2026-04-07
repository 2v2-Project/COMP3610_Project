"""
Create sampled parquet files for the Streamlit Cloud webapp.
=============================================================
The full clean parquet (723 MB, 12.4 M rows) is too large for
Streamlit Cloud's ~1 GB RAM limit.  This script produces a 500 K
random sample (~35 MB) that includes archetype and elixir columns
from the companion feature files, so the webapp only needs ONE
parquet download.

Run locally:
    python scr/15_create_sample_parquet.py

Then upload the output file to HuggingFace:
    huggingface-cli upload lillyem/clash-royale-data \
        data/processed/clash_royale_clean_500k.parquet \
        clash_royale_clean_500k.parquet --repo-type dataset
"""

from pathlib import Path

import duckdb

SAMPLE_SIZE = 500_000
OUTPUT = Path("data/processed/clash_royale_clean_500k.parquet")

# Local paths
CLEAN_LOCAL = Path("data/processed/clash_royale_clean.parquet")
ARCH_LOCAL = Path("data/processed/archetype_features.parquet")
ELIXIR_LOCAL = Path("data/processed/deck_elixir_features.parquet")

HF_REPO = "lillyem/clash-royale-data"


def _resolve(local: Path, hf_name: str) -> str:
    """Return local path string, or download from HuggingFace."""
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=hf_name, repo_type="dataset")


def main() -> None:
    clean = _resolve(CLEAN_LOCAL, "clash_royale_clean.parquet")
    arch  = _resolve(ARCH_LOCAL,  "archetype_features.parquet")
    elixir = _resolve(ELIXIR_LOCAL, "deck_elixir_features.parquet")

    print(f"CLEAN  : {clean}")
    print(f"ARCH   : {arch}")
    print(f"ELIXIR : {elixir}")

    con = duckdb.connect()
    total = con.sql(f"SELECT count(*) AS n FROM '{clean}'").fetchone()[0]
    print(f"Total rows: {total:,}")

    # POSITIONAL JOIN all three (row-aligned), then sample
    con.sql(f"""
        COPY (
            SELECT
                c.*,
                a.player_archetype,
                e.player_avg_elixir,
                e.player_cycle_cards,
                e.player_troop_count,
                e.player_spell_count,
                e.player_building_count
            FROM '{clean}' c
            POSITIONAL JOIN '{arch}' a
            POSITIONAL JOIN '{elixir}' e
            USING SAMPLE {SAMPLE_SIZE}
        ) TO '{OUTPUT}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    out_rows = con.sql(f"SELECT count(*) AS n FROM '{OUTPUT}'").fetchone()[0]
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"Wrote {out_rows:,} rows → {OUTPUT}  ({size_mb:.1f} MB)")
    con.close()


if __name__ == "__main__":
    main()
