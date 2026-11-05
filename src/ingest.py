"""Build or update the ChromaDB collection from the CSV files in data/raw.

Safe to re-run: play records that are already stored are skipped (their IDs
are stable), so adding new weeks of the current season only embeds the new
rows. Score rows are re-checked because their text changes over a season
(TBD -> FINAL); only rows whose text changed are re-embedded.

The document text and IDs are deliberately identical to the first version of
this script, so databases built earlier stay compatible.

Usage:
    python src/ingest.py                 # everything (skips what exists)
    python src/ingest.py --seasons 2026  # only new 2026 plays + scores
    python src/ingest.py --scores-only   # only refresh the score records
"""

import argparse
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_DIR = ROOT / "chroma_db"
BATCH_SIZE = 256
COLLECTION_NAME = "nfl_plays"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # must match src/rag.py
COMBINED_SCORES = "2010-2026_scores.csv"


def open_collection():
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(DB_DIR))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"description": "NFL play-by-play and score records"},
    )


def clean(value):
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip()
    return value if value else "Unknown"


def build_play_records(df, season, source_file):
    """Turn one season DataFrame into Chroma records (same format as v1)."""
    records = []

    for index, row in enumerate(df.to_dict("records")):
        description = clean(row.get("PlayDescription", ""))
        if description == "Unknown":
            continue

        scoring = pd.to_numeric(
            row.get("IsScoringPlay", 0), errors="coerce"
        )
        scoring = int(scoring) if pd.notna(scoring) else 0

        document = (
            f"Record type: play. Season: {season}. "
            f"Week: {clean(row.get('Week', 'Unknown'))}. "
            f"Date: {clean(row.get('Date', 'Unknown'))}. "
            f"Home team: {clean(row.get('HomeTeam', 'Unknown'))}. "
            f"Away team: {clean(row.get('AwayTeam', 'Unknown'))}. "
            f"Quarter: {clean(row.get('Quarter', 'Unknown'))}. "
            f"Possession: "
            f"{clean(row.get('TeamWithPossession', 'Unknown'))}. "
            f"Scoring play: {scoring}. "
            f"Outcome: {clean(row.get('PlayOutcome', 'Unknown'))}. "
            f"Description: {description}"
        )

        records.append({
            # Stable IDs allow this script to resume.
            "id": f"{season}_{index}",
            "document": document,
            "metadata": {
                "record_type": "play",
                "season": season,
                "is_scoring_play": scoring,
                "source_file": source_file,
                "row_number": int(index),
            },
        })

    return records


def build_score_records(df, source_file, file_stem, default_season=None):
    """Turn a scores DataFrame into Chroma records (same format as v1)."""
    records = []

    for index, row in enumerate(df.to_dict("records")):
        season_value = row.get("Season", None)

        if pd.isna(season_value) or season_value == "":
            # Infer season from a filename such as 2020_scores.csv.
            if default_season is None:
                continue
            season_value = default_season

        try:
            season = int(float(season_value))
        except (ValueError, TypeError):
            continue

        # Keep all original score columns in the text. This avoids
        # assuming the score CSV uses particular column names.
        details = "; ".join(
            f"{column}: {clean(value)}"
            for column, value in row.items()
        )

        records.append({
            "id": f"score_{file_stem}_{index}",
            "document": f"Record type: game score. Season: {season}. {details}",
            "metadata": {
                "record_type": "score",
                "season": season,
                "is_scoring_play": 0,
                "source_file": source_file,
                "row_number": int(index),
            },
        })

    return records


def _progress(label, start, batch_size, total, written):
    checked = min(start + batch_size, total)
    print(
        f"\r{label}: checked {checked:,}/{total:,}; "
        f"written {written:,}",
        end="",
        flush=True,
    )


def add_resumable(collection, records, label, batch_size=BATCH_SIZE):
    """Add only IDs that are not already stored (existing rows are skipped)."""
    added = 0

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        ids = [r["id"] for r in batch]

        existing = collection.get(ids=ids, include=[])
        existing_ids = set(existing["ids"])
        missing = [r for r in batch if r["id"] not in existing_ids]

        if missing:
            collection.add(
                ids=[r["id"] for r in missing],
                documents=[r["document"] for r in missing],
                metadatas=[r["metadata"] for r in missing],
            )
            added += len(missing)

        _progress(label, start, batch_size, len(records), added)

    print()
    return added


def upsert_changed(collection, records, label, batch_size=BATCH_SIZE):
    """Add new records and re-embed only those whose text has changed."""
    written = 0

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        ids = [r["id"] for r in batch]

        existing = collection.get(ids=ids, include=["documents"])
        stored = dict(zip(existing["ids"], existing["documents"] or []))
        changed = [r for r in batch if stored.get(r["id"]) != r["document"]]

        if changed:
            collection.upsert(
                ids=[r["id"] for r in changed],
                documents=[r["document"] for r in changed],
                metadatas=[r["metadata"] for r in changed],
            )
            written += len(changed)

        _progress(label, start, batch_size, len(records), written)

    print()
    return written


def season_of(path):
    try:
        return int(path.name.split("_")[0])
    except ValueError:
        return None


def ingest_plays(collection, seasons=None, batch_size=BATCH_SIZE):
    # Prefer season-wide files; do not ingest weekly files as well.
    play_files = sorted(RAW_DIR.glob("*_plays.csv"))

    if not play_files:
        raise FileNotFoundError(
            f"No season-wide play files found in {RAW_DIR}"
        )

    total_added = 0

    for path in play_files:
        season = season_of(path)

        if season is None:
            print(f"Skipping unrecognized file: {path.name}")
            continue

        if seasons and season not in seasons:
            continue

        print(f"\nReading {path.name}")
        df = pd.read_csv(path).fillna("")
        records = build_play_records(df, season, path.name)

        added = add_resumable(collection, records, path.name, batch_size)
        total_added += added
        print(f"{season}: added {added:,} new play records.")

    return total_added


def ingest_scores(collection, batch_size=BATCH_SIZE):
    # Index scores from the combined file, if present. If absent, look for
    # individual score files without indexing the combined file twice.
    combined = RAW_DIR / COMBINED_SCORES

    if combined.exists():
        score_files = [combined]
    else:
        score_files = sorted(
            p for p in RAW_DIR.rglob("*scores.csv")
            if p.name != COMBINED_SCORES
        )

    total_written = 0

    for path in score_files:
        print(f"\nReading scores: {path.relative_to(ROOT)}")
        df = pd.read_csv(path).fillna("")
        records = build_score_records(
            df, path.name, path.stem, default_season=season_of(path)
        )

        written = upsert_changed(collection, records, path.name, batch_size)
        total_written += written
        print(f"New or updated score records: {written:,}")

    return total_written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--seasons", type=int, nargs="*",
        help="only ingest plays for these seasons (scores are always checked)",
    )
    parser.add_argument(
        "--scores-only", action="store_true",
        help="skip play files and only refresh score records",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    collection = open_collection()
    before = collection.count()
    total = 0

    if not args.scores_only:
        total += ingest_plays(
            collection, set(args.seasons) if args.seasons else None,
            args.batch_size,
        )

    total += ingest_scores(collection, args.batch_size)

    print("\n========== INGESTION SUMMARY ==========")
    print(f"Records before this run: {before:,}")
    print(f"New or updated records this run: {total:,}")
    print(f"Total records in collection: {collection.count():,}")
    print("Ingestion finished.")


if __name__ == "__main__":
    main()
