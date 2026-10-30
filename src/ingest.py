
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_DIR = ROOT / "chroma_db"
BATCH_SIZE = 128
COLLECTION_NAME = "nfl_plays"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"description": "NFL play-by-play and score records"},
)


def clean(value):
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip()
    return value if value else "Unknown"


def add_resumable(records, label):
    """Add only IDs that are not already stored."""
    added = 0

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
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

        checked = min(start + BATCH_SIZE, len(records))
        print(
            f"\r{label}: checked {checked:,}/{len(records):,}; "
            f"new records {added:,}",
            end="",
            flush=True,
        )

    print()
    return added


# Prefer season-wide files; do not ingest weekly files as well.
play_files = sorted(RAW_DIR.glob("*_plays.csv"))

if not play_files:
    raise FileNotFoundError(
        f"No season-wide play files found in {RAW_DIR}"
    )

total_added = 0

for path in play_files:
    try:
        season = int(path.name.split("_")[0])
    except ValueError:
        print(f"Skipping unrecognized file: {path.name}")
        continue

    print(f"\nReading {path.name}")
    df = pd.read_csv(path).fillna("")
    records = []

    for index, row in df.iterrows():
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
                "source_file": path.name,
                "row_number": int(index),
            },
        })

    added = add_resumable(records, path.name)
    total_added += added
    print(f"{season}: added {added:,} new play records.")


# Index scores from the combined file, if present.
# If absent, look for individual score files without indexing
# the same combined file twice.
combined_scores = RAW_DIR / "2010-2026_scores.csv"

if combined_scores.exists():
    score_files = [combined_scores]
else:
    score_files = sorted(
        p for p in RAW_DIR.rglob("*scores.csv")
        if p.name != "2010-2026_scores.csv"
    )

for path in score_files:
    print(f"\nReading scores: {path.relative_to(ROOT)}")
    df = pd.read_csv(path).fillna("")
    records = []

    for index, row in df.iterrows():
        season_value = row.get("Season", None)

        if pd.isna(season_value) or season_value == "":
            # Infer season from a filename such as 2020_scores.csv.
            try:
                season_value = int(path.name.split("_")[0])
            except ValueError:
                continue

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
            "id": f"score_{path.stem}_{index}",
            "document": f"Record type: game score. Season: {season}. {details}",
            "metadata": {
                "record_type": "score",
                "season": season,
                "is_scoring_play": 0,
                "source_file": path.name,
                "row_number": int(index),
            },
        })

    added = add_resumable(records, path.name)
    total_added += added
    print(f"New score records: {added:,}")

print("\n========== INGESTION SUMMARY ==========")
print(f"New records added this run: {total_added:,}")
print(f"Total records in collection: {collection.count():,}")
print("Ingestion finished.")
