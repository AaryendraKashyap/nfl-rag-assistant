"""Quick health check of the ChromaDB collection (read-only).

Counts records per season and record type with metadata filters, instead of
paging through the metadata of every record, so it finishes in seconds.

Usage:
    python check_database.py
"""

import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.rag import COLLECTION_NAME, DB_DIR, FIRST_SEASON, LAST_SEASON  # noqa: E402


def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()

    print(f"Total records: {total:,}\n")
    print(f"{'season':<8}{'play':>10}{'score':>8}")

    counted = 0

    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        row = {}

        for record_type in ("play", "score"):
            ids = collection.get(
                where={"$and": [
                    {"season": season},
                    {"record_type": record_type},
                ]},
                include=[],
            )["ids"]
            row[record_type] = len(ids)

        counted += row["play"] + row["score"]
        print(f"{season:<8}{row['play']:>10,}{row['score']:>8,}")

    missing = total - counted

    print(f"\nCounted: {counted:,}")

    if missing:
        print(
            f"WARNING: {missing:,} records have no season/record_type "
            "metadata (or a season outside the range above)."
        )
    else:
        print("All records have season and record_type metadata.")


if __name__ == "__main__":
    main()
