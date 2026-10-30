
from pathlib import Path
from collections import Counter

import chromadb

db_path = Path("chroma_db").resolve()

client = chromadb.PersistentClient(path=str(db_path))
collection = client.get_collection("nfl_plays")

print(f"Total records: {collection.count():,}")

season_counts = Counter()
type_counts = Counter()
source_counts = Counter()

offset = 0
batch_size = 5000
total = collection.count()

print("\nReading stored metadata...")

while offset < total:
    result = collection.get(
        limit=batch_size,
        offset=offset,
        include=["metadatas"],
    )

    if not result["ids"]:
        break

    for metadata in result["metadatas"]:
        if not metadata:
            continue

        if "season" in metadata:
            season_counts[int(metadata["season"])] += 1

        type_counts[metadata.get("record_type", "unknown")] += 1
        source_counts[metadata.get("source_file", "unknown")] += 1

    offset += len(result["ids"])
    print(f"\rChecked {offset:,}/{total:,} records", end="", flush=True)

print("\n\nRECORDS BY SEASON")
for season, count in sorted(season_counts.items()):
    print(f"{season}: {count:,}")

print("\nRECORDS BY TYPE")
for record_type, count in type_counts.items():
    print(f"{record_type}: {count:,}")

print("\nNUMBER OF DISTINCT SOURCE FILES")
print(len(source_counts))

print("\nDATABASE VERIFICATION COMPLETE")
