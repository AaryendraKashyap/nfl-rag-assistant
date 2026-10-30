
import chromadb
from pathlib import Path
from collections import Counter

client = chromadb.PersistentClient(
    path=str(Path("chroma_db").resolve())
)
collection = client.get_collection("nfl_plays")

batch_size = 5000
offset = 0
missing_count = 0
source_counts = Counter()
examples = []

while True:
    results = collection.get(
        limit=batch_size,
        offset=offset,
        include=["metadatas"]
    )

    ids = results["ids"]
    metadatas = results["metadatas"]

    if not ids:
        break

    for record_id, metadata in zip(ids, metadatas):
        metadata = metadata or {}

        if "record_type" not in metadata:
            missing_count += 1
            source_counts[metadata.get("source_file", "(missing)")] += 1

            if len(examples) < 5:
                examples.append((record_id, metadata))

    offset += len(ids)

print("Records missing record_type:", missing_count)
print("\nCounts by source file:")

for source, count in sorted(source_counts.items()):
    print(f"{source}: {count}")

print("\nSample records:")
for record_id, metadata in examples:
    print(record_id, metadata)
