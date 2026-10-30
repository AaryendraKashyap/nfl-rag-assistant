
import chromadb
from pathlib import Path

client = chromadb.PersistentClient(
    path=str(Path("chroma_db").resolve())
)

collection = client.get_collection("nfl_plays")

print("Total records:", collection.count())
print("\nSearching for records without record_type...\n")

# Read a small sample only; this does not modify the database.
results = collection.get(
    limit=100,
    include=["metadatas", "documents"]
)

found = 0

for metadata, document in zip(
    results["metadatas"],
    results["documents"]
):
    if not metadata or "record_type" not in metadata:
        print("Metadata:", metadata)
        print("Document:", (document or "")[:300])
        print("-" * 60)
        found += 1

        if found >= 5:
            break

if found == 0:
    print("No records missing record_type in the first 100 records.")
