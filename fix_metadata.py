
import chromadb
from pathlib import Path

client = chromadb.PersistentClient(
    path=str(Path("chroma_db").resolve())
)

collection = client.get_collection("nfl_plays")

print("Total records before:", collection.count())

# Inspect records in batches and add record_type only
# where it is missing. Existing documents and embeddings
# are preserved.
batch_size = 5000
offset = 0
fixed = 0

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

    ids_to_update = []
    metadata_to_update = []

    for record_id, metadata in zip(ids, metadatas):
        if not metadata or "record_type" not in metadata:
            updated_metadata = dict(metadata or {})

            source_file = updated_metadata.get("source_file", "")
            if "score" in source_file.lower():
                updated_metadata["record_type"] = "score"
            else:
                updated_metadata["record_type"] = "play"

            ids_to_update.append(record_id)
            metadata_to_update.append(updated_metadata)

    if ids_to_update:
        collection.update(
            ids=ids_to_update,
            metadatas=metadata_to_update
        )
        fixed += len(ids_to_update)
        print(f"Updated {fixed} records so far...")

    offset += len(ids)

print("\nMetadata repair complete.")
print("Records updated:", fixed)
print("Total records after:", collection.count())
