
from pathlib import Path
from zipfile import ZipFile

archive_path = Path("archive.zip")
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

# Extract only the top-level season CSV files.
# This avoids extracting duplicate weekly CSVs.
with ZipFile(archive_path, "r") as archive:
    for member in archive.namelist():
        path = Path(member)

        if (
            len(path.parts) == 1
            and path.suffix.lower() == ".csv"
            and (
                path.name.endswith("_plays.csv")
                or path.name == "2010-2026_scores.csv"
            )
        ):
            target = output_dir / path.name

            with archive.open(member) as source, target.open("wb") as dest:
                dest.write(source.read())

print("Dataset extraction completed.")
print("Files:", len(list(output_dir.glob("*.csv"))))
