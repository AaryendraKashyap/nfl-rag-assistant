"""Extract the season-wide CSV files from the dataset archive into data/raw.

Only the top-level season files are extracted (<year>_plays.csv and
2010-2026_scores.csv). The weekly CSVs inside the archive are duplicates of
the same plays and would be ingested twice.

Usage:
    python extract_data.py
    python extract_data.py --archive path/to/archive.zip --out data/raw
"""

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
SCORES_FILE = "2010-2026_scores.csv"


def wanted(member):
    path = Path(member)

    return (
        len(path.parts) == 1
        and path.suffix.lower() == ".csv"
        and (path.name.endswith("_plays.csv") or path.name == SCORES_FILE)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", type=Path, default=ROOT / "archive.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"Archive not found: {args.archive}")

    args.out.mkdir(parents=True, exist_ok=True)

    with ZipFile(args.archive, "r") as archive:
        for member in archive.namelist():
            if not wanted(member):
                continue

            target = args.out / Path(member).name

            # Stream to disk instead of loading each CSV into memory.
            with archive.open(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)

    print("Dataset extraction completed.")
    print("Files:", len(list(args.out.glob("*.csv"))))


if __name__ == "__main__":
    main()
