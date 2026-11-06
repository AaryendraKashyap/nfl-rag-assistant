"""Evaluate the NFL RAG assistant against ground truth computed from the CSVs.

Measures
  1. Exact-count accuracy   "How many touchdowns did <team> have in <season>?"
  2. Play-type precision    share of retrieved plays with the asked outcome,
                            compared with plain semantic search (baseline)
  3. Score accuracy         Super Bowl winner and games in a week
  4. Latency                median milliseconds per count/score question

Requires data/raw (run extract_data.py first) and an existing chroma_db.
Count and score checks do not need Ollama. The player-name check uses the
LLM, so it needs Ollama running to show a full answer.

Usage:
    python evaluate.py
    python evaluate.py --seasons 2020 2015 --out evaluation_results.md
"""

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import rag  # noqa: E402

RAW = ROOT / "data" / "raw"

TEAMS = {  # wording used in the question -> franchise name in the data
    "Kansas City": "Kansas City Chiefs",
    "Denver": "Denver Broncos",
    "Detroit": "Detroit Lions",
    "Dallas": "Dallas Cowboys",
}

OUTCOMES = {  # plural noun in the question -> exact PlayOutcome in the data
    "touchdowns": "Touchdown",
    "field goals": "Field Goal",
    "interceptions": "Interception",
}

PRECISION_QUESTIONS = {  # outcome -> question template
    "Touchdown": "Show me touchdown plays from {s}",
    "Field Goal": "Show me field goals from {s}",
    "Interception": "Show me interceptions in the {s} season",
}

COUNT_RE = re.compile(r"\*\*([\d,]+)\*\*")
OUTCOME_RE = re.compile(r"Outcome: (.*?)\. Description")
WINNER_RE = re.compile(r"Winner: (\w+)")
GAMES_RE = re.compile(r"\((\d+) games\)")


def load_plays(raw_dir, season):
    """Plays for one season, with the same empty-description rows dropped as ingest."""
    df = pd.read_csv(
        raw_dir / f"{season}_plays.csv",
        usecols=["TeamWithPossession", "PlayOutcome", "PlayDescription"],
        low_memory=False,
    ).fillna("")
    return df[df["PlayDescription"].str.strip() != ""]


def load_scores(raw_dir):
    df = pd.read_csv(raw_dir / "2010-2026_scores.csv").fillna("")
    df["SeasonNum"] = pd.to_numeric(df["Season"], errors="coerce")
    return df


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def count_checks(plays, season, player_check=True):
    rows = []

    for team_word, team in TEAMS.items():
        for noun, outcome in OUTCOMES.items():
            question = f"How many {noun} did {team_word} have in {season}?"
            expected = int(
                ((plays.TeamWithPossession == team) & (plays.PlayOutcome == outcome)).sum()
            )
            result, ms = timed(rag.ask_question, question)
            match = COUNT_RE.search(result["answer"])
            got = int(match.group(1).replace(",", "")) if match else None

            rows.append({
                "area": "Exact count",
                "question": question,
                "expected": expected,
                "got": got,
                "passed": got == expected,
                "ms": ms,
            })

    # A player name is not a plain count: the app must warn, not give a number.
    # Only run once (it calls the LLM), so it is skipped for other seasons.
    if not player_check:
        return rows

    question = f"How many touchdowns did Patrick Mahomes have in {season}?"
    result, _ = timed(rag.ask_question, question)
    warned = bool(result.get("warning")) and not COUNT_RE.search(result["answer"])
    rows.append({
        "area": "Exact count",
        "question": question,
        "expected": "warning, no count",
        "got": "warning" if result.get("warning") else "no warning",
        "passed": warned,
        "ms": None,
    })

    return rows


def precision_checks(season):
    rows = []

    for outcome, template in PRECISION_QUESTIONS.items():
        question = template.format(s=season)

        records, ms = timed(rag.retrieve_plays, question, top_k=5)
        ours = [_outcome(r["document"]) for r in records]

        baseline_records = list(
            rag.query_candidates(question, {"season": season}, None, 50).values()
        )[:5]
        baseline = [_outcome(r["document"]) for r in baseline_records]

        ours_share = _share(ours, outcome)
        base_share = _share(baseline, outcome)

        rows.append({
            "area": "Play-type precision",
            "question": question,
            "expected": f"100% {outcome}",
            "got": f"{ours_share:.0%} (baseline {base_share:.0%})",
            "passed": ours_share == 1.0,
            "ms": ms,
            "ours": ours_share,
            "baseline": base_share,
        })

    return rows


def score_checks(scores, season):
    rows = []
    season_scores = scores[scores.SeasonNum == season]

    super_bowl = season_scores[
        (season_scores.Week.str.upper() == "SUPER BOWL")
        & (season_scores.GameStatus.str.upper() == "FINAL")
    ]

    if len(super_bowl):
        game = super_bowl.iloc[0]
        expected = (
            game.HomeTeam
            if float(game.HomeScore) > float(game.AwayScore)
            else game.AwayTeam
        )
        question = f"Who won the {season} Super Bowl?"
        result, ms = timed(rag.ask_question, question)
        match = WINNER_RE.search(result["answer"])
        got = match.group(1) if match else None
        rows.append({
            "area": "Score lookup",
            "question": question,
            "expected": expected,
            "got": got,
            "passed": got == expected,
            "ms": ms,
        })

    week_one = season_scores[
        (season_scores.Week.str.upper() == "WEEK 1")
        & (season_scores.GameStatus.str.upper() == "FINAL")
    ]
    expected = len(week_one)
    question = f"What are the final scores of week 1 in {season}?"
    result, ms = timed(rag.ask_question, question)
    match = GAMES_RE.search(result["answer"])
    # No "(N games)" text means zero FINAL games, which is correct for seasons
    # that have not been played yet (for example, 2026 week 1).
    got = int(match.group(1)) if match else (0 if expected == 0 else None)
    rows.append({
        "area": "Score lookup",
        "question": question,
        "expected": expected,
        "got": got,
        "passed": got == expected,
        "ms": ms,
    })

    return rows


def _outcome(document):
    match = OUTCOME_RE.search(document)
    return match.group(1) if match else None


def _share(outcomes, wanted):
    if not outcomes:
        return 0.0
    return sum(o == wanted for o in outcomes) / len(outcomes)


def run(raw_dir, seasons):
    scores = load_scores(raw_dir)
    rows = []

    for season in seasons:
        plays = load_plays(raw_dir, season)
        rows += count_checks(plays, season, player_check=(season == seasons[0]))
        rows += precision_checks(season)
        rows += score_checks(scores, season)

    return rows


def summarise(rows):
    counts = [r for r in rows if r["area"] == "Exact count"]
    precision = [r for r in rows if r["area"] == "Play-type precision"]
    scores = [r for r in rows if r["area"] == "Score lookup"]

    latency = {}
    for area in ("Exact count", "Play-type precision", "Score lookup"):
        times = [r["ms"] for r in rows if r["area"] == area and r["ms"] is not None]
        latency[area] = statistics.median(times) if times else None

    return {
        "count_passed": sum(r["passed"] for r in counts),
        "count_total": len(counts),
        "precision_ours": statistics.mean(r["ours"] for r in precision),
        "precision_baseline": statistics.mean(r["baseline"] for r in precision),
        "score_passed": sum(r["passed"] for r in scores),
        "score_total": len(scores),
        "latency": latency,
    }


def render(rows, seasons):
    s = summarise(rows)
    lines = [
        "# Evaluation results",
        "",
        f"Seasons tested: {', '.join(str(x) for x in seasons)}",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Exact-count accuracy | {s['count_passed']}/{s['count_total']} |",
        f"| Play-type precision, this system | {s['precision_ours']:.0%} |",
        f"| Play-type precision, semantic baseline | {s['precision_baseline']:.0%} |",
        f"| Score lookup accuracy | {s['score_passed']}/{s['score_total']} |",
    ]

    for area, ms in s["latency"].items():
        value = f"{ms:,.0f} ms" if ms is not None else "n/a"
        lines.append(f"| Median latency, {area.lower()} | {value} |")

    lines += [
        "",
        "## Detail",
        "",
        "| Area | Question | Expected | Got | Pass |",
        "|---|---|---|---|---|",
    ]

    for r in rows:
        mark = "yes" if r["passed"] else "no"
        lines.append(
            f"| {r['area']} | {r['question']} | {r['expected']} | {r['got']} | {mark} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, default=RAW, help="folder with the CSVs")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2020])
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation_results.md")
    args = parser.parse_args()

    rows = run(args.raw, args.seasons)
    report = render(rows, args.seasons)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
