from pathlib import Path
import re

import chromadb
import ollama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAC": "Jacksonville Jaguars",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams", "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets", "OAK": "Oakland Raiders",
    "LV": "Las Vegas Raiders", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SD": "San Diego Chargers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington",
    "WSH": "Washington Commanders", "STL": "St. Louis Rams", "LVR": "Las Vegas Raiders",
}


def team_label(abbreviation):
    """Return a readable team name while preserving the source abbreviation."""
    code = str(abbreviation).strip().upper()
    name = TEAM_NAMES.get(code)
    return f"{name} ({code})" if name else code


ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "chroma_db"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(
    name="nfl_plays",
    embedding_function=embedding_function,
)


def infer_question_settings(question, season, scoring_only, record_type):
    """Infer season, record-type, and scoring filters from a question."""
    q = question.lower()

    # Streamlit select boxes may pass "Play" / "Score" while Chroma metadata
    # stores lowercase values. Normalize once so filters remain consistent.
    if record_type is None or str(record_type).strip().lower() == "all":
        record_type = "All"
    else:
        normalized_type = str(record_type).strip().lower()
        record_type = normalized_type if normalized_type in {"play", "score"} else record_type

    range_match = re.search(
        r"\b(20\d{2})\s*(?:through|thru|to|[-–])\s*(20\d{2})\b",
        q,
    )

    seasons = None

    if range_match:
        start_year = int(range_match.group(1))
        end_year = int(range_match.group(2))

        if (
            2010 <= start_year <= 2026
            and 2010 <= end_year <= 2026
            and start_year <= end_year
        ):
            seasons = list(range(start_year, end_year + 1))
            season = None
    else:
        years = [
            int(year)
            for year in re.findall(r"\b(20\d{2})\b", q)
            if 2010 <= int(year) <= 2026
        ]
        years = list(dict.fromkeys(years))

        if len(years) >= 2:
            seasons = years
            season = None
        elif len(years) == 1:
            season = years[0]
            seasons = [years[0]]

    if record_type == "All":
        score_phrases = (
            "final score", "final scores", "game scores", "score of the game",
            "scores of games", "what was the score", "what is the score",
            "who won", "game results", "final result", "result of the game",
            "winning team", "winner",
        )
        asks_teams_in_named_game = (
            ("hall of fame" in q or "super bowl" in q or "superbowl" in q)
            and bool(re.search(r"\b(which|what) teams?\b|\bwho played\b|\bteams played\b", q))
        )
        asks_score_in_named_game = (
            ("hall of fame" in q or "super bowl" in q or "superbowl" in q)
            and bool(re.search(r"\bscores?\b|\bwon\b|\bwinner\b", q))
        )
        if any(phrase in q for phrase in score_phrases) or asks_teams_in_named_game or asks_score_in_named_game:
            record_type = "score"

    if record_type != "score" and any(
        term in q
        for term in ("touchdown", "touchdowns", "scoring plays")
    ):
        scoring_only = True

    return season, seasons, scoring_only, record_type


def build_where_filter(
    season=None,
    seasons=None,
    scoring_only=False,
    record_type="All",
):
    """Build a ChromaDB metadata filter supporting multiple seasons."""
    if record_type is None or str(record_type).strip().lower() == "all":
        record_type = "All"
    else:
        normalized_type = str(record_type).strip().lower()
        if normalized_type in {"play", "score"}:
            record_type = normalized_type

    conditions = []

    if seasons:
        if len(seasons) == 1:
            conditions.append({"season": int(seasons[0])})
        else:
            conditions.append(
                {"season": {"$in": [int(year) for year in seasons]}}
            )
    elif season is not None:
        conditions.append({"season": int(season)})

    if record_type != "All":
        conditions.append({"record_type": record_type})

    if scoring_only and record_type != "score":
        conditions.append({"is_scoring_play": 1})

    if len(conditions) == 1:
        return conditions[0]

    if len(conditions) > 1:
        return {"$and": conditions}

    return None


def record_number(record_id):
    """Return the numeric suffix of an ID such as score_file_3990."""
    match = re.search(r"_(\d+)$", record_id)
    return int(match.group(1)) if match else 10**12


def score_field(document, name):
    """Extract a field from a score record."""
    match = re.search(
        rf"\b{re.escape(name)}:\s*([^;,\n]+)",
        document,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def filter_score_records(records, question):
    """Filter score records by week, regular season, or Super Bowl."""
    q = question.lower()

    week_match = re.search(r"\bweek\s*(\d{1,2})\b", q)
    requested_week = int(week_match.group(1)) if week_match else None

    regular_season_only = bool(
        re.search(r"\bregular[\s-]+season\b", q)
    )
    super_bowl_only = "super bowl" in q or "superbowl" in q
    hall_of_fame_only = "hall of fame" in q

    filtered = []

    for record in records:
        document = record["document"]
        week_value = score_field(document, "Week")

        if not week_value:
            continue

        if hall_of_fame_only and week_value.strip().upper() != "HALL OF FAME":
            continue

        numeric_week = re.fullmatch(
            r"\s*(?:week\s*)?(\d{1,2})\s*",
            week_value,
            re.IGNORECASE,
        )

        if regular_season_only and not numeric_week:
            continue

        if requested_week is not None:
            if not numeric_week:
                continue

            if int(numeric_week.group(1)) != requested_week:
                continue

        if super_bowl_only:
            game_type = (
                score_field(document, "GameType")
                or score_field(document, "game_type")
                or score_field(document, "SeasonType")
                or score_field(document, "season_type")
                or ""
            )

            is_super_bowl = (
                bool(
                    re.search(
                        r"\bsuper\s*bowl\b",
                        document,
                        re.IGNORECASE,
                    )
                )
                or game_type.strip().upper()
                in {"SB", "SUPER BOWL", "SUPERBOWL"}
                or bool(
                    numeric_week
                    and int(numeric_week.group(1)) == 22
                )
            )

            if not is_super_bowl:
                continue

        filtered.append(record)

    return filtered


def get_all_records(where, batch_size=500):
    """Retrieve matching records in batches to avoid oversized queries."""
    records = []
    offset = 0

    while True:
        result = collection.get(
            where=where,
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        ids = result["ids"]
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []

        for index, record_id in enumerate(ids):
            records.append(
                {
                    "id": record_id,
                    "document": documents[index],
                    "metadata": metadatas[index] or {},
                    "distance": None,
                }
            )

        received = len(ids)

        if received < batch_size:
            break

        offset += received

    return records


def retrieve_hall_of_fame_game(where):
    """
    Retrieve Hall of Fame Game play records in source order.

    Records are fetched in batches, filtered to the specific game,
    and sorted by row_number. The database is read-only here.
    """
    game_records = []
    offset = 0
    batch_size = 500

    while True:
        page = collection.get(
            where=where,
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        page_ids = page["ids"]
        page_documents = page["documents"] or []
        page_metadatas = page["metadatas"] or []

        for index, record_id in enumerate(page_ids):
            document = page_documents[index]
            metadata = page_metadatas[index] or {}

            week = str(metadata.get("week", "")).upper()

            # Some indexed records store the week in the document
            # rather than in metadata.
            is_hall_of_fame = (
                week == "HALL OF FAME"
                or "Week: HALL OF FAME." in document
            )

            if (
                metadata.get("record_type") == "play"
                and is_hall_of_fame
            ):
                game_records.append(
                    {
                        "id": record_id,
                        "document": document,
                        "metadata": metadata,
                        "distance": None,
                    }
                )

        received = len(page_ids)

        if received < batch_size:
            break

        offset += received

    def chronological_key(record):
        metadata = record["metadata"]

        row_number = metadata.get("row_number")

        if row_number is None:
            row_number = record_number(record["id"])

        return (
            int(metadata.get("season", 0)),
            int(row_number),
        )

    game_records.sort(key=chronological_key)

    return game_records


def select_key_game_records(records, limit=28):
    """Keep representative chronological events for long game play-by-play."""
    if len(records) <= limit:
        return records

    important_terms = (
        "touchdown", "field goal", "extra point", "intercept", "fumble",
        "recovered", "safety", "two-point", "kickoff", "kicks", "kneel",
    )
    selected = {}
    # Always retain the opening and final indexed plays.
    selected[records[0]["id"]] = records[0]
    selected[records[-1]["id"]] = records[-1]

    for record in records:
        text = record["document"].lower()
        if any(term in text for term in important_terms):
            selected[record["id"]] = record

    ordered = sorted(
        selected.values(),
        key=lambda record: (
            int(record["metadata"].get("season", 0)),
            int(record["metadata"].get("row_number") or record_number(record["id"])),
        ),
    )
    if len(ordered) <= limit:
        return ordered

    # Preserve the start/end and sample the important events evenly if needed.
    indices = sorted({round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)})
    return [ordered[i] for i in indices]


def retrieve_plays(
    question,
    season=None,
    top_k=5,
    scoring_only=False,
    record_type="All",
):
    """Retrieve relevant NFL records for a question."""
    if collection.count() == 0:
        return []

    season, seasons, scoring_only, record_type = (
        infer_question_settings(
            question,
            season,
            scoring_only,
            record_type,
        )
    )

    where = build_where_filter(
        season=season,
        seasons=seasons,
        scoring_only=scoring_only,
        record_type=record_type,
    )

    q = question.lower()

    is_score_question = record_type == "score"
    is_kickoff_question = (
        "kickoff" in q or "kick off" in q
    )
    is_hall_of_fame = "hall of fame" in q

    # Broad Hall of Fame Game questions need the entire game sequence,
    # not a semantic shortlist that can mix unrelated games.
    #
    # Require a season to avoid scanning the entire multi-season database.
    # Specific kickoff questions continue through the targeted path below.
    if (
        is_hall_of_fame
        and not is_kickoff_question
        and not is_score_question
        and (season is not None or seasons)
    ):
        game_records = retrieve_hall_of_fame_game(where)

        if game_records:
            return game_records

    # Score questions use score records directly.
    if is_score_question:
        records = get_all_records(where)
        records = filter_score_records(records, question)

        return sorted(
            records,
            key=lambda record: (
                int(record["metadata"].get("season", 0)),
                record_number(record["id"]),
            ),
        )

    # Normal semantic retrieval with active metadata filters.
    total = collection.count()
    candidate_k = min(max(top_k * 10, 50), total)

    result = collection.query(
        query_texts=[question],
        n_results=candidate_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    candidates = {}

    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    ids = result["ids"][0] if result["ids"] else []
    distances = result["distances"][0] if result["distances"] else []

    for index, document in enumerate(documents):
        candidates[ids[index]] = {
            "id": ids[index],
            "document": document,
            "metadata": metadatas[index] or {},
            "distance": (
                distances[index]
                if index < len(distances)
                else None
            ),
        }

    # Supplement semantic retrieval with targeted event searches.
    search_terms = []

    if is_hall_of_fame:
        search_terms.append("HALL OF FAME")

    if is_kickoff_question:
        search_terms.extend(["kicks", "kickoff", "kick off"])

    if scoring_only:
        search_terms.extend(["TOUCHDOWN", "FIELD GOAL"])

    for term in dict.fromkeys(search_terms):
        try:
            kwargs = {
                "where_document": {"$contains": term},
                "limit": 1000,
                "include": ["documents", "metadatas"],
            }

            if where is not None:
                kwargs["where"] = where

            extra = collection.get(**kwargs)

            extra_documents = extra["documents"] or []
            extra_metadatas = extra["metadatas"] or []

            for index, document in enumerate(extra_documents):
                record_id = extra["ids"][index]
                metadata = extra_metadatas[index] or {}

                if record_id not in candidates:
                    candidates[record_id] = {
                        "id": record_id,
                        "document": document,
                        "metadata": metadata,
                        "distance": None,
                    }

        except Exception as exc:
            print(
                f"Targeted retrieval warning for {term!r}: {exc}"
            )

    question_terms = set(re.findall(r"[a-z0-9]+", q))

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "what",
        "which", "who", "when", "where", "how", "did", "do",
        "does", "me", "show", "tell", "about", "from", "in",
        "of", "and", "or", "to", "for", "played", "play",
        "game", "games", "season", "please", "on", "opening",
        "happened", "during", "their",
    }

    question_terms -= stopwords

    def score(record):
        text = record["document"].lower()
        metadata = record["metadata"]
        text_terms = set(re.findall(r"[a-z0-9]+", text))

        value = 2.0 * len(question_terms & text_terms)

        if is_hall_of_fame and "hall of fame" in text:
            value += 15

        if is_kickoff_question:
            if any(
                term in text
                for term in ("kicks", "kickoff", "kick off")
            ):
                value += 12

            if str(metadata.get("week", "")).upper() == "HALL OF FAME":
                value -= record_number(record["id"]) * 0.01

        if scoring_only and (
            "touchdown" in text or "field goal" in text
        ):
            value += 8

        distance = record.get("distance")

        if distance is not None:
            value += max(0.0, 2.0 - float(distance))

        return value

    # Prioritize the earliest matching opening Hall of Fame kickoff.
    if (
        is_kickoff_question
        and "opening" in q
        and is_hall_of_fame
    ):
        kickoff_records = [
            record
            for record in candidates.values()
            if (
                str(record["metadata"].get("week", "")).upper()
                == "HALL OF FAME"
                or "Week: HALL OF FAME." in record["document"]
            )
            and any(
                term in record["document"].lower()
                for term in ("kicks", "kickoff", "kick off")
            )
        ]

        if kickoff_records:
            kickoff_records.sort(
                key=lambda record: record_number(record["id"])
            )

            return kickoff_records[:top_k]

    ranked = sorted(
        candidates.values(),
        key=score,
        reverse=True,
    )

    return ranked[:top_k]


def format_score_records(records, include_winner=False):
    """Format final score records and optionally state the winner."""
    lines = []

    for record in records:
        document = record["document"]

        status = score_field(document, "GameStatus")

        if status and status.upper() != "FINAL":
            continue

        away = score_field(document, "AwayTeam")
        away_score = score_field(document, "AwayScore")
        home = score_field(document, "HomeTeam")
        home_score = score_field(document, "HomeScore")
        week = score_field(document, "Week")
        date = score_field(document, "GameDate")

        if not all((away, away_score, home, home_score)):
            continue

        try:
            away_points = float(away_score)
            home_points = float(home_score)
        except ValueError:
            continue

        details = (
            f"{team_label(away)} {away_points:g} – "
            f"{team_label(home)} {home_points:g}"
        )

        if include_winner:
            if away_points > home_points:
                details += f" | Winner: {away}"
            elif home_points > away_points:
                details += f" | Winner: {home}"
            else:
                details += " | Result: Tie"

        if week:
            details += (
                f" | Week {week}"
                if week.isdigit()
                else f" | {week}"
            )

        if date:
            details += f" | {date}"

        lines.append(details)

    return lines


def ask_question(
    question,
    season=None,
    scoring_only=False,
    record_type="All",
    top_k=5,
):
    """Answer an NFL question using retrieved records as evidence."""
    (
        effective_season,
        effective_seasons,
        effective_scoring,
        effective_type,
    ) = infer_question_settings(
        question,
        season,
        scoring_only,
        record_type,
    )

    records = retrieve_plays(
        question=question,
        season=effective_season,
        top_k=top_k,
        scoring_only=effective_scoring,
        record_type=effective_type,
    )

    if not records:
        return {
            "answer": (
                "No matching records were found for those filters. "
                "For Super Bowl questions, this code checks an explicit "
                "Super Bowl label or schedule Week 22 in the score record."
            ),
            "sources": [],
        }

    # Handle score-list requests directly from the score data.
    if effective_type == "score":
        q = question.lower()

        include_winner = (
            "super bowl" in q
            or "superbowl" in q
            or "winning team" in q
            or "who won" in q
            or "winner" in q
        )

        lines = format_score_records(
            records,
            include_winner=include_winner,
        )

        if lines:
            if effective_seasons:
                season_label = " and ".join(str(year) for year in effective_seasons)
            elif effective_season is not None:
                season_label = str(effective_season)
            else:
                season_label = "selected"

            if "hall of fame" in q and len(records) == 1:
                record = records[0]
                away = score_field(record["document"], "AwayTeam")
                away_score = score_field(record["document"], "AwayScore")
                home = score_field(record["document"], "HomeTeam")
                home_score = score_field(record["document"], "HomeScore")
                date = score_field(record["document"], "GameDate")
                try:
                    away_points = float(away_score)
                    home_points = float(home_score)
                except (TypeError, ValueError):
                    away_points = home_points = None

                if all((away, home, away_points is not None, home_points is not None)):
                    answer = (
                        f"The {effective_season or score_field(record['document'], 'Season') or ''} "
                        f"Hall of Fame Game was played between {team_label(away)} and "
                        f"{team_label(home)}. Final score: {team_label(away)} {away_points:g}, "
                        f"{team_label(home)} {home_points:g}."
                    )
                    if away_points > home_points:
                        answer += f" {team_label(away)} won."
                    elif home_points > away_points:
                        answer += f" {team_label(home)} won."
                    else:
                        answer += " The game ended in a tie."
                    if date:
                        answer += f" The game date in the dataset is {date}."
                else:
                    answer = "Hall of Fame Game score records were found, but the team or score fields are incomplete.\n\n" + "\n".join(lines)
            else:
                answer = (
                    f"Final game scores found for {season_label} "
                    f"({len(lines)} games):\n\n"
                    + "\n".join(f"- {line}" for line in lines)
                )
        else:
            answer = (
                "Score records were retrieved, but no final score records "
                "matched the requested filters. Check whether the dataset "
                "labels the Super Bowl as Week 22 or includes a Super Bowl "
                "game-type field."
            )

        return {
            "answer": answer,
            "sources": records,
        }

    # Keep all retrieved source IDs with their evidence.
    evidence = "\n\n".join(
        f"[Source ID: {record['id']}]\n{record['document']}"
        for record in records
    )

    is_full_hall_of_fame_game = (
        "hall of fame" in question.lower()
        and any(
            str(record["metadata"].get("week", "")).upper() == "HALL OF FAME"
            or "Week: HALL OF FAME." in record["document"]
            for record in records
        )
        and len(records) > top_k
    )

    # For long Hall of Fame play-by-play, use compact key-event evidence instead
    # of sending every indexed play to the LLM or displaying 176 source rows.
    # The score table is authoritative for the final score; the play log may
    # contain omissions or contradictory scoring events.
    source_records = records
    if is_full_hall_of_fame_game:
        key_records = select_key_game_records(records)
        score_where = build_where_filter(
            season=effective_season,
            seasons=effective_seasons,
            record_type="score",
        )
        try:
            score_records = filter_score_records(
                get_all_records(score_where), "Hall of Fame Game"
            )
        except Exception as exc:
            print(f"Hall of Fame score lookup warning: {exc}")
            score_records = []

        # Add score-table evidence to the prompt and the displayed source list.
        prompt_records = key_records + [r for r in score_records if r["id"] not in {x["id"] for x in key_records}]
        source_records = prompt_records
        evidence_records = prompt_records
    else:
        evidence_records = records

    evidence = "\n\n".join(
        f"[Source ID: {record['id']}]\n{record['document']}"
        for record in evidence_records
    )

    if is_full_hall_of_fame_game:
        task_instructions = """
The evidence contains selected chronological play-by-play events and, when
available, the game's score-table record. Summarize only the events supported
by the evidence. Pay attention to scoring plays, turnovers, and the ending.
Do not calculate an official final score from play-by-play if the score-table
record is present. If play-by-play appears inconsistent with the score table,
state that the sources disagree and report the score-table result separately.
Do not claim every play is shown; the evidence is a compact event selection.
"""
    else:
        task_instructions = """
Answer the user's question directly and concisely.
Use only the supplied evidence. Do not infer the final result of a game
from a few individual plays. If the evidence is insufficient to answer
part of the question, say what is missing.
"""

    prompt = f"""
You are an accurate NFL data assistant specializing in play-by-play analysis.

RULES:
1. Answer every part of the user's question.
2. Use only the evidence provided below.
3. Never invent events, scores, statistics, dates, or player actions.
4. When asked about a game, describe supported events chronologically.
5. Connect related plays into a coherent summary.
6. Cite relevant source IDs using the format [Source ID: 2010_16].
7. Clearly distinguish confirmed facts from information the evidence
   does not establish.
8. Do not claim the evidence covers a complete game unless it does.
9. For an opening-kickoff question, identify the kickoff record itself.
10. Keep the answer clear and concise.

ADDITIONAL INSTRUCTIONS:
{task_instructions}

EVIDENCE:
{evidence}

USER QUESTION:
{question}

Write the answer now, following all rules above.
"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": 160
        },
    )

    return {
        "answer": response["message"]["content"],
        "sources": source_records,
    }
