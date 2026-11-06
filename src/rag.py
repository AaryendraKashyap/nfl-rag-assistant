from pathlib import Path
import re

import chromadb
import ollama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ---- Settings (must match what src/ingest.py used to build the database) ----
COLLECTION_NAME = "nfl_plays"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"
FIRST_SEASON = 2010
LAST_SEASON = 2026

# temperature 0 keeps answers repeatable; num_ctx stops Ollama silently
# truncating the prompt; num_predict is high enough to finish a sentence.
LLM_OPTIONS = {"temperature": 0, "num_ctx": 8192, "num_predict": 512}


TEAM_NAMES = {
    "AZ": "Arizona Cardinals",
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

_collection = None


class DatabaseNotReady(RuntimeError):
    """Raised when the Chroma database has not been built yet."""


def get_collection():
    """Open the existing collection lazily (no work happens at import time)."""
    global _collection

    if _collection is None:
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=str(DB_DIR))

        try:
            _collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_function,
            )
        except Exception as exc:
            raise DatabaseNotReady(
                f"Collection '{COLLECTION_NAME}' was not found in {DB_DIR}. "
                "Run `python src/ingest.py` first."
            ) from exc

    return _collection


def record_week(record):
    """Week label of a record. Uses metadata if present, else the document text.

    The existing database stores the week only inside the document text
    ("Week: HALL OF FAME."), not in metadata, so the fallback is required.
    """
    week = (record.get("metadata") or {}).get("week")
    if week:
        return str(week).strip().upper()

    match = re.search(
        r"\bWeek:\s*(.+?)(?:\.\s+Date:|;\s)",
        record.get("document", ""),
    )
    return match.group(1).strip().upper() if match else ""


def chronological_key(record):
    """Sort key giving source order: (season, row number in the CSV)."""
    metadata = record.get("metadata") or {}
    row_number = metadata.get("row_number")

    if row_number is None:
        row_number = record_number(record["id"])

    return int(metadata.get("season", 0)), int(row_number)


# Outcome phrases as stored in the documents ("Outcome: Touchdown. Description: ...").
# Filtering on these server-side is exact, unlike "is_scoring_play", which is 1
# for touchdowns, field goals AND extra points.
OUTCOME_PATTERNS = (
    (r"\btouchdowns?\b|\btds?\b", "Outcome: Touchdown."),
    (r"\bfield goals?\b", "Outcome: Field Goal."),
    (r"\bextra points?\b", "Outcome: Extra Point."),
    (r"\binterceptions?\b|\bpick[- ]?six(?:es)?\b", "Outcome: Interception."),
    (r"\bfumbles?\b", "Outcome: Fumble."),
    (r"\bpunts?\b", "Outcome: Punt."),
    (r"\bsacks?\b", "Yard Sack."),
    (r"\bkick[- ]?offs?\b", "Outcome: Kickoff"),
)

AGGREGATE_PATTERN = re.compile(
    r"\b(how many|most|least|fewest|total|totals|average|avg|leader|leaders|"
    r"leading|ranked|ranking|rank|highest|lowest|longest|shortest|best|worst|"
    r"top \d+|per game|all[- ]time)\b"
)

AGGREGATE_WARNING = (
    "This assistant answers from the few records it retrieves, so it cannot "
    "reliably count, rank or total across a season. Treat any numbers as "
    "based on the retrieved sample only."
)


def infer_outcome_filter(question):
    """Return a Chroma where_document filter for the play type asked about."""
    q = question.lower()
    phrases = [
        phrase for pattern, phrase in OUTCOME_PATTERNS
        if re.search(pattern, q)
    ]

    if not phrases:
        return None

    if len(phrases) == 1:
        return {"$contains": phrases[0]}

    return {"$or": [{"$contains": phrase} for phrase in phrases]}


def is_aggregate_question(question):
    """True for count/rank/total questions that retrieval cannot answer."""
    return bool(AGGREGATE_PATTERN.search(question.lower()))


def combine_document_filters(*filters):
    """AND together any non-empty where_document filters."""
    active = [f for f in filters if f]

    if not active:
        return None

    if len(active) == 1:
        return active[0]

    return {"$and": active}


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
            FIRST_SEASON <= start_year <= LAST_SEASON
            and FIRST_SEASON <= end_year <= LAST_SEASON
            and start_year <= end_year
        ):
            seasons = list(range(start_year, end_year + 1))
            season = None
    else:
        years = [
            int(year)
            for year in re.findall(r"\b(20\d{2})\b", q)
            if FIRST_SEASON <= int(year) <= LAST_SEASON
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

    # "touchdown" is no longer mapped to is_scoring_play: that flag is also 1
    # for field goals and extra points. Touchdowns are matched exactly through
    # infer_outcome_filter() instead.
    if record_type != "score" and "scoring plays" in q:
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


def get_all_records(where, where_document=None, batch_size=500):
    """Retrieve every matching record, paging through Chroma."""
    collection = get_collection()
    records = []
    offset = 0

    while True:
        kwargs = {
            "limit": batch_size,
            "offset": offset,
            "include": ["documents", "metadatas"],
        }

        if where is not None:
            kwargs["where"] = where

        if where_document is not None:
            kwargs["where_document"] = where_document

        result = collection.get(**kwargs)

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


HALL_OF_FAME_MARKER = {"$contains": "Week: HALL OF FAME."}


def retrieve_hall_of_fame_game(play_where, outcome_filter=None):
    """
    Retrieve Hall of Fame Game play records in source order.

    The week is matched inside Chroma (where_document) instead of paging
    through every play of the season and filtering in Python. The existing
    documents contain "Week: HALL OF FAME." so no re-ingest is needed.
    """
    records = get_all_records(
        play_where,
        combine_document_filters(HALL_OF_FAME_MARKER, outcome_filter),
    )

    return sorted(records, key=chronological_key)


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


def query_candidates(question, where, where_document, n_results):
    """Semantic search with optional metadata and document filters."""
    kwargs = {
        "query_texts": [question],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }

    if where is not None:
        kwargs["where"] = where

    if where_document is not None:
        kwargs["where_document"] = where_document

    result = get_collection().query(**kwargs)

    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    ids = result["ids"][0] if result["ids"] else []
    distances = result["distances"][0] if result["distances"] else []

    candidates = {}

    for index, document in enumerate(documents):
        candidates[ids[index]] = {
            "id": ids[index],
            "document": document,
            "metadata": metadatas[index] or {},
            "distance": (
                distances[index] if index < len(distances) else None
            ),
        }

    return candidates


QUESTION_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what",
    "which", "who", "when", "where", "how", "did", "do",
    "does", "me", "show", "tell", "about", "from", "in",
    "of", "and", "or", "to", "for", "played", "play",
    "game", "games", "season", "please", "on", "opening",
    "happened", "during", "their",
}


def retrieve_plays(
    question,
    season=None,
    top_k=5,
    scoring_only=False,
    record_type="All",
):
    """Retrieve relevant NFL records for a question."""
    total = get_collection().count()

    if total == 0:
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
    is_kickoff_question = "kickoff" in q or "kick off" in q
    is_hall_of_fame = "hall of fame" in q
    outcome_filter = infer_outcome_filter(question)

    # Broad Hall of Fame Game questions need the entire game sequence,
    # not a semantic shortlist that can mix unrelated games.
    # Require a season so the lookup stays small.
    if (
        is_hall_of_fame
        and not is_kickoff_question
        and not is_score_question
        and (season is not None or seasons)
    ):
        play_where = build_where_filter(
            season=season,
            seasons=seasons,
            scoring_only=scoring_only,
            record_type="play",
        )
        game_records = retrieve_hall_of_fame_game(
            play_where, outcome_filter
        )

        if game_records:
            return game_records

    # Score questions use score records directly.
    if is_score_question:
        records = get_all_records(
            where,
            {"$contains": "Week: HALL OF FAME"} if is_hall_of_fame else None,
        )
        records = filter_score_records(records, question)

        return sorted(records, key=chronological_key)

    hall_of_fame_filter = HALL_OF_FAME_MARKER if is_hall_of_fame else None

    # "Opening kickoff of the Hall of Fame game": deterministic, not semantic.
    if is_kickoff_question and "opening" in q and is_hall_of_fame:
        play_where = build_where_filter(
            season=season,
            seasons=seasons,
            scoring_only=False,
            record_type="play",
        )
        kickoffs = get_all_records(
            play_where,
            combine_document_filters(
                hall_of_fame_filter, outcome_filter
            ),
        )

        if kickoffs:
            return sorted(kickoffs, key=chronological_key)[:top_k]

    # Normal semantic retrieval, narrowed server-side by season, record type,
    # and (when the question names one) the play outcome.
    candidate_k = min(max(top_k * 10, 50), total)

    candidates = query_candidates(
        question,
        where,
        combine_document_filters(hall_of_fame_filter, outcome_filter),
        candidate_k,
    )

    # The outcome word may not be a hard requirement, so retry without it.
    if not candidates and outcome_filter:
        candidates = query_candidates(
            question, where, hall_of_fame_filter, candidate_k
        )

    question_terms = set(re.findall(r"[a-z0-9]+", q)) - QUESTION_STOPWORDS

    def score(record):
        text_terms = set(re.findall(r"[a-z0-9]+", record["document"].lower()))

        value = 2.0 * len(question_terms & text_terms)

        distance = record.get("distance")

        if distance is not None:
            value += max(0.0, 2.0 - float(distance))

        return value

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


DOCUMENT_PATTERN = re.compile(
    r"Season: (?P<season>\d+)\. Week: (?P<week>.*?)\. Date: (?P<date>.*?)\. "
    r"Home team: (?P<home>.*?)\. Away team: (?P<away>.*?)\. "
    r"Quarter: (?P<quarter>.*?)\. Possession: (?P<possession>.*?)\. "
    r"Scoring play: \d\. Outcome: (?P<outcome>.*?)\. Description: (?P<description>.*)",
    re.DOTALL,
)


def compact_document(document):
    """Shorten a stored play document for the LLM prompt.

    The stored text repeats labels ("Home team: ...") on every play. Dropping
    them saves roughly 40% of the prompt tokens. Anything that does not match
    the play format (score records, for example) is passed through unchanged.
    """
    match = DOCUMENT_PATTERN.search(document)

    if not match:
        return document

    f = match.groupdict()

    return (
        f"{f['season']} {f['week']} | {f['away']} at {f['home']} | "
        f"{f['quarter']} | possession: {f['possession']} | "
        f"{f['outcome']} | {f['description']}"
    )


def format_evidence(records):
    return "\n\n".join(
        f"[Source ID: {record['id']}]\n{compact_document(record['document'])}"
        for record in records
    )


SYSTEM_PROMPT = """You are an accurate NFL data assistant specializing in play-by-play analysis.

RULES:
1. Answer every part of the user's question.
2. Use only the evidence provided by the user message.
3. Never invent events, scores, statistics, dates, or player actions.
4. When asked about a game, describe supported events chronologically.
5. Connect related plays into a coherent summary.
6. Cite relevant source IDs using the format [Source ID: 2010_16].
7. Clearly distinguish confirmed facts from information the evidence
   does not establish.
8. Do not claim the evidence covers a complete game unless it does.
9. For an opening-kickoff question, identify the kickoff record itself.
10. If the question asks for a count, total, ranking or "most/longest/best"
    across many plays, say the evidence is only a small sample and do not
    state a definitive number.
11. Keep the answer clear and concise."""


def call_llm(user_prompt):
    """Ask Llama 3 (via Ollama). Returns (text, error_message_or_None)."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options=LLM_OPTIONS,
        )
        return response["message"]["content"], None
    except Exception as exc:
        return None, (
            f"Could not get an answer from {LLM_MODEL} through Ollama "
            f"({type(exc).__name__}: {exc}). Make sure Ollama is running "
            f"(`ollama serve`) and the model is installed "
            f"(`ollama pull {LLM_MODEL}`). The retrieved records are still "
            "shown below."
        )


# ---------------------------------------------------------------------------
# Exact counts ("How many touchdowns did Kansas City score in 2020?")
#
# Counting by retrieval is impossible (it only sees a few records), so these
# questions are answered by counting matching records inside ChromaDB with
# metadata/document filters. No LLM and no re-ingest are involved.
# ---------------------------------------------------------------------------

# TeamWithPossession is stored with the current franchise name in every season.
POSSESSION_TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
    "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers",
    "Los Angeles Rams", "Miami Dolphins", "Minnesota Vikings",
    "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers",
    "San Francisco 49ers", "Seattle Seahawks", "Tampa Bay Buccaneers",
    "Tennessee Titans", "Washington Commanders",
]

AMBIGUOUS_CITIES = {"new york", "los angeles"}

EXTRA_TEAM_ALIASES = {
    "niners": "San Francisco 49ers",
    "bucs": "Tampa Bay Buccaneers",
    "oakland": "Las Vegas Raiders",
    "san diego": "Los Angeles Chargers",
    "st louis": "Los Angeles Rams",
    "redskins": "Washington Commanders",
    "jaguars": "Jacksonville Jaguars",
    "jags": "Jacksonville Jaguars",
}


def _build_team_aliases():
    aliases = {}

    for full in POSSESSION_TEAMS:
        parts = full.split()
        city = " ".join(parts[:-1]).lower()
        aliases[full.lower()] = full
        aliases[parts[-1].lower()] = full

        if city not in AMBIGUOUS_CITIES:
            aliases[city] = full

    aliases.update(EXTRA_TEAM_ALIASES)
    return aliases


TEAM_ALIASES = _build_team_aliases()

COUNT_INTENT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b")

# Words that may remain in a counting question once the team, years and play
# type are removed. Anything else (a player name, "per game", "by quarter"...)
# means the question is more specific than this feature can answer exactly.
COUNT_FILLER = {
    "how", "many", "number", "of", "count", "did", "do", "does", "the", "a",
    "an", "in", "were", "was", "are", "is", "there", "score", "scored",
    "scoring", "by", "during", "season", "seasons", "all", "for", "and", "or",
    "through", "thru", "to", "total", "have", "has", "had", "been", "happen",
    "happened", "occur", "occurred", "recorded", "regular", "playoff",
    "playoffs", "postseason", "preseason", "nfl", "kicked", "thrown", "threw",
    "throw", "from", "s", "their", "team", "overall", "league", "combined",
}
COUNT_FILLER_MADE = {"made", "make", "makes", "kick", "kicks"}  # kicks only

OUTCOME_LABELS = {
    "Outcome: Touchdown.": "touchdown plays",
    "Outcome: Field Goal.": "made field goals",
    "Outcome: Extra Point.": "made extra points",
    "Outcome: Interception.": "interceptions",
    "Outcome: Fumble.": "fumbles",
    "Outcome: Punt.": "punts",
    "Yard Sack.": "sacks",
    "Outcome: Kickoff": "kickoffs",
}


def find_teams(question_lower):
    """Return (set of full team names, question text with team words removed)."""
    found = set()
    remaining = question_lower

    for alias in sorted(TEAM_ALIASES, key=len, reverse=True):
        pattern = rf"\b{re.escape(alias)}\b"

        if re.search(pattern, remaining):
            found.add(TEAM_ALIASES[alias])
            remaining = re.sub(pattern, " ", remaining)

    return found, remaining


def week_group(week):
    """Preseason / regular season / postseason from a week label."""
    week = week.upper()

    if week.startswith("PRESEASON") or week == "HALL OF FAME":
        return "preseason"

    if re.fullmatch(r"WEEK \d+", week):
        return "regular"

    return "postseason"


def season_phrase(season, seasons):
    if seasons and len(seasons) > 1:
        years = sorted(seasons)
        if years == list(range(years[0], years[-1] + 1)):
            return f"in {years[0]}\u2013{years[-1]}"
        return "in " + " and ".join(str(y) for y in years)

    if seasons:
        return f"in {seasons[0]}"

    if season is not None:
        return f"in {season}"

    return f"across all seasons ({FIRST_SEASON}\u2013{LAST_SEASON})"


def answer_count_question(question, season, seasons, record_type, top_k):
    """Exact answer for "how many <play type> [team] [season]" questions.

    Returns a result dict, or None when the question is not a simple count
    (the normal retrieval path then handles it).
    """
    q = question.lower()

    if (
        not COUNT_INTENT.search(q)
        or record_type == "score"
        or "hall of fame" in q
    ):
        return None

    outcome_filter = infer_outcome_filter(question)

    if not outcome_filter or "$or" in outcome_filter:
        return None

    outcome_phrase = outcome_filter["$contains"]

    teams, remaining = find_teams(q)

    if len(teams) > 1:
        return None

    # Strip everything we understand; if words are left over, the question is
    # more specific than a plain count (a player, "per game", ...).
    for pattern, _phrase in OUTCOME_PATTERNS:
        remaining = re.sub(pattern, " ", remaining)

    remaining = re.sub(r"\b20\d{2}\b", " ", remaining)
    allowed = set(COUNT_FILLER)

    if outcome_phrase in ("Outcome: Field Goal.", "Outcome: Extra Point."):
        allowed |= COUNT_FILLER_MADE

    leftover = [w for w in re.findall(r"[a-z0-9]+", remaining) if w not in allowed]

    if leftover:
        return None

    team = next(iter(teams)) if teams else None

    where = build_where_filter(
        season=season, seasons=seasons, record_type="play"
    )
    where_document = combine_document_filters(
        outcome_filter,
        {"$contains": f"Possession: {team}."} if team else None,
    )

    records = get_all_records(where, where_document)
    records.sort(key=chronological_key)

    groups = {"regular": 0, "postseason": 0, "preseason": 0}

    for record in records:
        groups[week_group(record_week(record))] += 1

    total = len(records)

    scope = None
    if "regular season" in q:
        scope = "regular"
    elif "playoff" in q or "postseason" in q:
        scope = "postseason"
    elif "preseason" in q:
        scope = "preseason"

    headline = groups[scope] if scope else total
    scope_text = {
        "regular": "regular-season ",
        "postseason": "postseason ",
        "preseason": "preseason ",
        None: "",
    }[scope]

    label = OUTCOME_LABELS.get(outcome_phrase, "plays")
    subject = f" by the {team}" if team else ""

    answer = (
        f"**{headline:,}** {scope_text}{label}{subject} "
        f"{season_phrase(season, seasons)}.\n\n"
        f"Breakdown: regular season {groups['regular']:,}, "
        f"postseason {groups['postseason']:,}, "
        f"preseason/Hall of Fame {groups['preseason']:,} "
        f"(total {total:,}).\n\n"
        "Counted exactly from the database, not estimated by the model. "
        "Plays are matched on their recorded outcome and, for a team, on the "
        "team in possession; scores recorded under a different outcome "
        "(for example some defensive scores) are not included. "
        "The records below are examples."
    )

    return {
        "answer": answer,
        "sources": records[:top_k],
        "warning": None,
        "llm_error": False,
    }


FIRST_LAST_INTENT = re.compile(r"\b(first|earliest|opening|last|latest|final)\b")

FIRST_LAST_FILLER = COUNT_FILLER | {
    "who", "what", "which", "whose", "scored", "score", "first", "earliest",
    "opening", "last", "latest", "final", "game", "of", "on", "that", "scoring",
    "play", "plays", "scorer", "season", "year", "nfl", "was", "were", "is",
    "kicked", "made", "make", "get", "got", "gets", "recorded",
}


def answer_first_last_question(question, season, seasons, record_type):
    """Exact answer for "who scored the first touchdown in 2010?" questions.

    Sorts every matching play in source (chronological) order instead of
    trusting the few records a similarity search returns. Returns None when
    the question is not this simple shape.
    """
    q = question.lower()

    if (
        not FIRST_LAST_INTENT.search(q)
        or COUNT_INTENT.search(q)
        or record_type == "score"
        or "hall of fame" in q
        or not (season or seasons)
    ):
        return None

    outcome_filter = infer_outcome_filter(question)

    if not outcome_filter or "$or" in outcome_filter:
        return None

    teams, remaining = find_teams(q)

    if len(teams) > 1:
        return None

    for pattern, _phrase in OUTCOME_PATTERNS:
        remaining = re.sub(pattern, " ", remaining)

    remaining = re.sub(r"\b20\d{2}\b", " ", remaining)
    leftover = [
        w for w in re.findall(r"[a-z0-9]+", remaining) if w not in FIRST_LAST_FILLER
    ]

    if leftover:
        return None

    want_last = bool(re.search(r"\b(last|latest|final)\b", q))
    team = next(iter(teams)) if teams else None

    where = build_where_filter(season=season, seasons=seasons, record_type="play")
    where_document = combine_document_filters(
        outcome_filter,
        {"$contains": f"Possession: {team}."} if team else None,
    )

    records = get_all_records(where, where_document)
    records.sort(key=chronological_key)

    if not records:
        return None

    def pick(items):
        return (items[-1] if want_last else items[0]) if items else None

    overall = pick(records)
    regular = pick([r for r in records if week_group(record_week(r)) == "regular"])

    word = "last" if want_last else "first"
    label = OUTCOME_LABELS.get(outcome_filter["$contains"], "play").rstrip("s")
    subject = f" by the {team}" if team else ""

    def describe(record):
        doc = record["document"]
        desc = re.search(r"Description:\s*(.*)$", doc, re.S)
        week = record_week(record).title()
        quarter = re.search(r"Quarter:\s*(.*?)\.", doc)
        date = re.search(r"Date:\s*(.*?)\.", doc)
        game = re.search(r"Home team:\s*(.*?)\.\s*Away team:\s*(.*?)\.", doc)
        parts = [week]
        if date:
            parts.append(date.group(1))
        if game:
            parts.append(f"{game.group(2)} at {game.group(1)}")
        if quarter:
            parts.append(quarter.group(1))
        text = desc.group(1).strip() if desc else ""
        return f"{text} ({', '.join(parts)}; record `{record['id']}`)"

    answer = (
        f"The {word} {label}{subject} {season_phrase(season, seasons)} "
        f"(preseason and Hall of Fame included) was:\n\n{describe(overall)}"
    )

    if regular is not None and regular is not overall:
        answer += (
            f"\n\nThe {word} regular-season one was:\n\n{describe(regular)}"
        )

    answer += (
        "\n\nFound by sorting every matching play in the database in game "
        "order, not by the model. The scorer is the player named in the play "
        "description (players are stored as initials)."
    )

    sources = [overall] + ([regular] if regular is not None and regular is not overall else [])

    return {"answer": answer, "sources": sources, "warning": None, "llm_error": False}


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

    counted = answer_count_question(
        question,
        effective_season,
        effective_seasons,
        effective_type,
        top_k,
    )

    if counted is not None:
        return counted

    first_last = answer_first_last_question(
        question,
        effective_season,
        effective_seasons,
        effective_type,
    )

    if first_last is not None:
        return first_last

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
                "Try removing the season/record-type filters or rephrasing."
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
                "Score records were retrieved, but none of them is a "
                "FINAL game for the requested filters (the game may not "
                "have been played yet)."
            )

        return {
            "answer": answer,
            "sources": records,
        }

    warning = (
        AGGREGATE_WARNING if is_aggregate_question(question) else None
    )

    is_full_hall_of_fame_game = (
        "hall of fame" in question.lower()
        and any(record_week(record) == "HALL OF FAME" for record in records)
        and len(records) > top_k
    )

    # For long Hall of Fame play-by-play, use compact key-event evidence instead
    # of sending every indexed play to the LLM or displaying 176 source rows.
    # The score table is authoritative for the final score; the play log may
    # contain omissions or contradictory scoring events.
    source_records = records
    evidence_records = records

    if is_full_hall_of_fame_game:
        key_records = select_key_game_records(records)
        score_where = build_where_filter(
            season=effective_season,
            seasons=effective_seasons,
            record_type="score",
        )
        try:
            score_records = filter_score_records(
                get_all_records(
                    score_where, {"$contains": "Week: HALL OF FAME"}
                ),
                "Hall of Fame Game",
            )
        except Exception as exc:
            print(f"Hall of Fame score lookup warning: {exc}")
            score_records = []

        # Add score-table evidence to the prompt and the displayed source list.
        key_ids = {x["id"] for x in key_records}
        evidence_records = key_records + [
            r for r in score_records if r["id"] not in key_ids
        ]
        source_records = evidence_records

    if is_full_hall_of_fame_game:
        task_instructions = """
The evidence contains selected chronological play-by-play events and, when
available, the game's score-table record. Summarize only the events supported
by the evidence. Pay attention to scoring plays, turnovers, and the ending.
Do not calculate an official final score from play-by-play if the score-table
record is present. If play-by-play appears inconsistent with the score table,
state that the sources disagree and report the score-table result separately.
Do not claim every play is shown; the evidence is a compact event selection.
Write the summary as a bullet list, one event per line in order, starting with
the quarter. Do not describe drives, plays or details that are not in the
evidence, and do not add transitions such as "then drove down the field".
"""
    else:
        task_instructions = """
Answer the user's question directly and concisely.
Use only the supplied evidence. Do not infer the final result of a game
from a few individual plays. If the evidence is insufficient to answer
part of the question, say what is missing.
"""

    if warning:
        task_instructions += f"""
IMPORTANT: the evidence is only a SAMPLE of {len(evidence_records)} records,
not every play. Do NOT state totals, counts, rankings or "the most/longest"
as facts. Describe only what appears in the sample, and begin by saying
"In the {len(evidence_records)} retrieved records".
"""

    user_prompt = f"""ADDITIONAL INSTRUCTIONS:
{task_instructions}

EVIDENCE:
{format_evidence(evidence_records)}

USER QUESTION:
{question}

Write the answer now, following all rules above."""

    answer, error = call_llm(user_prompt)

    if answer is not None and warning:
        answer = (
            f"_Based on a sample of {len(evidence_records)} retrieved "
            f"records, not a full count._\n\n{answer}"
        )

    return {
        "answer": answer if answer is not None else error,
        "sources": source_records,
        "warning": warning,
        "llm_error": error is not None,
    }
