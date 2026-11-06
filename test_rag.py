"""Unit tests for the routing and exact-answer logic in src/rag.py.

They run on a tiny in-memory database with a fake embedding, so they need
neither Ollama, the downloaded embedding model nor the real chroma_db.

Usage:
    python -m unittest test_rag -v
"""

import sys
import unittest
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import rag  # noqa: E402


class FakeEmbedding:
    """Constant vectors: retrieval order is irrelevant, only filters matter."""

    def __call__(self, input):
        return [[0.1, 0.2, 0.3] for _ in input]

    def name(self):
        return "fake"

    def embed_query(self, input):
        return self(input)

    def embed_documents(self, input):
        return self(input)


def play(season, week, date, home, away, possession, outcome, description):
    return (
        f"Record type: play. Season: {season}. Week: {week}. Date: {date}. "
        f"Home team: {home}. Away team: {away}. Quarter: 1st Quarter. "
        f"Possession: {possession}. Scoring play: 1. Outcome: {outcome}. "
        f"Description: {description}"
    )


PLAYS = [
    play(2020, "HALL OF FAME", "August 6th", "PIT", "DAL",
         "Dallas Cowboys", "Touchdown", "— A.One for 5 yards, TOUCHDOWN."),
    play(2020, "WEEK 1", "September 10th", "KC", "HOU",
         "Kansas City Chiefs", "Touchdown", "— P.Mahomes pass to T.Kelce, TOUCHDOWN."),
    play(2020, "WEEK 2", "September 20th", "LAC", "KC",
         "Kansas City Chiefs", "Field Goal", "— H.Butker 40 yard field goal is GOOD."),
    play(2020, "WEEK 3", "September 27th", "BAL", "KC",
         "Kansas City Chiefs", "Touchdown", "— C.Edwards-Helaire run, TOUCHDOWN."),
    play(2020, "WEEK 3", "September 27th", "DEN", "PIT",
         "Denver Broncos", "Interception", "— Pass INTERCEPTED."),
    play(2020, "WILD CARD", "January 9th", "KC", "CLE",
         "Kansas City Chiefs", "Touchdown", "— T.Hill 20 yard pass, TOUCHDOWN."),
    play(2019, "WEEK 1", "September 8th", "DEN", "OAK",
         "Denver Broncos", "Touchdown", "— P.Lindsay run, TOUCHDOWN."),
]


def make_collection():
    collection = chromadb.EphemeralClient().get_or_create_collection(
        "test_nfl_rag", embedding_function=FakeEmbedding()
    )
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)

    collection.add(
        ids=[f"test_{i}" for i in range(len(PLAYS))],
        documents=PLAYS,
        metadatas=[
            {
                "record_type": "play",
                "season": 2019 if i == 6 else 2020,
                "is_scoring_play": 1,
                "source_file": "test.csv",
                "row_number": i,
            }
            for i in range(len(PLAYS))
        ],
    )
    return collection


class RoutingHelpers(unittest.TestCase):
    def test_week_groups(self):
        self.assertEqual(rag.week_group("WEEK 7"), "regular")
        self.assertEqual(rag.week_group("PRESEASON WEEK 2"), "preseason")
        self.assertEqual(rag.week_group("HALL OF FAME"), "preseason")
        self.assertEqual(rag.week_group("WILD CARD"), "postseason")
        self.assertEqual(rag.week_group("SUPER BOWL"), "postseason")

    def test_outcome_filter_is_exact(self):
        touchdown = rag.infer_outcome_filter("Show me touchdown plays from 2020")
        self.assertEqual(touchdown, {"$contains": "Outcome: Touchdown."})

        field_goal = rag.infer_outcome_filter("field goals in 2015")
        self.assertEqual(field_goal, {"$contains": "Outcome: Field Goal."})

        self.assertIsNone(rag.infer_outcome_filter("Who won the game?"))

    def test_team_aliases(self):
        teams, _ = rag.find_teams("how many touchdowns did kansas city have")
        self.assertEqual(teams, {"Kansas City Chiefs"})

        teams, _ = rag.find_teams("chiefs and broncos")
        self.assertEqual(len(teams), 2)

    def test_chronological_order_uses_row_number(self):
        early = {"id": "x_5", "metadata": {"season": 2020, "row_number": 5}}
        late = {"id": "x_90", "metadata": {"season": 2020, "row_number": 90}}
        self.assertLess(rag.chronological_key(early), rag.chronological_key(late))


class ExactAnswers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rag._collection = make_collection()

    def ask(self, question):
        return rag.ask_question(question)

    def test_team_touchdown_count(self):
        result = self.ask("How many touchdowns did Kansas City have in 2020?")
        self.assertIn("**3**", result["answer"])
        self.assertIn("regular season 2", result["answer"])
        self.assertIn("postseason 1", result["answer"])

    def test_season_filter_applies(self):
        result = self.ask("How many touchdowns in 2019?")
        self.assertIn("**1**", result["answer"])

    def test_field_goal_count_excludes_touchdowns(self):
        result = self.ask("How many field goals did Kansas City have in 2020?")
        self.assertIn("**1**", result["answer"])

    def test_player_count_is_not_answered_as_exact(self):
        # A player name is more specific than a plain count, so the exact
        # route must decline and let the sampling path handle it.
        result = rag.answer_count_question(
            "How many touchdowns did Patrick Mahomes have in 2020?",
            2020, None, "All", 5,
        )
        self.assertIsNone(result)

    def test_first_touchdown_is_earliest_in_game_order(self):
        result = self.ask("Who scored the first touchdown in 2020?")
        self.assertIn("A.One", result["answer"])
        self.assertIn("P.Mahomes", result["answer"])  # first regular-season one

    def test_last_touchdown(self):
        result = self.ask("Who scored the last touchdown in 2020?")
        self.assertIn("T.Hill", result["answer"])

    def test_first_touchdown_for_team(self):
        result = self.ask("Who scored the first touchdown for Denver in 2019?")
        self.assertIn("P.Lindsay", result["answer"])

    def test_first_last_needs_a_season(self):
        result = rag.answer_first_last_question(
            "Who scored the first touchdown?", None, None, "All"
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
