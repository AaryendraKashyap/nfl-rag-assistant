# NFL RAG Assistant

A local question-answering assistant over NFL play-by-play and game-score data
(2010–2026). It combines exact database filters, deterministic counting and a
local Llama 3 model, and shows the records behind every answer.

Everything runs locally: ChromaDB for storage, `all-MiniLM-L6-v2` for
embeddings, Llama 3 through Ollama for generation, and Streamlit for the UI.

![App screenshot](screenshot.png)

## Why it is built this way

Retrieval-augmented generation alone fails on this data. A language model
can only see the handful of records retrieved, so it cannot count or rank
across a season. This project therefore uses three routes:

| Question type | Route | Language model used? |
|---|---|---|
| "How many touchdowns did Kansas City have in 2020?" | Exact count over all matching records in ChromaDB | No |
| "Who won the 2015 Super Bowl?", "final scores of week 1" | Read directly from the game-score records | No |
| "Show me interceptions in 2018", "summarise the Hall of Fame game" | Filtered semantic retrieval, then Llama 3 | Yes |

Counting questions get an exact number with a breakdown of regular season,
postseason and preseason. Questions that are too specific to count (for
example, a player name) fall back to retrieval and show a warning that the
answer is based on a sample.

## Setup

Requires Python 3.10 or newer and [Ollama](https://ollama.com).

```bash
ollama pull llama3
pip install -r requirements.txt
```

## Running it

1. **Put the data in `data/raw/`.** The season files are `<year>_plays.csv` and
   `2010-2026_scores.csv`. If you have the original archive:
   ```bash
   python extract_data.py --archive archive.zip
   ```
2. **Build the vector database** (one time, slow):
   ```bash
   python src/ingest.py
   ```
   Re-running is safe. Existing records are skipped, so you can add a new
   season later with `python src/ingest.py --seasons 2026`.
3. **Check the database:**
   ```bash
   python check_database.py
   ```
   It should report about 900,782 records.
4. **Start the app:**
   ```bash
   streamlit run app.py
   ```
5. **Reproduce the evaluation** (see below):
   ```bash
   python evaluate.py
   ```
6. **Run the tests** (no Ollama or database needed):
   ```bash
   python -m unittest test_rag -v
   ```

## Example questions

| Question | Expected behaviour |
|---|---|
| How many touchdowns did Kansas City have in 2020? | Exact count, 63 |
| Who won the 2015 Super Bowl? | Denver Broncos 24, Carolina Panthers 10, no LLM |
| Show me field goals from 2015 | Every source is `Outcome: Field Goal.` |
| What was the opening kickoff of the 2015 Hall of Fame game? | First source is `2015_0` |
| Summarise the 2015 Hall of Fame game | Chronological events, ending with the score table |
| Who scored the first touchdown in 2010? | Earliest touchdown (Hall of Fame game) and first regular-season one, no LLM |
| Who had the most touchdowns in 2020? | Warning: answer is from a sample only |

## Filters

Sidebar filters: **Season**, **Record type** (All, Play-by-play, Game scores),
**Scoring plays only**, and **Number of records to retrieve** (3–15).

The app also reads the question itself:
- **Years:** "in 2020", "2019 through 2021", or several years.
- **Play type:** touchdown, field goal, extra point, interception, fumble,
  punt, sack, kickoff. These match the stored `Outcome:` text exactly, so a
  "touchdown" query never returns field goals.
- **Score words:** "final score", "who won", "winner" send the question to the
  game-score records.
- **Hall of Fame:** the whole game is retrieved in order.

Team, player, quarter and week are not filters for play-by-play. They only
influence ranking, so those searches can be loose.

## Evaluation

`evaluate.py` checks the app against ground truth computed directly from the
CSV files. It reports:

1. **Exact-count accuracy:** counts for four teams × three play types, plus a
   check that a player-name question shows the sample warning instead of a
   number.
2. **Play-type precision:** the share of retrieved plays that have the asked
   outcome, compared with a plain semantic search over the same season
   (the baseline).
3. **Score accuracy:** Super Bowl winner and number of games in a week.
4. **Latency:** median time per question.

Results are written to `evaluation_results.md`. Run it with
`python evaluate.py --seasons 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026`.

### Results (all 17 seasons, 2010–2026)

| Metric | Result |
|---|---|
| Exact-count accuracy | 205/205 |
| Play-type precision, this system | 100% |
| Play-type precision, semantic baseline | 68% |
| Score lookup accuracy | 33/33 |
| Median latency, exact count | ~3.6 s |

The semantic baseline scored 68% on average across the 17 seasons. These figures come from running `evaluate.py` on the full database with the all-MiniLM-L6-v2 embeddings.

The exact-count, score and precision results do not depend on the embedding
model, because they use exact text filters and direct reads.

## Limitations

- **Players are stored as initials** (`P.Mahomes`). Full-name searches are weak.
- **Team codes change over time** (`OAK`→`LV`, `SD`→`LAC`, `STL`→`LAR`,
  `JAC`→`JAX`, `ARI`/`AZ`). Counts use the franchise name stored in each
  record's possession field.
- **Counts exclude some scores.** A touchdown recorded under another outcome
  is not counted, which is noted in each count answer.
- **"First/last" answers use only plays labelled `Touchdown`.** A score
  recorded under another outcome label is not considered.
- **Preseason and Hall of Fame games are included** in season totals; the
  breakdown shows them separately.
- **Years outside 2010–2026 are ignored.** A question about, say, 2030 is
  answered across all seasons instead of reporting that there is no data.
- **The 2026 season** contains only preseason data in the source.
- **Answer quality** from Llama 3 is not scored automatically. The
  evaluation covers retrieval and exact answers only.

## Project layout

```
app.py                  Streamlit interface
src/rag.py              Routing, filters, exact counts, prompt and Llama 3 call
src/ingest.py           CSV → ChromaDB (resumable, incremental)
extract_data.py         Extracts the season files from the archive
check_database.py       Record counts per season and type (read-only)
evaluate.py             Ground-truth evaluation and results report
```

