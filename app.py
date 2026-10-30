
import streamlit as st

from src.rag import ask_question, collection

st.set_page_config(
    page_title="NFL RAG Assistant",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 NFL RAG Assistant")
st.caption("Ask questions about NFL plays and game scores.")

# Display the database count without scanning all records.
total_records = collection.count()

if total_records == 0:
    st.warning(
        "The database is empty. Run `python src/ingest.py` first."
    )
    st.stop()

# Use known seasons instead of scanning 900,000+ records.
seasons = list(range(2026, 2009, -1))

with st.sidebar:
    st.header("Search filters")

    selected_season = st.selectbox(
        "Season",
        ["All seasons"] + seasons,
    )

    record_type = st.selectbox(
        "Record type",
        ["All", "play", "score"],
        format_func=lambda x: {
            "All": "All records",
            "play": "Play-by-play",
            "score": "Game scores",
        }[x],
    )

    scoring_only = st.checkbox("Scoring plays only")

    top_k = st.slider(
        "Number of records to retrieve",
        min_value=3,
        max_value=15,
        value=5,
    )

st.subheader("Ask an NFL question")

question = st.text_area(
    "Your question",
    placeholder="Example: Show me touchdown plays from the 2020 season.",
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif scoring_only and record_type == "score":
        st.warning(
            "Scoring-play filtering applies to play records, "
            "not game-score records. Select Play-by-play or All records."
        )
    else:
        season = (
            None
            if selected_season == "All seasons"
            else int(selected_season)
        )

        with st.spinner("Searching records and generating an answer..."):
            try:
                result = ask_question(
                    question=question.strip(),
                    season=season,
                    scoring_only=scoring_only,
                    record_type=record_type,
                    top_k=top_k,
                )

                st.subheader("Answer")
                st.write(result["answer"])

                with st.expander("Retrieved evidence", expanded=True):
                    if not result["sources"]:
                        st.info("No supporting records were retrieved.")
                    else:
                        for source in result["sources"]:
                            metadata = source["metadata"]

                            st.markdown(
                                f"**Source:** `{source['id']}` · "
                                f"**Season:** {metadata.get('season', 'Unknown')} · "
                                f"**Type:** {metadata.get('record_type', 'Unknown')}"
                            )
                            st.write(source["document"])
                            st.divider()

            except Exception as exc:
                st.error(f"An error occurred: {exc}")

st.caption(f"Records currently stored in ChromaDB: {total_records:,}")
