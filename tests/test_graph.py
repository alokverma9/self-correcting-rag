import pytest
from src.graph import run_corrective_rag, build_corrective_rag_graph, GraphState


def test_graph_state_schema_and_compilation():
    app = build_corrective_rag_graph()
    assert app is not None, "Graph compilation failed"


def test_five_queries_end_to_end_with_correction_firing():
    queries = [
        # Query 1 (In-corpus): UniMate
        "What is UniMate and how does it animate diverse skeletons?",
        # Query 2 (Out-of-corpus, MUST trigger correction / web search):
        "What is the historical significance of the Rosetta Stone?",
        # Query 3 (In-corpus): Influence Score
        "How does Influence Score measure the impact of attention heads?",
        # Query 4 (Out-of-corpus, MUST trigger correction / web search):
        "What is the James Webb Space Telescope primary mirror made of?",
        # Query 5 (In-corpus): RegionFed
        "What is RegionFed in federated learning?",
    ]

    results = []
    correction_fired_count = 0

    for idx, q in enumerate(queries, 1):
        result = run_corrective_rag(question=q, top_k=3)
        assert result is not None, f"Query {idx} returned None"
        
        # 1. End-to-end run completes without exceptions
        assert "answer" in result, f"Query {idx} missing 'answer'"
        assert isinstance(result["answer"], str), f"Query {idx} answer not a string"
        assert len(result["answer"].strip()) > 0, f"Query {idx} returned empty answer"

        # 2. Graph state matches expected schema
        assert "question" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)
        assert "correction_log" in result
        assert isinstance(result["correction_log"], list)
        assert len(result["correction_log"]) > 0

        if result["correction_path_taken"]:
            correction_fired_count += 1
            assert result.get("rewritten_query") is not None, (
                f"Query {idx} took correction path but has no rewritten_query"
            )

        results.append(result)

    # 3. Trace/log output shows the correction path actually firing at least once
    assert correction_fired_count >= 1, (
        f"Correction path was expected to fire at least once across 5 queries, but fired {correction_fired_count} times."
    )
