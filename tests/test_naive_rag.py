import pytest
from src.naive_rag import retrieve, generate_answer, run_naive_rag


def test_known_paper_in_top_5():
    # Query matching the known paper: 2609.05415v1 "UniMate: One Unified Model to Animate Diverse Skeletons"
    query = "UniMate unified model animate diverse skeletons"
    results = retrieve(query, top_k=5)
    
    assert len(results) > 0, "No results returned"
    retrieved_ids = [r["arxiv_id"] for r in results]
    assert "2609.05415v1" in retrieved_ids, (
        f"Expected paper 2609.05415v1 to be in top-5 results, but got: {retrieved_ids}"
    )


def test_naive_rag_generation_non_empty():
    query = "What does the UniMate model do for skeleton animation?"
    result = run_naive_rag(query, top_k=3)
    
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0, "Expected non-empty answer from naive RAG"
    assert len(result["sources"]) > 0, "Expected sources in naive RAG result"


def test_naive_rag_empty_retrieval_edge_case():
    # Edge case: empty retrieval set should not crash
    empty_chunks = []
    answer = generate_answer("What is the speed of light in vacuum?", retrieved_chunks=empty_chunks)
    
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0, "Expected non-empty answer even with empty retrieval set"
    # Should indicate insufficient information or no context
    assert any(
        phrase in answer.lower()
        for phrase in ["context", "sufficient", "provided", "not contain", "information", "cannot", "do not know"]
    ), f"Expected answer to state lack of information, got: {answer}"
