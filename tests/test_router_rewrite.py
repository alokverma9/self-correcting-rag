import pytest
from src.router import route_retrieval, route_decision_details
from src.rewrite import rewrite_query


def test_router_all_relevant():
    # All-relevant grades -> router picks vectorstore only ("generate")
    graded_chunks = [
        {"arxiv_id": "p1", "content": "c1", "is_relevant": True, "binary_score": "yes"},
        {"arxiv_id": "p2", "content": "c2", "is_relevant": True, "binary_score": "yes"},
        {"arxiv_id": "p3", "content": "c3", "is_relevant": True, "binary_score": "yes"},
    ]
    route = route_retrieval(graded_chunks, threshold=0.5)
    assert route == "generate", f"Expected 'generate', got: {route}"

    details = route_decision_details(graded_chunks, threshold=0.5)
    assert details["needs_web_search"] is False
    assert details["relevance_ratio"] == 1.0


def test_router_all_irrelevant():
    # All-irrelevant grades -> router triggers rewrite + web search path ("web_search")
    graded_chunks = [
        {"arxiv_id": "p1", "content": "c1", "is_relevant": False, "binary_score": "no"},
        {"arxiv_id": "p2", "content": "c2", "is_relevant": False, "binary_score": "no"},
    ]
    route = route_retrieval(graded_chunks, threshold=0.5)
    assert route == "web_search", f"Expected 'web_search', got: {route}"

    details = route_decision_details(graded_chunks, threshold=0.5)
    assert details["needs_web_search"] is True
    assert details["relevance_ratio"] == 0.0


def test_router_below_threshold():
    # 1 relevant out of 4 -> 25% < 50% threshold -> web_search
    graded_chunks = [
        {"is_relevant": True, "binary_score": "yes"},
        {"is_relevant": False, "binary_score": "no"},
        {"is_relevant": False, "binary_score": "no"},
        {"is_relevant": False, "binary_score": "no"},
    ]
    route = route_retrieval(graded_chunks, threshold=0.5)
    assert route == "web_search"


def test_query_rewriter_modifies_and_improves():
    original_query = "how do they do that thing with attention heads in modern nlp?"
    rewritten = rewrite_query(original_query)
    
    assert isinstance(rewritten, str)
    assert len(rewritten.strip()) > 0
    # Rewritten query is not identical to the original (sanity check the LLM actually changed something)
    assert rewritten.strip().lower() != original_query.strip().lower(), (
        f"Rewritten query '{rewritten}' was identical to original '{original_query}'"
    )
    # Check that technical terms like attention or transformer or nlp appear in rewritten
    assert any(term in rewritten.lower() for term in ["attention", "transformer", "nlp", "head"])
