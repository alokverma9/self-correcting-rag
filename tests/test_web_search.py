import pytest
from unittest.mock import patch, MagicMock
from src.web_search import web_search, search_duckduckgo, search_tavily


def test_known_query_returns_results_with_content():
    # Known query returns >=1 result with non-empty content
    query = "transformer attention mechanism"
    results = web_search(query, top_k=3)
    
    assert len(results) >= 1, "Expected at least 1 web result"
    top_result = results[0]
    
    # Assert chunk schema format
    assert "arxiv_id" in top_result
    assert "title" in top_result
    assert "chunk_idx" in top_result
    assert "content" in top_result
    assert isinstance(top_result["content"], str)
    assert len(top_result["content"].strip()) > 0, "Expected non-empty content in web result"


def test_tavily_bad_key_raises_clear_error():
    # Bad API key raises clear error rather than silent empty return
    with pytest.raises(Exception) as excinfo:
        search_tavily(query="transformer", api_key="tvly-invalid-fake-key-12345")
    
    assert "failure" in str(excinfo.value).lower() or "error" in str(excinfo.value).lower() or "api" in str(excinfo.value).lower()


def test_ddg_failure_raises_clear_error():
    # Network / API failure in DDG is caught and raises a clear error, not a silent empty return
    with patch("ddgs.DDGS") as mock_ddgs_cls:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value.text.side_effect = ConnectionError("Network unreachable")
        mock_ddgs_cls.return_value = mock_instance
        
        with pytest.raises(RuntimeError) as excinfo:
            search_duckduckgo("quantum computing")
        
        assert "failed" in str(excinfo.value).lower()
        assert "Network unreachable" in str(excinfo.value)
