from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_returns_200():
    """Verify /health endpoint returns 200 OK and status is ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_query_valid_question_returns_200_and_required_fields():
    """Verify /query with valid question returns 200 and JSON with answer and sources fields."""
    mock_pipeline_output = {
        "question": "What is Transformer architecture?",
        "answer": "The Transformer is a neural network architecture based entirely on self-attention mechanisms.",
        "documents": [],
        "sources": [
            {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "chunk_idx": 0,
                "source_type": "arxiv",
            }
        ],
        "rewritten_query": None,
        "correction_path_taken": False,
        "correction_log": ["Retrieved 5 chunks", "Graded documents: 3/5 relevant"],
    }

    with patch("src.api.run_corrective_rag", return_value=mock_pipeline_output) as mock_run:
        response = client.post(
            "/query",
            json={"question": "What is Transformer architecture?", "top_k": 3},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["answer"] == mock_pipeline_output["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["arxiv_id"] == "1706.03762"
        assert data["correction_path_taken"] is False
        mock_run.assert_called_once_with(
            question="What is Transformer architecture?",
            top_k=3,
        )


def test_query_empty_string_returns_400():
    """Verify /query with an empty string returns 400 Bad Request, not a crash."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "empty" in detail.lower()


def test_query_whitespace_string_returns_400():
    """Verify /query with whitespace-only string returns 400 Bad Request."""
    response = client.post("/query", json={"question": "   \n\t  "})
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "empty" in detail.lower()


def test_query_missing_question_field_returns_422():
    """Verify /query without question field returns 422 Unprocessable Entity."""
    response = client.post("/query", json={"top_k": 5})
    assert response.status_code == 422


def test_query_internal_error_returns_500():
    """Verify /query returns 500 when underlying pipeline encounters an exception."""
    with patch("src.api.run_corrective_rag", side_effect=RuntimeError("Vectorstore connection lost")):
        response = client.post("/query", json={"question": "Sample question"})
        assert response.status_code == 500
        detail = response.json().get("detail", "")
        assert "Vectorstore connection lost" in detail
