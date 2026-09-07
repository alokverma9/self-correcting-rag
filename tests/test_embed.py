import pytest
import time
from src.embed import (
    get_embedding_model,
    embed_texts,
    init_db,
    insert_chunks,
    search_similar_chunks,
    get_chunk_count,
    get_db_connection,
    DEFAULT_DB_URL,
    EMBEDDING_DIM,
)


@pytest.fixture(scope="module")
def setup_vector_db():
    init_db(db_url=DEFAULT_DB_URL, drop_existing=True)
    yield
    # Cleanup after tests
    init_db(db_url=DEFAULT_DB_URL, drop_existing=True)


def test_embedding_dimension_matches(setup_vector_db):
    model = get_embedding_model()
    test_text = "Transformers are deep neural models based on self-attention."
    embeddings = embed_texts([test_text], model=model)
    
    # 1. Embedding dimension matches the model's expected output (1024 for bge-large)
    assert embeddings.shape == (1, EMBEDDING_DIM), f"Expected shape (1, {EMBEDDING_DIM}), got {embeddings.shape}"
    assert len(embeddings[0]) == 1024

    # 2. VECTOR(1024) column accepts it without error
    chunk = {
        "arxiv_id": "test.0001",
        "title": "Dimension Test",
        "chunk_idx": 0,
        "content": test_text,
    }
    inserted = insert_chunks([chunk], db_url=DEFAULT_DB_URL, model=model)
    assert inserted == 1


def test_round_trip_cosine_similarity(setup_vector_db):
    model = get_embedding_model()
    target_sentence = "Self-correcting retrieval augmented generation improves factual accuracy."
    
    chunk = {
        "arxiv_id": "test.0002",
        "title": "Round Trip Test",
        "chunk_idx": 0,
        "content": target_sentence,
    }
    insert_chunks([chunk], db_url=DEFAULT_DB_URL, model=model)

    # Round-trip query via ORDER BY embedding <=> query_embedding
    results = search_similar_chunks(target_sentence, top_k=1, db_url=DEFAULT_DB_URL, model=model)
    assert len(results) >= 1
    top_result = results[0]

    # Top-1 result cosine similarity > 0.99
    assert top_result["content"] == target_sentence
    assert top_result["similarity"] > 0.99, f"Expected similarity > 0.99, got {top_result['similarity']}"
    # Verify traceability fields
    assert top_result["arxiv_id"] == "test.0002"
    assert top_result["title"] == "Round Trip Test"
    assert top_result["chunk_idx"] == 0


def test_row_count_matches_ingested_no_drops(setup_vector_db):
    model = get_embedding_model()
    init_db(db_url=DEFAULT_DB_URL, drop_existing=True)
    
    test_batch = [
        {
            "arxiv_id": f"test.batch.{i:03d}",
            "title": f"Batch Test Paper {i}",
            "chunk_idx": j,
            "content": f"Content for paper {i} chunk {j} discussing machine learning algorithms.",
        }
        for i in range(5)
        for j in range(3)
    ]
    total_expected = len(test_batch)  # 15 chunks
    
    inserted = insert_chunks(test_batch, db_url=DEFAULT_DB_URL, batch_size=4, model=model)
    assert inserted == total_expected

    # Row count in the chunks table == number of chunks ingested (no silent drops)
    actual_count = get_chunk_count(db_url=DEFAULT_DB_URL)
    assert actual_count == total_expected, f"Expected {total_expected} rows, found {actual_count}"


def test_container_down_or_unreachable_error():
    # If the pgvector container isn't running or port is unreachable,
    # connection attempt raises a clear ConnectionError rather than hanging
    unreachable_url = "postgresql://rag:rag@localhost:5439/nonexistent_db"
    start_time = time.time()
    
    with pytest.raises(ConnectionError) as excinfo:
        get_db_connection(unreachable_url, timeout=2)
        
    duration = time.time() - start_time
    assert duration < 5.0, f"Connection attempt hung for {duration} seconds"
    assert "Could not connect to pgvector" in str(excinfo.value)
