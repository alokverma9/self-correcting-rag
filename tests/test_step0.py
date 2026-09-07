import subprocess
import psycopg
import pytest

def test_imports():
    import langgraph
    import psycopg
    import pgvector
    import sentence_transformers
    import arxiv
    import pypdf
    import fastapi
    import ragas

    assert langgraph is not None
    assert psycopg is not None
    assert pgvector is not None
    assert sentence_transformers is not None
    assert arxiv is not None
    assert pypdf is not None
    assert fastapi is not None
    assert ragas is not None

def test_pgvector_connection_and_extension():
    conn_url = "postgresql://rag:rag@localhost:5432/corrective_rag"
    with psycopg.connect(conn_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            row = cur.fetchone()
            assert row is not None, "pgvector extension not found in pg_extension"
            assert row[0] == "vector"

            cur.execute("SELECT 1;")
            assert cur.fetchone()[0] == 1
