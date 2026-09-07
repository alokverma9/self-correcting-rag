import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_DB_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/corrective_rag")
EMBEDDING_DIM = 1024

_cached_model = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    global _cached_model
    if _cached_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _cached_model = SentenceTransformer(model_name)
    return _cached_model


def get_db_connection(db_url: str = DEFAULT_DB_URL, timeout: int = 5) -> psycopg.Connection:
    """Connect to pgvector database with timeout and register pgvector types."""
    try:
        conn = psycopg.connect(db_url, connect_timeout=timeout, autocommit=False)
        register_vector(conn)
        return conn
    except Exception as e:
        raise ConnectionError(f"Could not connect to pgvector instance at {db_url}: {e}") from e


def init_db(db_url: str = DEFAULT_DB_URL, drop_existing: bool = False) -> None:
    """Initialize Postgres database schema and HNSW cosine index."""
    with get_db_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            if drop_existing:
                cur.execute("DROP TABLE IF EXISTS chunks;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    arxiv_id TEXT NOT NULL,
                    title TEXT,
                    chunk_idx INT,
                    content TEXT,
                    embedding VECTOR({EMBEDDING_DIM})
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                    ON chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
        conn.commit()
    logger.info("Database schema initialized with HNSW cosine index.")


def embed_texts(
    texts: List[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute normalized dense embeddings for a list of text strings."""
    if model is None:
        model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def insert_chunks(
    chunks: List[Dict[str, Any]],
    db_url: str = DEFAULT_DB_URL,
    batch_size: int = 64,
    model: Optional[SentenceTransformer] = None,
) -> int:
    """Embed chunks and insert them into the pgvector chunks table."""
    if not chunks:
        return 0

    if model is None:
        model = get_embedding_model()

    total_inserted = 0
    with get_db_connection(db_url) as conn:
        with conn.cursor() as cur:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                texts = [c["content"] for c in batch]
                embeddings = embed_texts(texts, model=model, batch_size=batch_size)

                rows_to_insert = [
                    (
                        c["arxiv_id"],
                        c.get("title", ""),
                        c.get("chunk_idx", 0),
                        c["content"],
                        embeddings[idx].tolist(),
                    )
                    for idx, c in enumerate(batch)
                ]

                cur.executemany(
                    """
                    INSERT INTO chunks (arxiv_id, title, chunk_idx, content, embedding)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    rows_to_insert,
                )
                total_inserted += len(rows_to_insert)
        conn.commit()

    logger.info(f"Inserted {total_inserted} chunks into vector database.")
    return total_inserted


def search_similar_chunks(
    query: str,
    top_k: int = 5,
    db_url: str = DEFAULT_DB_URL,
    model: Optional[SentenceTransformer] = None,
) -> List[Dict[str, Any]]:
    """Search for top-k similar chunks using pgvector cosine distance (<=>)."""
    if model is None:
        model = get_embedding_model()

    query_embedding = embed_texts([query], model=model)[0].tolist()

    with get_db_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, arxiv_id, title, chunk_idx, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

    results = [
        {
            "id": row[0],
            "arxiv_id": row[1],
            "title": row[2],
            "chunk_idx": row[3],
            "content": row[4],
            "similarity": float(row[5]),
        }
        for row in rows
    ]
    return results


def get_chunk_count(db_url: str = DEFAULT_DB_URL) -> int:
    """Return the total number of chunks in the database."""
    with get_db_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks;")
            count = cur.fetchone()[0]
    return count


def index_file(
    input_file: str | Path = "data/processed/chunks.jsonl",
    db_url: str = DEFAULT_DB_URL,
    drop_existing: bool = False,
    batch_size: int = 64,
) -> int:
    """Read processed chunks from jsonl and index into pgvector."""
    in_path = Path(input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {in_path}")

    init_db(db_url=db_url, drop_existing=drop_existing)

    chunks = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks from {input_file}. Beginning embedding and indexing...")
    return insert_chunks(chunks, db_url=db_url, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Dense embedding and pgvector indexing")
    parser.add_argument("--input", type=str, default="data/processed/chunks.jsonl", help="Input chunks jsonl")
    parser.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="Database URL")
    parser.add_argument("--init-only", action="store_true", help="Initialize schema only")
    parser.add_argument("--drop", action="store_true", help="Drop existing chunks table")
    parser.add_argument("--query", type=str, default=None, help="Query vector store")
    parser.add_argument("--top-k", type=int, default=5, help="Top k results")
    args = parser.parse_args()

    if args.init_only:
        init_db(args.db_url, drop_existing=args.drop)
    elif args.query:
        results = search_similar_chunks(args.query, top_k=args.top_k, db_url=args.db_url)
        for r in results:
            print(f"[{r['similarity']:.4f}] ({r['arxiv_id']} #{r['chunk_idx']}) {r['title']}: {r['content'][:120]}...")
    else:
        index_file(args.input, db_url=args.db_url, drop_existing=args.drop)


if __name__ == "__main__":
    main()
