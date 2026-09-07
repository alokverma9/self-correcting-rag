import argparse
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.embed import search_similar_chunks, DEFAULT_DB_URL
from src.llm import get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NAIVE_RAG_SYSTEM_PROMPT = """You are a helpful research assistant specializing in computer science research papers from arXiv (cs.CL and cs.LG).
Answer the user question strictly using the provided context. If the context does not contain enough information or is empty, state clearly that the provided context does not contain sufficient information to answer the question. Do not hallucinate external facts.

Context:
{context}
"""

NAIVE_RAG_USER_PROMPT = """Question: {question}

Answer:"""


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format a list of retrieved chunk dictionaries into a readable context string."""
    if not retrieved_chunks:
        return "No relevant documents found."

    context_parts = []
    for i, c in enumerate(retrieved_chunks, 1):
        arxiv_id = c.get("arxiv_id", "unknown")
        title = c.get("title", "Untitled")
        content = c.get("content", "").strip()
        context_parts.append(
            f"--- Document [{i}] (arXiv: {arxiv_id} | Title: {title}) ---\n{content}"
        )
    return "\n\n".join(context_parts)


def retrieve(
    query: str,
    top_k: int = 5,
    db_url: str = DEFAULT_DB_URL,
) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks from the vector database."""
    return search_similar_chunks(query=query, top_k=top_k, db_url=db_url)


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    llm: Optional[Any] = None,
) -> str:
    """Stuff retrieved context into the prompt and generate an answer using LLM."""
    if llm is None:
        llm = get_llm(temperature=0.0)

    context = format_context(retrieved_chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", NAIVE_RAG_SYSTEM_PROMPT),
        ("user", NAIVE_RAG_USER_PROMPT),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": query, "context": context})
    return answer.strip()


def run_naive_rag(
    query: str,
    top_k: int = 5,
    db_url: str = DEFAULT_DB_URL,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute the naive RAG pipeline: retrieve -> format -> generate."""
    chunks = retrieve(query=query, top_k=top_k, db_url=db_url)
    answer = generate_answer(query=query, retrieved_chunks=chunks, llm=llm)

    sources = [
        {
            "arxiv_id": c.get("arxiv_id"),
            "title": c.get("title"),
            "chunk_idx": c.get("chunk_idx"),
            "similarity": c.get("similarity"),
        }
        for c in chunks
    ]

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": chunks,
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline Naive RAG for arXiv papers")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="pgvector database URL")
    args = parser.parse_args()

    result = run_naive_rag(query=args.query, top_k=args.top_k, db_url=args.db_url)
    print("\n" + "=" * 50)
    print(f"Query: {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("\nSources:")
    for s in result["sources"]:
        print(f" - [{s['similarity']:.4f}] arXiv:{s['arxiv_id']} #{s['chunk_idx']} - {s['title']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
