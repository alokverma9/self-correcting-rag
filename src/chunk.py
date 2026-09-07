import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_INPUT_PATH = Path("data/raw/papers.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/chunks.jsonl")

_cached_tokenizer = None


def get_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    global _cached_tokenizer
    if _cached_tokenizer is None:
        _cached_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _cached_tokenizer


def get_text_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    tokenizer: Optional[Any] = None,
) -> RecursiveCharacterTextSplitter:
    """Create a token-based RecursiveCharacterTextSplitter using the HuggingFace tokenizer."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_paper(
    paper: Dict[str, Any],
    splitter: Optional[RecursiveCharacterTextSplitter] = None,
    tokenizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Chunk a single paper and attach traceable metadata."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    if splitter is None:
        splitter = get_text_splitter(tokenizer=tokenizer)

    arxiv_id = paper.get("arxiv_id", "").strip()
    if not arxiv_id:
        raise ValueError("Paper missing required 'arxiv_id'")

    title = paper.get("title", "").strip()
    text = paper.get("text") or paper.get("abstract") or ""

    if not text.strip():
        return []

    chunks_text = splitter.split_text(text)
    chunk_records: List[Dict[str, Any]] = []

    for idx, chunk_str in enumerate(chunks_text):
        tokens = tokenizer.tokenize(chunk_str)
        chunk_records.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "chunk_idx": idx,
                "content": chunk_str,
                "token_count": len(tokens),
            }
        )

    return chunk_records


def chunk_corpus(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: Optional[str | Path] = DEFAULT_OUTPUT_PATH,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Read papers from jsonl, chunk them, and write chunked records to jsonl."""
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    tokenizer = get_tokenizer()
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer)

    all_chunks: List[Dict[str, Any]] = []
    out_file = None

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_p, "w", encoding="utf-8")

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                paper = json.loads(line)
                paper_chunks = chunk_paper(paper, splitter=splitter, tokenizer=tokenizer)
                for chunk in paper_chunks:
                    all_chunks.append(chunk)
                    if out_file:
                        out_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        out_file.flush()
    finally:
        if out_file:
            out_file.close()

    logger.info(f"Chunked corpus into {len(all_chunks)} chunks.")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk arXiv papers with traceable IDs")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Input papers.jsonl")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output chunks.jsonl")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in tokens")
    args = parser.parse_args()

    chunk_corpus(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
