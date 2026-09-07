import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional

import arxiv
import pypdf
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("data/raw/papers.jsonl")


def clean_text(text: str) -> str:
    """Normalize whitespace and clean newlines in text."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_paper(record: Dict[str, Any]) -> bool:
    """Validate that required fields exist and are non-empty."""
    arxiv_id = record.get("arxiv_id")
    title = record.get("title")
    abstract = record.get("abstract")

    if not arxiv_id or not str(arxiv_id).strip():
        return False
    if not title or not str(title).strip():
        return False
    if not abstract or not str(abstract).strip():
        return False
    return True


def extract_pdf_text(pdf_url: str, max_pages: int = 5, timeout: int = 15) -> str:
    """Optionally download and extract text from arXiv PDF with page limit."""
    try:
        response = requests.get(pdf_url, timeout=timeout, headers={"User-Agent": "CorrectiveRAG/1.0"})
        if response.status_code != 200:
            return ""
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            reader = pypdf.PdfReader(tmp_path)
            pages_text = []
            for idx, page in enumerate(reader.pages[:max_pages]):
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
            return "\n\n".join(pages_text)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        logger.debug(f"Could not extract PDF text from {pdf_url}: {e}")
        return ""


def fetch_arxiv_papers(
    limit: int = 300,
    categories: Optional[List[str]] = None,
    download_pdf: bool = False,
    output_path: Optional[str | Path] = None,
    page_size: int = 100,
    delay_between_batches: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Fetch papers from arXiv API for specified categories (default cs.CL and cs.LG).
    Deduplicates by arxiv_id and validates metadata.
    """
    if categories is None:
        categories = ["cs.CL", "cs.LG"]

    query_str = " OR ".join([f"cat:{cat}" for cat in categories])
    logger.info(f"Querying arXiv with query='{query_str}', limit={limit}")

    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_between_batches,
        num_retries=5,
    )

    search = arxiv.Search(
        query=query_str,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    seen_ids = set()
    records: List[Dict[str, Any]] = []

    out_file = None
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path, "w", encoding="utf-8")

    try:
        count = 0
        for result in client.results(search):
            arxiv_id = result.get_short_id()
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            title = clean_text(result.title)
            abstract = clean_text(result.summary)
            pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            categories_list = list(result.categories)
            authors_list = [author.name for author in result.authors]
            published_str = result.published.isoformat() if result.published else ""

            body_text = ""
            if download_pdf and pdf_url:
                body_text = extract_pdf_text(pdf_url)

            # Combined textual content for chunking / indexing
            full_text = f"Title: {title}\n\nAbstract: {abstract}"
            if body_text:
                full_text += f"\n\nContent:\n{body_text}"

            record = {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "categories": categories_list,
                "authors": authors_list,
                "published": published_str,
                "pdf_url": pdf_url,
                "entry_id": result.entry_id,
                "text": full_text,
            }

            if not validate_paper(record):
                logger.warning(f"Skipping invalid paper record: {arxiv_id}")
                continue

            records.append(record)
            count += 1

            if out_file:
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()

            if count >= limit:
                break

            if count % 100 == 0:
                logger.info(f"Ingested {count}/{limit} papers...")

    finally:
        if out_file:
            out_file.close()

    logger.info(f"Completed ingestion: {len(records)} papers saved.")
    return records


def main():
    parser = argparse.ArgumentParser(description="arXiv corpus ingestion pipeline (cs.CL & cs.LG)")
    parser.add_argument("--limit", type=int, default=300, help="Number of papers to fetch (default: 300)")
    parser.add_argument("--full", action="store_true", help="Fetch full corpus (5000+ papers)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output path for papers.jsonl")
    parser.add_argument("--download-pdf", action="store_true", help="Attempt to download and extract PDF text")
    args = parser.parse_args()

    limit = 5200 if args.full else args.limit
    fetch_arxiv_papers(
        limit=limit,
        download_pdf=args.download_pdf,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
