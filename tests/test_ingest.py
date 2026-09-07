import json
import os
import tempfile
from pathlib import Path
import pytest

from src.ingest import fetch_arxiv_papers, validate_paper


def test_fetch_papers_metadata_and_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_papers.jsonl"
        
        # Test fetching 10 papers
        records = fetch_arxiv_papers(limit=10, output_path=output_file)
        
        # 1. Fetching 10 papers returns exactly 10 records
        assert len(records) == 10, f"Expected 10 records, got {len(records)}"
        
        # 2. Assert output file exists and has 10 lines
        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
        assert len(lines) == 10, f"Expected 10 lines in jsonl, got {len(lines)}"

        seen_ids = set()
        has_long_abstract = False

        for rec in records:
            # 3. Every record has non-empty title, abstract, arxiv_id
            assert rec.get("arxiv_id"), "Missing arxiv_id"
            assert rec.get("title"), "Missing title"
            assert rec.get("abstract"), "Missing abstract"
            assert len(rec["arxiv_id"].strip()) > 0
            assert len(rec["title"].strip()) > 0
            assert len(rec["abstract"].strip()) > 0

            # 4. No duplicate arxiv_ids
            arxiv_id = rec["arxiv_id"]
            assert arxiv_id not in seen_ids, f"Duplicate arxiv_id found: {arxiv_id}"
            seen_ids.add(arxiv_id)

            # Check abstract length
            if len(rec["abstract"]) > 200:
                has_long_abstract = True

        # 5. Assert at least one record's abstract length > 200 chars
        assert has_long_abstract, "Expected at least one paper with abstract > 200 chars"


def test_validate_paper():
    valid_record = {
        "arxiv_id": "2401.00001v1",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    }
    assert validate_paper(valid_record) is True

    # Empty arxiv_id
    assert validate_paper({"arxiv_id": "", "title": "T", "abstract": "A"}) is False
    # Missing title
    assert validate_paper({"arxiv_id": "123", "abstract": "A"}) is False
    # Empty abstract
    assert validate_paper({"arxiv_id": "123", "title": "T", "abstract": "   "}) is False
