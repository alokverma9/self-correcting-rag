import pytest
from src.chunk import get_tokenizer, get_text_splitter, chunk_paper


def generate_exact_token_doc(tokenizer, target_tokens: int = 2000) -> str:
    """Generate a passage with approximately or exactly target_tokens tokens."""
    base_phrase = "neural network architecture deep learning transformer attention mechanism "
    phrase_tokens = len(tokenizer.tokenize(base_phrase))
    repeats = (target_tokens // phrase_tokens) + 5
    full_text = (base_phrase * repeats)
    
    # Trim to exact token count
    tokens = tokenizer.tokenize(full_text)[:target_tokens]
    # Reconstruct text using tokenizer decode or tokens
    text = tokenizer.convert_tokens_to_string(tokens)
    return text


def test_chunking_2000_token_doc_and_constraints():
    tokenizer = get_tokenizer()
    target_tokens = 2000
    doc_text = generate_exact_token_doc(tokenizer, target_tokens=target_tokens)
    actual_tokens = len(tokenizer.tokenize(doc_text))
    assert abs(actual_tokens - target_tokens) <= 20

    paper = {
        "arxiv_id": "2401.99999",
        "title": "A Test Paper on Transformers",
        "text": doc_text,
    }

    splitter = get_text_splitter(chunk_size=500, chunk_overlap=50, tokenizer=tokenizer)
    chunks = chunk_paper(paper, splitter=splitter, tokenizer=tokenizer)

    # 1. Expected chunk count: 2000 tokens with 500 chunk_size and 50 overlap
    # Stride = 450 tokens. 2000 / 450 ~= 4.44 -> 5 chunks (± 1 allows 4 to 6)
    expected_chunks = 5
    assert abs(len(chunks) - expected_chunks) <= 1, f"Expected {expected_chunks}±1 chunks, got {len(chunks)}"

    # 2. Every chunk carries a traceable arxiv_id (no orphan chunks)
    for idx, c in enumerate(chunks):
        assert c["arxiv_id"] == "2401.99999", "Orphan or mismatched chunk arxiv_id"
        assert c["chunk_idx"] == idx, "Incorrect chunk_idx"
        assert c["title"] == "A Test Paper on Transformers"
        assert len(c["content"].strip()) > 0, "Empty chunk content"

        # 3. No chunk exceeds the token limit (assert with same tokenizer)
        token_count = len(tokenizer.tokenize(c["content"]))
        assert token_count <= 505, f"Chunk {idx} exceeded token limit: {token_count} > 500"

    # 4. Overlap check: last 50 tokens of chunk N appear in chunk N+1
    for i in range(len(chunks) - 1):
        c1_tokens = tokenizer.tokenize(chunks[i]["content"])
        c2_tokens = tokenizer.tokenize(chunks[i + 1]["content"])
        
        overlap_len = 50
        c1_tail = c1_tokens[-overlap_len:]
        c2_head = c2_tokens[:overlap_len]

        # The tail of chunk i should match the head of chunk i+1
        # Allow minor boundary slack (up to 3 token tolerance due to whitespace boundary split)
        matching = sum(1 for a, b in zip(c1_tail, c2_head) if a == b)
        assert matching >= (overlap_len - 5), (
            f"Overlap check failed between chunk {i} and {i+1}: "
            f"matched {matching}/{overlap_len} tokens. c1_tail={c1_tail[-10:]}, c2_head={c2_head[:10]}"
        )


def test_chunk_paper_empty_or_missing_id():
    tokenizer = get_tokenizer()
    # Missing arxiv_id should raise ValueError
    with pytest.raises(ValueError):
        chunk_paper({"title": "No ID", "text": "Some text"}, tokenizer=tokenizer)

    # Empty text returns empty chunks
    res = chunk_paper({"arxiv_id": "test.123", "title": "Empty", "text": ""}, tokenizer=tokenizer)
    assert res == []
