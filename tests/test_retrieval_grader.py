import pytest
from src.graders.retrieval_grader import grade_chunk, grade_chunks, RetrievalGrade


def test_relevant_and_irrelevant_pair():
    question = "How does multi-head self-attention work in transformer architectures?"
    
    relevant_text = (
        "Multi-head attention allows the model to jointly attend to information from "
        "different representation subspaces at different positions. Given queries Q, keys K, "
        "and values V, multi-head attention computes scaled dot-product attention in parallel across h heads."
    )
    
    irrelevant_text = (
        "To prepare authentic Neapolitan pizza dough, mix 1000g of Type 00 flour with 600ml "
        "of cold water, 30g of sea salt, and 2g of fresh yeast. Allow the dough balls to proof "
        "at 20 degrees Celsius for 24 hours."
    )

    relevant_grade = grade_chunk(question=question, document_content=relevant_text)
    irrelevant_grade = grade_chunk(question=question, document_content=irrelevant_text)

    # Grader labels them correctly in both directions
    assert relevant_grade.binary_score == "yes", f"Expected 'yes' for relevant chunk, got: {relevant_grade}"
    assert irrelevant_grade.binary_score == "no", f"Expected 'no' for irrelevant chunk, got: {irrelevant_grade}"
    assert len(relevant_grade.reason.strip()) > 0
    assert len(irrelevant_grade.reason.strip()) > 0


def test_schema_parsing_zero_failures():
    # Test 5 different queries, zero parse failures
    test_cases = [
        ("What is backpropagation?", "Backpropagation calculates gradients of the loss function with respect to weights."),
        ("Explain reinforcement learning with human feedback.", "RLHF aligns language models with human preferences using reward models."),
        ("What are convolutional layers used for?", "Convolutional neural networks apply kernel filters over spatial grid inputs like images."),
        ("What is LoRA parameter efficient fine-tuning?", "Low-Rank Adaptation freezes pretrained weights and injects trainable rank decomposition matrices."),
        ("How does gradient clipping prevent exploding gradients?", "Gradient clipping caps gradient norms to a maximum threshold during optimization."),
    ]

    for question, doc in test_cases:
        grade = grade_chunk(question=question, document_content=doc)
        assert isinstance(grade, RetrievalGrade), f"Failed to return RetrievalGrade for {question}"
        assert grade.binary_score in ["yes", "no"]
        assert isinstance(grade.reason, str) and len(grade.reason) > 0


def test_batch_10_chunks_exact_count():
    question = "What are optimization algorithms for training neural networks?"
    
    chunks = [
        {"arxiv_id": f"paper.{i}", "chunk_idx": 0, "content": f"Text discussing Adam and SGD optimization method {i}."}
        if i % 2 == 0
        else {"arxiv_id": f"paper.{i}", "chunk_idx": 0, "content": f"Completely unrelated text about baking pastries item {i}."}
        for i in range(10)
    ]

    # Batch of 10 chunks -> grader returns exactly 10 labels (no dropped items)
    graded_results = grade_chunks(question=question, chunks=chunks)
    assert len(graded_results) == 10, f"Expected exactly 10 results, got {len(graded_results)}"

    for i, res in enumerate(graded_results):
        assert res["arxiv_id"] == f"paper.{i}"
        assert "binary_score" in res
        assert res["binary_score"] in ["yes", "no"]
        assert "is_relevant" in res
        assert isinstance(res["is_relevant"], bool)
        assert "grade_reason" in res
