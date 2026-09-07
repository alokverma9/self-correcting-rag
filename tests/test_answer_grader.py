import pytest
from unittest.mock import MagicMock, patch
from src.graders.answer_grader import (
    grade_answer,
    execute_graded_generation_loop,
    AnswerGrade,
)
from src.graders.hallucination_grader import HallucinationGrade


def test_on_topic_grounded_answer_first_pass():
    question = "What is the role of self-attention in transformers?"
    context = (
        "Self-attention allows transformers to model pairwise interactions between all tokens "
        "in a sequence, enabling global context aggregation without recurrence."
    )
    
    def generate_fn(q, doc, attempt):
        return "Self-attention enables transformers to model pairwise token interactions to aggregate global context."

    result = execute_graded_generation_loop(
        question=question,
        documents=context,
        generate_fn=generate_fn,
        max_retries=2,
    )

    # Passes both graders, loop exits on first pass
    assert result["attempts"] == 1, f"Expected 1 attempt, took: {result['attempts']}"
    assert result["exceeded_max_retries"] is False
    assert result["passed_groundedness"] is True
    assert result["passed_answer_relevance"] is True
    assert "Disclaimer" not in result["final_answer"]


def test_off_topic_answer_triggers_retry():
    question = "What is the role of self-attention in transformers?"
    context = (
        "In transformer architectures, self-attention models dependencies across tokens. In culinary arts, "
        "sourdough bread is made by fermenting dough using wild lactobacillaceae and yeast."
    )

    # Attempt 0: grounded in context, but off-topic (about sourdough)
    # Attempt 1: on-topic (about self-attention)
    answers = [
        "Sourdough bread is made by fermenting dough with wild lactobacillaceae and yeast.",
        "Self-attention models dependencies across tokens in a transformer sequence.",
    ]

    def generate_fn(q, doc, attempt):
        return answers[min(attempt, len(answers) - 1)]

    result = execute_graded_generation_loop(
        question=question,
        documents=context,
        generate_fn=generate_fn,
        max_retries=2,
    )

    # Off-topic answer failed answer grader on attempt 0, succeeded on attempt 1
    assert result["attempts"] == 2, f"Expected 2 attempts, took: {result['attempts']}"
    assert result["exceeded_max_retries"] is False
    assert result["passed_answer_relevance"] is True


def test_max_retries_termination_no_hang():
    question = "What is quantum gravity?"
    context = "Document about classical mechanics."

    # Force failure 3 times in a row
    with patch("src.graders.answer_grader.grade_hallucination") as mock_h, \
         patch("src.graders.answer_grader.grade_answer") as mock_a:
        
        # Grounded but never answers question
        mock_h.return_value = HallucinationGrade(binary_score="yes", reason="Grounded")
        mock_a.return_value = AnswerGrade(binary_score="no", reason="Off topic")

        def failing_gen(q, doc, attempt):
            return f"Answer attempt {attempt}"

        result = execute_graded_generation_loop(
            question=question,
            documents=context,
            generate_fn=failing_gen,
            max_retries=2,
        )

        # Confirm the loop terminates at max retries (3 total executions) instead of hanging
        assert result["exceeded_max_retries"] is True
        assert result["attempts"] == 3
        assert "Disclaimer" in result["final_answer"]
