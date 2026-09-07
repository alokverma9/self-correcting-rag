import pytest
from src.graders.hallucination_grader import grade_hallucination, HallucinationGrade


def test_quote_and_fabricated_pair():
    context = (
        "Influence Score measures the effective impact of individual attention heads "
        "at inference time by tracking gradient flows through the residual stream."
    )

    # 1. Answer that directly quotes / paraphrases the context -> grounded ('yes')
    grounded_answer = "Influence Score measures the impact of attention heads during inference time by monitoring gradient flows in the residual stream."
    grade_grounded = grade_hallucination(documents=context, generation=grounded_answer)
    assert grade_grounded.binary_score == "yes", f"Expected 'yes', got: {grade_grounded}"
    assert grade_grounded.is_grounded is True

    # 2. Answer containing fabricated fact -> not grounded ('no')
    fabricated_answer = "Influence Score was developed by Albert Einstein in 1905 to calculate the mass-energy equivalence of photons."
    grade_fabricated = grade_hallucination(documents=context, generation=fabricated_answer)
    assert grade_fabricated.binary_score == "no", f"Expected 'no', got: {grade_fabricated}"
    assert grade_fabricated.is_grounded is False


def test_five_handwritten_grounded_pairs_100_percent_match():
    # 5 hand-written grounded / not-grounded pairs with 100% assertion match
    test_cases = [
        # Pair 1: Grounded
        {
            "context": "The Transformer architecture was introduced by Vaswani et al. in 2017 in the paper 'Attention Is All You Need'.",
            "generation": "Vaswani et al. proposed the Transformer architecture in 2017 in their paper titled Attention Is All You Need.",
            "expected": "yes",
        },
        # Pair 2: Not Grounded (fabricated dimension and author)
        {
            "context": "BGE-large-en-v1.5 produces dense vector representations of dimension 1024.",
            "generation": "BGE-large-en-v1.5 outputs 4096-dimensional embeddings and was invented by OpenAI in 1999.",
            "expected": "no",
        },
        # Pair 3: Grounded (faithful paraphrase)
        {
            "context": "Stochastic Gradient Descent (SGD) with momentum accelerates gradient vectors in the right directions, leading to faster converging.",
            "generation": "Adding momentum to SGD accelerates gradient vectors in the relevant directions to speed up convergence.",
            "expected": "yes",
        },
        # Pair 4: Not Grounded (hallucinated topic)
        {
            "context": "UniMate is a unified framework designed to animate diverse skeleton structures across 3D character models.",
            "generation": "UniMate is a medical diagnostics AI system designed to analyze patient electrocardiograms for cardiac arrest.",
            "expected": "no",
        },
        # Pair 5: Grounded (exact hyperparameter alignment)
        {
            "context": "The learning rate was initialized to 0.001 and decayed by a factor of 0.1 every 10 epochs for a total of 50 epochs.",
            "generation": "The learning rate started at 0.001 and was decayed by a factor of 0.1 every 10 epochs across a total of 50 epochs.",
            "expected": "yes",
        },
    ]

    for idx, tc in enumerate(test_cases, 1):
        grade = grade_hallucination(documents=tc["context"], generation=tc["generation"])
        assert grade.binary_score == tc["expected"], (
            f"Failed on Pair {idx}: expected '{tc['expected']}', got '{grade.binary_score}'. "
            f"Reason: {grade.reason}"
        )
