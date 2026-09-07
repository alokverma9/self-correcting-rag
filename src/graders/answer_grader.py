import logging
from typing import List, Dict, Any, Optional, Literal, Union, Callable
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_structured_llm
from src.graders.hallucination_grader import grade_hallucination

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AnswerGrade(BaseModel):
    """Binary score and explanation for whether an answer resolves the user's question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Whether the answer resolves the question: 'yes' (on-topic/resolves) or 'no' (off-topic/does not resolve)"
    )
    reason: str = Field(
        description="Concise explanation of whether the answer directly resolves the question"
    )

    @property
    def is_useful(self) -> bool:
        return self.binary_score.lower() == "yes"


ANSWER_GRADER_SYSTEM_PROMPT = """You are an expert evaluator assessing whether a generated answer actually resolves the user's question.
Give a binary score 'yes' or 'no':
- 'yes': The answer directly addresses and resolves the user's question with meaningful information.
- 'no': The answer does not resolve the question, is evasive, discusses an unrelated topic, or fails to provide an answer to what was asked.

Return strictly according to the schema with binary_score ('yes' or 'no') and a concise reason."""

ANSWER_GRADER_USER_PROMPT = """User Question: {question}

Generated Answer: {generation}"""


def get_answer_grader_chain(llm: Optional[Any] = None):
    """Build structured LLM chain for answer relevance grading."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_GRADER_SYSTEM_PROMPT),
        ("user", ANSWER_GRADER_USER_PROMPT),
    ])

    if llm is None:
        structured_llm = get_structured_llm(AnswerGrade, temperature=0.0)
    else:
        structured_llm = llm.with_structured_output(AnswerGrade)

    return prompt | structured_llm


def grade_answer(
    question: str,
    generation: str,
    llm: Optional[Any] = None,
) -> AnswerGrade:
    """Grade whether an answer resolves the user question."""
    chain = get_answer_grader_chain(llm=llm)
    try:
        grade: AnswerGrade = chain.invoke({
            "question": question,
            "generation": generation,
        })
        logger.info(f"Answer grading: binary_score='{grade.binary_score}', reason='{grade.reason}'")
        return grade
    except Exception as e:
        logger.warning(f"Answer grading exception: {e}. Defaulting to 'no'.")
        return AnswerGrade(binary_score="no", reason=f"Grading exception: {e}")


def execute_graded_generation_loop(
    question: str,
    documents: Union[str, List[Dict[str, Any]]],
    generate_fn: Callable[[str, Union[str, List[Dict[str, Any]]], int], str],
    max_retries: int = 2,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate answer and verify against both Hallucination and Answer graders.
    Loops with bounded retries (max_retries, default 2).
    If max retries exceeded, returns best-effort answer with disclaimer.
    """
    attempt = 0
    last_answer = ""
    is_grounded = False
    is_useful = False

    while attempt <= max_retries:
        logger.info(f"Answer generation attempt {attempt + 1}/{max_retries + 1}")
        answer = generate_fn(question, documents, attempt)
        last_answer = answer

        # 1. Hallucination check (groundedness)
        h_grade = grade_hallucination(documents=documents, generation=answer, llm=llm)
        is_grounded = h_grade.is_grounded

        if not is_grounded:
            logger.info(f"Attempt {attempt + 1}: Hallucination detected ({h_grade.reason}).")
            attempt += 1
            continue

        # 2. Answer relevance check (does it actually resolve the question?)
        a_grade = grade_answer(question=question, generation=answer, llm=llm)
        is_useful = a_grade.is_useful

        if not is_useful:
            logger.info(f"Attempt {attempt + 1}: Answer not relevant/useful ({a_grade.reason}).")
            attempt += 1
            continue

        # Passed both checks
        logger.info(f"Attempt {attempt + 1}: Passed groundedness and answer relevance.")
        return {
            "final_answer": answer,
            "attempts": attempt + 1,
            "passed_groundedness": True,
            "passed_answer_relevance": True,
            "exceeded_max_retries": False,
        }

    # If we exited loop, max retries was reached
    logger.warning(f"Exceeded max retries ({max_retries}). Returning best-effort answer with disclaimer.")
    disclaimer = (
        "\n\n[Disclaimer: This answer is provided as a best-effort response after multiple "
        "verification attempts and may require independent verification against original sources.]"
    )
    return {
        "final_answer": last_answer + disclaimer,
        "attempts": attempt,
        "passed_groundedness": is_grounded,
        "passed_answer_relevance": is_useful,
        "exceeded_max_retries": True,
    }
