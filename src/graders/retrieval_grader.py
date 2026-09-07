import logging
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RetrievalGrade(BaseModel):
    """Binary score and justification for document relevance check."""
    binary_score: Literal["yes", "no"] = Field(
        description="Whether the document is relevant to the question: 'yes' or 'no'"
    )
    reason: str = Field(
        description="Brief concise reason explaining the relevance decision"
    )


RETRIEVAL_GRADER_SYSTEM_PROMPT = """You are an expert evaluator assessing relevance of a retrieved document to a user question.
Evaluate strictly: if the document contains keywords, concepts, or semantic information directly relevant to the question, grade it as 'yes'.
If the document is off-topic, unrelated, or discusses completely different concepts, grade it as 'no'.
Return your assessment strictly according to the requested schema with binary_score ('yes' or 'no') and a concise reason."""

RETRIEVAL_GRADER_USER_PROMPT = """Retrieved Document:
{document}

User Question: {question}"""


def get_retrieval_grader_chain(llm: Optional[Any] = None):
    """Build structured LLM chain for retrieval grading."""
    if llm is None:
        llm = get_llm(temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RETRIEVAL_GRADER_SYSTEM_PROMPT),
        ("user", RETRIEVAL_GRADER_USER_PROMPT),
    ])

    structured_llm = llm.with_structured_output(RetrievalGrade)
    return prompt | structured_llm


def grade_chunk(
    question: str,
    document_content: str,
    llm: Optional[Any] = None,
) -> RetrievalGrade:
    """Grade a single retrieved document chunk."""
    chain = get_retrieval_grader_chain(llm=llm)
    try:
        grade: RetrievalGrade = chain.invoke({
            "question": question,
            "document": document_content,
        })
        return grade
    except Exception as e:
        logger.warning(f"Grading parse error: {e}. Defaulting to conservative 'no'.")
        return RetrievalGrade(binary_score="no", reason=f"Grading parse exception: {e}")


def grade_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    llm: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Grade a list of retrieved chunks.
    Guarantees exactly len(chunks) results returned (no dropped items).
    Each chunk dictionary receives 'binary_score', 'grade_reason', and 'is_relevant'.
    """
    if not chunks:
        return []

    graded_results: List[Dict[str, Any]] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        grade = grade_chunk(question=question, document_content=content, llm=llm)

        is_relevant = grade.binary_score.lower() == "yes"
        graded_chunk = dict(chunk)
        graded_chunk["binary_score"] = grade.binary_score.lower()
        graded_chunk["grade_reason"] = grade.reason
        graded_chunk["is_relevant"] = is_relevant
        graded_results.append(graded_chunk)

    return graded_results
