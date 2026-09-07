import logging
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm, get_structured_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HallucinationGrade(BaseModel):
    """Binary score and explanation for groundedness check of generated answer against documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Whether the answer is grounded in the documents: 'yes' (grounded) or 'no' (not grounded / hallucinated)"
    )
    reason: str = Field(
        description="Explanation of whether every fact in the answer is grounded in the retrieved documents"
    )

    @property
    def is_grounded(self) -> bool:
        return self.binary_score.lower() == "yes"


HALLUCINATION_GRADER_SYSTEM_PROMPT = """You are an expert evaluator assessing whether an LLM generation is grounded in / supported by a set of retrieved documents.
Give a binary score 'yes' or 'no':
- 'yes': The generation is fully grounded in the retrieved context. Every factual assertion, number, or claim in the answer is supported by or directly inferable from the documents.
- 'no': The generation is NOT grounded in the retrieved context. It introduces outside claims, fabricated numbers, invented authors/events, or assertions not supported by the provided text.

Be strict: any unsupported factual claim makes the generation 'no'.
Return strictly following the schema with binary_score ('yes' or 'no') and a concise reason."""

HALLUCINATION_GRADER_USER_PROMPT = """Retrieved Context Documents:
{documents}

Generated Answer:
{generation}"""


def format_docs_for_grader(documents: Union[str, List[Dict[str, Any]]]) -> str:
    """Format documents into string context for hallucination grading."""
    if isinstance(documents, str):
        return documents
    if not documents:
        return "No documents provided."

    doc_parts = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("content", "")
        title = doc.get("title", "Untitled")
        doc_parts.append(f"Document [{i}] ({title}):\n{content}")
    return "\n\n".join(doc_parts)


def get_hallucination_grader_chain(llm: Optional[Any] = None):
    """Build structured LLM chain for hallucination grading."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", HALLUCINATION_GRADER_SYSTEM_PROMPT),
        ("user", HALLUCINATION_GRADER_USER_PROMPT),
    ])

    if llm is None:
        structured_llm = get_structured_llm(HallucinationGrade, temperature=0.0)
    else:
        structured_llm = llm.with_structured_output(HallucinationGrade)

    return prompt | structured_llm


def grade_hallucination(
    documents: Union[str, List[Dict[str, Any]]],
    generation: str,
    llm: Optional[Any] = None,
) -> HallucinationGrade:
    """Grade whether a generated answer is grounded in the provided documents."""
    docs_str = format_docs_for_grader(documents)
    chain = get_hallucination_grader_chain(llm=llm)

    try:
        grade: HallucinationGrade = chain.invoke({
            "documents": docs_str,
            "generation": generation,
        })
        logger.info(f"Hallucination grading: binary_score='{grade.binary_score}', reason='{grade.reason}'")
        return grade
    except Exception as e:
        logger.warning(f"Hallucination grading exception: {e}. Defaulting to 'no'.")
        return HallucinationGrade(binary_score="no", reason=f"Grading exception: {e}")
