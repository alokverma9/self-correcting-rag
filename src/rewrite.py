import logging
from typing import Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm import get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You are an expert search query optimizer for scientific and technical information retrieval.
Look at the user's initial question that failed to retrieve relevant internal documents.
Formulate an improved, clear, standalone search query optimized for web and academic search engines (such as Tavily / Google).
Focus on the core technical keywords, entities, and conceptual definitions.
Do not include conversational filler, preamble, quotes, or markdown formatting. Output ONLY the rewritten query string."""

REWRITE_USER_PROMPT = """Original Question: {question}

Improved Search Query:"""


def get_rewrite_chain(llm: Optional[Any] = None):
    """Build the query rewriting chain."""
    if llm is None:
        llm = get_llm(temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM_PROMPT),
        ("user", REWRITE_USER_PROMPT),
    ])

    return prompt | llm | StrOutputParser()


def rewrite_query(question: str, llm: Optional[Any] = None) -> str:
    """Rewrite a failed question into a more effective search query."""
    chain = get_rewrite_chain(llm=llm)
    rewritten = chain.invoke({"question": question})
    clean_rewritten = rewritten.strip().strip('"').strip("'")
    logger.info(f"Query rewrite: '{question}' -> '{clean_rewritten}'")
    return clean_rewritten
