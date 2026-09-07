import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RELEVANCE_THRESHOLD = 0.5


def route_retrieval(
    graded_chunks: List[Dict[str, Any]],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> str:
    """
    Decide next step based on the ratio of relevant chunks:
    - If ratio >= threshold: return "generate" (sufficient internal context)
    - If ratio < threshold (or empty): return "web_search" (trigger rewrite and web fallback)
    """
    if not graded_chunks:
        logger.info("No chunks retrieved. Routing to 'web_search'.")
        return "web_search"

    relevant_count = sum(
        1 for c in graded_chunks
        if c.get("is_relevant") is True or str(c.get("binary_score", "")).lower() == "yes"
    )
    total_count = len(graded_chunks)
    relevance_ratio = relevant_count / total_count

    logger.info(
        f"Retrieval evaluation: {relevant_count}/{total_count} relevant ({relevance_ratio:.1%}, threshold={threshold:.1%})"
    )

    if relevance_ratio >= threshold:
        return "generate"
    else:
        return "web_search"


def route_decision_details(
    graded_chunks: List[Dict[str, Any]],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> Dict[str, Any]:
    """Return comprehensive routing decision details."""
    route = route_retrieval(graded_chunks, threshold=threshold)
    total_count = len(graded_chunks)
    relevant_count = sum(
        1 for c in graded_chunks
        if c.get("is_relevant") is True or str(c.get("binary_score", "")).lower() == "yes"
    )
    ratio = (relevant_count / total_count) if total_count > 0 else 0.0

    return {
        "route": route,
        "relevant_count": relevant_count,
        "total_count": total_count,
        "relevance_ratio": ratio,
        "needs_web_search": (route == "web_search"),
    }
