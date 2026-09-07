import logging
from typing import TypedDict, List, Dict, Any, Optional, Union
from langgraph.graph import StateGraph, END

from src.embed import search_similar_chunks, DEFAULT_DB_URL
from src.graders.retrieval_grader import grade_chunks
from src.router import route_retrieval, DEFAULT_RELEVANCE_THRESHOLD
from src.rewrite import rewrite_query
from src.web_search import web_search
from src.naive_rag import generate_answer
from src.graders.hallucination_grader import grade_hallucination
from src.graders.answer_grader import grade_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """LangGraph state representation for Corrective RAG pipeline."""
    question: str
    rewritten_query: Optional[str]
    documents: List[Dict[str, Any]]
    generation: str
    web_search_needed: bool
    retries: int
    max_retries: int
    sources: List[Dict[str, Any]]
    correction_log: List[str]
    top_k: int
    db_url: str


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Retrieve documents from the vectorstore for the user's question."""
    question = state["question"]
    top_k = state.get("top_k", 5)
    db_url = state.get("db_url", DEFAULT_DB_URL)

    logger.info(f"--- NODE: RETRIEVE (top_k={top_k}) ---")
    chunks = search_similar_chunks(query=question, top_k=top_k, db_url=db_url)
    
    correction_log = list(state.get("correction_log", []))
    correction_log.append(f"Retrieved {len(chunks)} chunks from pgvector.")

    return {
        "documents": chunks,
        "correction_log": correction_log,
    }


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """Grade retrieved chunks for relevance to the question."""
    question = state["question"]
    documents = state.get("documents", [])
    correction_log = list(state.get("correction_log", []))

    logger.info("--- NODE: GRADE DOCUMENTS ---")
    graded_chunks = grade_chunks(question=question, chunks=documents)

    # Filter to only relevant chunks
    relevant_chunks = [c for c in graded_chunks if c.get("is_relevant") is True]

    # Decide route
    route = route_retrieval(graded_chunks, threshold=DEFAULT_RELEVANCE_THRESHOLD)
    needs_web = (route == "web_search")

    correction_log.append(
        f"Document grading: {len(relevant_chunks)}/{len(graded_chunks)} relevant. "
        f"Route decision: '{route}' (needs_web_search={needs_web})."
    )

    return {
        "documents": relevant_chunks,
        "web_search_needed": needs_web,
        "correction_log": correction_log,
    }


def decide_to_generate(state: GraphState) -> str:
    """Conditional edge deciding whether to route to web search or proceed to generate."""
    if state.get("web_search_needed", False):
        logger.info("--- CONDITIONAL EDGE: Routing to 'rewrite_query' ---")
        return "rewrite_query"
    else:
        logger.info("--- CONDITIONAL EDGE: Routing to 'generate' ---")
        return "generate"


def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
    """Rewrite failed query into an optimized web search query."""
    question = state["question"]
    correction_log = list(state.get("correction_log", []))

    logger.info("--- NODE: REWRITE QUERY (CORRECTIVE PATH) ---")
    rewritten = rewrite_query(question)
    correction_log.append(f"Correction triggered: Rewrote query '{question}' -> '{rewritten}'.")

    return {
        "rewritten_query": rewritten,
        "correction_log": correction_log,
    }


def web_search_node(state: GraphState) -> Dict[str, Any]:
    """Fallback to web search for supplementary context."""
    query = state.get("rewritten_query") or state["question"]
    existing_docs = list(state.get("documents", []))
    correction_log = list(state.get("correction_log", []))

    logger.info(f"--- NODE: WEB SEARCH FALLBACK for '{query}' ---")
    try:
        web_results = web_search(query=query, top_k=3)
        correction_log.append(f"Web search fallback retrieved {len(web_results)} web documents.")
        combined_docs = existing_docs + web_results
    except Exception as e:
        logger.warning(f"Web search fallback encountered error: {e}")
        correction_log.append(f"Web search failed: {e}. Using existing documents.")
        combined_docs = existing_docs

    return {
        "documents": combined_docs,
        "correction_log": correction_log,
    }


def generate_node(state: GraphState) -> Dict[str, Any]:
    """Generate answer from current pool of verified / fallback documents."""
    question = state["question"]
    documents = state.get("documents", [])
    correction_log = list(state.get("correction_log", []))
    retries = state.get("retries", 0)

    logger.info(f"--- NODE: GENERATE (attempt={retries + 1}) ---")
    answer = generate_answer(query=question, retrieved_chunks=documents)

    sources = [
        {
            "arxiv_id": c.get("arxiv_id"),
            "title": c.get("title"),
            "chunk_idx": c.get("chunk_idx"),
            "similarity": c.get("similarity"),
            "source": c.get("source", "pgvector"),
            "url": c.get("url"),
        }
        for c in documents
    ]

    correction_log.append(f"Generated answer candidate on attempt {retries + 1}.")

    return {
        "generation": answer,
        "sources": sources,
        "correction_log": correction_log,
    }


def check_hallucination_and_answer(state: GraphState) -> str:
    """Conditional edge evaluating groundedness and answer relevance with bounded retries."""
    question = state["question"]
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", 2)
    correction_log = state.get("correction_log", [])

    logger.info("--- CONDITIONAL EDGE: EVALUATING GENERATION ---")

    # 1. Check hallucination (groundedness)
    h_grade = grade_hallucination(documents=documents, generation=generation)
    if not h_grade.is_grounded:
        logger.info(f"Hallucination detected: {h_grade.reason}")
        if retries < max_retries:
            state["retries"] = retries + 1
            correction_log.append(
                f"Correction: Hallucination detected ({h_grade.reason}). Retrying generation (retry {state['retries']}/{max_retries})."
            )
            return "generate"
        else:
            logger.warning(f"Exceeded max retries ({max_retries}). Ending with best-effort disclaimer.")
            state["generation"] += (
                "\n\n[Disclaimer: This answer is provided as a best-effort response after multiple "
                "verification attempts and may require independent verification against original sources.]"
            )
            return "end"

    # 2. Check answer relevance
    a_grade = grade_answer(question=question, generation=generation)
    if not a_grade.is_useful:
        logger.info(f"Answer not useful: {a_grade.reason}")
        if retries < max_retries:
            state["retries"] = retries + 1
            correction_log.append(
                f"Correction: Answer relevance failed ({a_grade.reason}). Retrying generation (retry {state['retries']}/{max_retries})."
            )
            return "generate"
        else:
            logger.warning(f"Exceeded max retries ({max_retries}). Ending with best-effort disclaimer.")
            state["generation"] += (
                "\n\n[Disclaimer: This answer is provided as a best-effort response after multiple "
                "verification attempts and could not fully resolve the question based on available documents.]"
            )
            return "end"

    logger.info("Generation passed both groundedness and answer relevance checks!")
    correction_log.append("Answer passed both hallucination and answer relevance grading.")
    return "end"


def build_corrective_rag_graph():
    """Build and compile the LangGraph StateGraph for Corrective RAG."""
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    # Add Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )

    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        check_hallucination_and_answer,
        {
            "generate": "generate",
            "end": END,
        },
    )

    return workflow.compile()


# Global compiled graph instance
corrective_rag_app = build_corrective_rag_graph()


def run_corrective_rag(
    question: str,
    top_k: int = 5,
    db_url: str = DEFAULT_DB_URL,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Execute the full Corrective RAG workflow on a user question."""
    initial_state: GraphState = {
        "question": question,
        "rewritten_query": None,
        "documents": [],
        "generation": "",
        "web_search_needed": False,
        "retries": 0,
        "max_retries": max_retries,
        "sources": [],
        "correction_log": [],
        "top_k": top_k,
        "db_url": db_url,
    }

    final_state = corrective_rag_app.invoke(initial_state)

    correction_path_taken = (
        final_state.get("web_search_needed", False)
        or final_state.get("retries", 0) > 0
    )

    return {
        "question": final_state["question"],
        "answer": final_state["generation"],
        "documents": final_state["documents"],
        "sources": final_state["sources"],
        "rewritten_query": final_state.get("rewritten_query"),
        "correction_path_taken": correction_path_taken,
        "correction_log": final_state.get("correction_log", []),
    }
