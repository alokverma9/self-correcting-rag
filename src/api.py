import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import run_corrective_rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Corrective RAG (arXiv Research Assistant) API",
    description=(
        "Production serving layer for Corrective RAG with retrieval grading, "
        "query rewriting, web fallback, and hallucination self-correction."
    ),
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question to answer via Corrective RAG")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    correction_path_taken: bool = False
    rewritten_query: Optional[str] = None
    correction_log: List[str] = []


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint returning 200 OK when service is running."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Process a question through the Corrective RAG workflow.
    Validates non-empty input, grades retrieved documents, rewrites and triggers
    web search fallback when needed, and validates answer groundedness.
    """
    cleaned_question = request.question.strip() if request.question else ""
    if not cleaned_question:
        logger.warning("Rejected /query request with empty or whitespace question.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace only.",
        )

    try:
        logger.info(f"Processing query: '{cleaned_question}' (top_k={request.top_k})")
        result = run_corrective_rag(
            question=cleaned_question,
            top_k=request.top_k,
        )

        return QueryResponse(
            question=result.get("question", cleaned_question),
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            correction_path_taken=result.get("correction_path_taken", False),
            rewritten_query=result.get("rewritten_query"),
            correction_log=result.get("correction_log", []),
        )
    except Exception as e:
        logger.error(f"Error executing Corrective RAG pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the request: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
