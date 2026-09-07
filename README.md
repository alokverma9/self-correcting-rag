# Corrective RAG (arXiv Research Assistant)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-336791.svg)](https://github.com/pgvector/pgvector)
[![RAGAS Evaluated](https://img.shields.io/badge/RAGAS-Benchmark%20Passed-purple.svg)](https://github.com/explodinggradients/ragas)

A production-grade **Corrective Retrieval-Augmented Generation (CRAG)** system engineered to eliminate standard RAG failure modes. Built with **LangGraph**, **pgvector**, **Google Gemini**, **DuckDuckGo / Tavily**, and **FastAPI**, this assistant self-evaluates retrieved evidence, rewrites deficient queries to trigger external web fallback, and verifies generated answers through double-loop hallucination and relevance graders.

---

## 🎯 The Problem & Why Naive RAG Fails

Standard Naive RAG pipelines follow a rigid linear pattern: *Retrieve Chunks → Stuff into Prompt → Generate Answer*. In production environments, this brittle flow suffers from three major failure modes:

1. **Irrelevant Context Pollution:** Vector stores frequently return high cosine-similarity chunks containing overlapping vocabulary that are contextually irrelevant, confusing the generator and degrading answer precision.
2. **Corpus Blindspots & Out-of-Domain Failure:** When a query addresses topics not represented in the local vector database, Naive RAG either hallucinates plausible-sounding fabrications or throws a generic "I don't know" error.
3. **Ungrounded Hallucinations & Drift:** Generator models often introduce unsupported claims or drift away from answering the user's explicit question without any internal validation mechanism.

### The Corrective RAG Solution
This project implements an agentic self-correcting state machine:
- **Retrieval Grader:** Evaluates every retrieved document with structured classification (`yes`/`no` relevance + reasoning).
- **Dynamic Routing & Query Rewriter:** When local evidence is insufficient (<50% relevance), queries are rewritten using search-optimized keyword extraction and routed to web search fallback.
- **Hallucination & Relevance Graders:** Generated answers are graded against the retrieved context for factual groundedness, and against the original question for relevance, driving a bounded retry correction loop.

---

## 🏗️ Architecture & State Machine

The entire workflow is orchestrated as a cyclical `StateGraph` in LangGraph:

```mermaid
flowchart TD
    Start([User Query]) --> Retrieve[Retrieve Chunks from pgvector<br/>BAAI/bge-large-en-v1.5]
    Retrieve --> GradeDocs{Retrieval Grader<br/>Score Relevance per Chunk}
    
    GradeDocs -->|Relevance Ratio >= 50%| Generate[Generate Answer Candidate<br/>Gemini Model]
    GradeDocs -->|Relevance Ratio < 50%| Rewrite[Rewrite Query<br/>Query Optimization]
    
    Rewrite --> WebSearch[Web Search Fallback<br/>DuckDuckGo / Tavily API]
    WebSearch --> Generate
    
    Generate --> GradeHallucination{Hallucination Grader<br/>Grounded in Evidence?}
    GradeHallucination -->|Not Grounded & Retries < Max| Generate
    GradeHallucination -->|Grounded| GradeAnswer{Answer Grader<br/>Addresses Query?}
    
    GradeAnswer -->|Off-Topic & Retries < Max| Generate
    GradeAnswer -->|On-Topic| ReturnAnswer([Return Verified Answer + Sources])
    
    GradeHallucination -->|Retries Exhausted| DisclaimerAnswer([Return Best-Effort Answer + Fallback Disclaimer])
    GradeAnswer -->|Retries Exhausted| DisclaimerAnswer
```

---

## 📊 Evaluation & Benchmark Results (RAGAS)

Both **Naive RAG** and **Corrective RAG** were evaluated side-by-side using the industry-standard **RAGAS** evaluation framework across a diverse benchmark of in-corpus, ambiguous, and out-of-corpus queries:

| Metric | Naive RAG | Corrective RAG | Delta / Improvement |
|---|:---:|:---:|:---:|
| **Faithfulness** | 0.9333 | **0.9333** | **Maintained High (0.0)** |
| **Answer Relevancy** | 0.3833 | **0.9388** | **+55.6% (+0.5556)** 🚀 |
| **Context Precision** | 0.5000 | **0.6667** | **+16.7% (+0.1667)** 🚀 |
| **Context Recall** | 0.5000 | **0.5000** | **Maintained (0.0)** |

### 🔬 Key Findings:
- **+55.6% Answer Relevancy Jump:** Naive RAG completely failed on out-of-corpus and ambiguous queries, generating irrelevant or unhelpful responses. Corrective RAG dynamically identified missing information, triggered query rewrites, and gathered fresh web knowledge.
- **+16.7% Context Precision:** By filtering out irrelevant chunks before generation, the model was fed clean, focused context rather than noisy distractors.

---

## 🔍 Case Studies: Naive RAG Failure vs. Corrective RAG

### Case 1: Out-of-Corpus Query (Corpus Blindspot)
* **Question:** *"What is FlashAttention and who proposed it?"*
* ❌ **Naive RAG Output:**
  > *"Based on the provided context, there is no mention of who proposed FlashAttention, nor a comprehensive definition of what it is... Therefore, the provided context does not contain sufficient information to answer the question."*
* ✅ **Corrective RAG Execution Trace:**
  1. Retrieved 3 chunks from pgvector.
  2. Document grader evaluated 0/3 chunks relevant.
  3. Router triggered web fallback and query rewriter: `FlashAttention algorithm architecture authors research paper original publication`.
  4. Retrieved verified web documents and generated accurate, grounded response:
  > *"FlashAttention is an IO-aware exact attention algorithm that uses tiling to reduce memory reads and writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM, proposed by Tri Dao et al."*
  5. Passed hallucination and answer relevance grading on attempt 1.

### Case 2: Ambiguous / Incomplete Context Query
* **Question:** *"How do attention mechanisms perform sparse selection in vision models?"*
* ❌ **Naive RAG Output:**
  > *"Based on the provided documents, the context does not contain sufficient information to answer how attention mechanisms perform sparse selection in vision models."*
* ✅ **Corrective RAG Execution Trace:**
  1. Retrieved 3 chunks from pgvector.
  2. Grader identified 0/3 chunks adequately addressing the specific mechanism.
  3. Query rewritten to: `sparse attention mechanisms vision transformers token selection computational efficiency computer vision`.
  4. Synthesized comprehensive grounded response covering Select and Pack Attention (SPA), STFormer, and token importance pruning.

---

## 📂 Repository Structure

```
corrective-rag-arxiv/
├── .env.example              # Environment variables template
├── docker-compose.yml        # Dockerized pgvector (PostgreSQL 16)
├── requirements.txt          # Pinned dependencies
├── pytest.ini                # Pytest configuration
├── src/
│   ├── __init__.py
│   ├── api.py                # FastAPI serving layer (/health, /query)
│   ├── graph.py              # LangGraph StateGraph assembly & workflow
│   ├── llm.py                # Multi-model Gemini client with fallback rotation
│   ├── embed.py              # BAAI/bge-large embeddings & pgvector interface
│   ├── ingest.py             # arXiv metadata & abstract ingestion
│   ├── chunk.py              # Recursive character chunking
│   ├── naive_rag.py          # Baseline RAG implementation
│   ├── rewrite.py            # LLM-based search query rewriter
│   ├── router.py             # Relevance threshold router
│   ├── web_search.py         # DuckDuckGo / Tavily web fallback
│   └── graders/
│       ├── retrieval_grader.py     # Document relevance grader (Pydantic)
│       ├── hallucination_grader.py # Groundedness verification grader
│       └── answer_grader.py        # Question-answer relevance grader
├── eval/
│   ├── eval_set.json         # 50 hand-labeled evaluation items
│   ├── run_eval.py           # Comparative RAGAS evaluation runner
│   ├── results.csv           # Evaluation metrics comparison table
│   └── results.json          # Full per-query traces and evaluation results
└── tests/
    ├── test_ingest.py        # arXiv ingestion test suite
    ├── test_chunk.py         # Chunking and boundary test suite
    ├── test_embed.py         # pgvector round-trip and cosine test suite
    ├── test_naive_rag.py     # Baseline naive RAG test suite
    ├── test_graders.py       # Grader test suite (100% pass verification)
    ├── test_router_rewrite.py# Router and rewrite test suite
    ├── test_web_search.py    # Web search fallback test suite
    ├── test_graph.py         # LangGraph integration and edge test suite
    ├── test_eval.py          # Evaluation harness test suite
    └── test_api.py           # FastAPI endpoints test suite
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- [Docker](https://www.docker.com/) & Docker Compose
- Python 3.10+ (tested on Python 3.10 through 3.14)
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 2. Clone & Environment Setup
```bash
git clone https://github.com/alokverma9/self-correcting-rag.git
cd self-correcting-rag

python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Start pgvector Database
```bash
docker compose up -d
```
Verify the container is healthy:
```bash
docker compose ps
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and provide your Google AI Studio API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GOOGLEAI_STUDIO_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://rag:rag@localhost:5432/corrective_rag
```

### 5. Ingest Corpus, Chunk, and Embed
```bash
# Ingest 300 papers from arXiv (cs.CL, cs.LG)
python -m src.ingest --max-results 300

# Chunk documents into ~500-token chunks with 50-token overlap
python -m src.chunk

# Embed chunks into pgvector with BAAI/bge-large-en-v1.5
python -m src.embed
```

### 6. Run Test Suite
All system components have automated unit and integration tests:
```bash
pytest
```

---

## ⚡ Serving with FastAPI

Launch the high-performance FastAPI server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### API Endpoints

#### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

#### Query Corrective RAG
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What does UniMate achieve in character animation?", "top_k": 3}'
```
**Response:**
```json
{
  "question": "What does UniMate achieve in character animation?",
  "answer": "UniMate synthesizes articulated motion for arbitrary character skeletons from a rigged 3D asset and a text prompt without requiring test-time optimization...",
  "sources": [
    {
      "arxiv_id": "2408.01234",
      "title": "UniMate: Unified Character Animation...",
      "chunk_idx": 0,
      "source_type": "arxiv"
    }
  ],
  "correction_path_taken": false,
  "rewritten_query": null,
  "correction_log": [
    "Retrieved 3 chunks from pgvector.",
    "Document grading: 3/3 relevant. Route decision: 'vectorstore' (needs_web_search=False).",
    "Generated answer candidate on attempt 1.",
    "Answer passed both hallucination and answer relevance grading."
  ]
}
```

---

## 📈 Running the Evaluation Harness

To replicate or extend the RAGAS comparative evaluation:
```bash
# Run comparative evaluation across query types
python -m eval.run_eval --limit 6 --balanced
```
This generates:
- `eval/results.csv`: Formatted side-by-side metric comparison.
- `eval/results.json`: Detailed query-by-query breakdown including latencies, rewrite strings, and grader decision traces.

---

## 📜 License
MIT License. Created for production-grade self-correcting agentic RAG exploration.
