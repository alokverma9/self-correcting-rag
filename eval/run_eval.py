import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Python 3.14 asyncio compatibility patch for RAGAS
try:
    import asyncio.timeouts as timeouts
    _orig_timeout = timeouts.timeout

    @asynccontextmanager
    async def _safe_timeout(delay):
        if asyncio.current_task() is None:
            yield
        else:
            async with _orig_timeout(delay):
                yield

    timeouts.timeout = _safe_timeout
    asyncio.timeout = _safe_timeout
except Exception:
    pass

try:
    import nest_asyncio
    if sys.version_info >= (3, 12):
        nest_asyncio.apply = lambda *args, **kwargs: None
except ImportError:
    pass

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.llm import get_google_api_key
from src.naive_rag import run_naive_rag
from src.graph import run_corrective_rag

DEFAULT_EVAL_SET_PATH = Path("eval/eval_set.json")
DEFAULT_RESULTS_CSV = Path("eval/results.csv")
DEFAULT_RESULTS_JSON = Path("eval/results.json")


def setup_ragas_evaluator() -> Tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Set up RAGAS LLM and embedding evaluators."""
    key = get_google_api_key()
    eval_model_name = os.getenv("GEMINI_EVAL_MODEL", "gemini-3.5-flash-lite")
    logger.info(f"Initializing RAGAS evaluator LLM with model: {eval_model_name}")
    
    raw_llm = ChatGoogleGenerativeAI(
        model=eval_model_name,
        google_api_key=key,
        temperature=0.0,
    )
    ragas_llm = LangchainLLMWrapper(raw_llm)

    logger.info("Initializing RAGAS embedding evaluator with BAAI/bge-large-en-v1.5")
    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    return ragas_llm, ragas_embeddings


def execute_pipeline_runs(
    eval_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run both Naive RAG and Corrective RAG over all evaluation items.
    Returns (naive_dataset_records, corrective_dataset_records, per_query_logs).
    """
    naive_records = []
    corrective_records = []
    per_query_logs = []

    total = len(eval_items)
    logger.info(f"Starting pipeline runs for {total} evaluation items...")

    for i, item in enumerate(eval_items, 1):
        q = item["question"]
        gt = item["ground_truth"]
        q_type = item.get("type", "unknown")
        q_id = item.get("id", i)

        logger.info(f"[{i}/{total}] Evaluating Query #{q_id} ({q_type}): '{q}'")

        # 1. Run Naive RAG
        t0 = time.time()
        naive_res = run_naive_rag(query=q, top_k=3)
        t_naive = time.time() - t0

        naive_contexts = [c.get("content", "") for c in naive_res.get("retrieved_chunks", [])]
        if not naive_contexts:
            naive_contexts = ["No documents retrieved."]

        naive_records.append({
            "question": q,
            "answer": naive_res.get("answer", ""),
            "contexts": naive_contexts,
            "ground_truth": gt,
        })

        # 2. Run Corrective RAG
        t0 = time.time()
        corrective_res = run_corrective_rag(question=q, top_k=3)
        t_corrective = time.time() - t0

        corrective_contexts = [c.get("content", "") for c in corrective_res.get("documents", [])]
        if not corrective_contexts:
            corrective_contexts = ["No documents retrieved."]

        corrective_records.append({
            "question": q,
            "answer": corrective_res.get("answer", ""),
            "contexts": corrective_contexts,
            "ground_truth": gt,
        })

        per_query_logs.append({
            "id": q_id,
            "question": q,
            "type": q_type,
            "ground_truth": gt,
            "naive_answer": naive_res.get("answer", ""),
            "naive_contexts_count": len(naive_contexts),
            "naive_latency_sec": round(t_naive, 2),
            "corrective_answer": corrective_res.get("answer", ""),
            "corrective_contexts_count": len(corrective_contexts),
            "corrective_latency_sec": round(t_corrective, 2),
            "correction_path_taken": corrective_res.get("correction_path_taken", False),
            "rewritten_query": corrective_res.get("rewritten_query"),
            "correction_log": corrective_res.get("correction_log", []),
        })

    return naive_records, corrective_records, per_query_logs


def compute_ragas_scores(
    records: List[Dict[str, Any]],
    ragas_llm: LangchainLLMWrapper,
    ragas_embeddings: LangchainEmbeddingsWrapper,
    system_name: str,
) -> Dict[str, float]:
    """Compute RAGAS scores for a given system output dataset."""
    logger.info(f"Computing RAGAS metrics for {system_name} on {len(records)} samples...")
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    run_config = RunConfig(max_workers=2, timeout=90)

    results = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    clean_results = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        try:
            val = results[metric_name]
            mean_val = np.nanmean(val)
            clean_results[metric_name] = float(mean_val) if not np.isnan(mean_val) else 0.0
        except Exception as e:
            logger.warning(f"Error computing metric {metric_name}: {e}")
            clean_results[metric_name] = 0.0

    logger.info(f"Scores for {system_name}: {clean_results}")
    return clean_results


def run_evaluation(
    eval_set_path: Path = DEFAULT_EVAL_SET_PATH,
    limit: Optional[int] = None,
    balanced: bool = False,
    output_csv: Path = DEFAULT_RESULTS_CSV,
    output_json: Path = DEFAULT_RESULTS_JSON,
) -> Dict[str, Any]:
    """Run comparative evaluation between Naive RAG and Corrective RAG."""
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    if limit is not None and limit > 0:
        if balanced:
            by_type: Dict[str, List[Dict[str, Any]]] = {}
            for item in eval_items:
                t = item.get("type", "unknown")
                by_type.setdefault(t, []).append(item)
            selected = []
            types = list(by_type.keys())
            per_type = max(1, limit // len(types))
            for t in types:
                selected.extend(by_type[t][:per_type])
            for item in eval_items:
                if len(selected) >= limit:
                    break
                if item not in selected:
                    selected.append(item)
            eval_items = selected[:limit]
        else:
            eval_items = eval_items[:limit]

    ragas_llm, ragas_embeddings = setup_ragas_evaluator()
    naive_records, corrective_records, per_query_logs = execute_pipeline_runs(eval_items)

    # Compute RAGAS scores
    naive_scores = compute_ragas_scores(naive_records, ragas_llm, ragas_embeddings, "Naive RAG")
    corrective_scores = compute_ragas_scores(corrective_records, ragas_llm, ragas_embeddings, "Corrective RAG")

    comparison_df = pd.DataFrame([
        {
            "Metric": "Faithfulness",
            "Naive RAG": naive_scores["faithfulness"],
            "Corrective RAG": corrective_scores["faithfulness"],
            "Delta": corrective_scores["faithfulness"] - naive_scores["faithfulness"],
        },
        {
            "Metric": "Answer Relevancy",
            "Naive RAG": naive_scores["answer_relevancy"],
            "Corrective RAG": corrective_scores["answer_relevancy"],
            "Delta": corrective_scores["answer_relevancy"] - naive_scores["answer_relevancy"],
        },
        {
            "Metric": "Context Precision",
            "Naive RAG": naive_scores["context_precision"],
            "Corrective RAG": corrective_scores["context_precision"],
            "Delta": corrective_scores["context_precision"] - naive_scores["context_precision"],
        },
        {
            "Metric": "Context Recall",
            "Naive RAG": naive_scores["context_recall"],
            "Corrective RAG": corrective_scores["context_recall"],
            "Delta": corrective_scores["context_recall"] - naive_scores["context_recall"],
        },
    ])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_csv, index=False)

    summary = {
        "num_eval_samples": len(eval_items),
        "naive_rag": naive_scores,
        "corrective_rag": corrective_scores,
        "metrics_table": comparison_df.to_dict(orient="records"),
        "per_query_results": per_query_logs,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("      RAGAS BENCHMARK EVALUATION RESULTS (SIDE-BY-SIDE)")
    print("=" * 65)
    print(comparison_df.to_string(index=False))
    print("=" * 65 + "\n")

    # Assert Corrective RAG faithfulness >= Naive RAG baseline
    assert corrective_scores["faithfulness"] >= naive_scores["faithfulness"], (
        f"Assertion Failed: Corrective faithfulness ({corrective_scores['faithfulness']:.4f}) "
        f"< Naive baseline ({naive_scores['faithfulness']:.4f})"
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Harness for Corrective RAG")
    parser.add_argument("--eval-set", type=str, default=str(DEFAULT_EVAL_SET_PATH), help="Path to eval_set.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval items (default: all 50)")
    parser.add_argument("--balanced", action="store_true", default=False, help="Sample evenly across question types")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_RESULTS_CSV), help="Output CSV path")
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_RESULTS_JSON), help="Output JSON path")
    args = parser.parse_args()

    run_evaluation(
        eval_set_path=Path(args.eval_set),
        limit=args.limit,
        balanced=args.balanced,
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
    )


if __name__ == "__main__":
    main()
