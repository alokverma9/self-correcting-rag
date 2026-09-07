import json
from pathlib import Path
import pytest
import pandas as pd

from eval.run_eval import run_evaluation, DEFAULT_EVAL_SET_PATH


def test_eval_set_exactly_50_items_and_schema():
    # Hard requirement: exactly 50 hand-labeled questions
    assert DEFAULT_EVAL_SET_PATH.exists(), f"Eval set not found at {DEFAULT_EVAL_SET_PATH}"
    
    with open(DEFAULT_EVAL_SET_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    assert len(items) == 50, f"Expected exactly 50 evaluation items, got {len(items)}"

    types = {"in_corpus": 0, "ambiguous": 0, "out_of_corpus": 0}
    for item in items:
        assert "question" in item and len(item["question"].strip()) > 0
        assert "ground_truth" in item and len(item["ground_truth"].strip()) > 0
        assert "type" in item and item["type"] in types
        types[item["type"]] += 1

    assert types["in_corpus"] == 30, f"Expected 30 in_corpus, got {types['in_corpus']}"
    assert types["ambiguous"] == 10, f"Expected 10 ambiguous, got {types['ambiguous']}"
    assert types["out_of_corpus"] == 10, f"Expected 10 out_of_corpus, got {types['out_of_corpus']}"


def test_eval_runner_produces_csv_and_asserts_faithfulness(tmp_path):
    csv_out = tmp_path / "test_results.csv"
    json_out = tmp_path / "test_results.json"

    # Run on sample of 2 queries for test speed
    summary = run_evaluation(
        eval_set_path=DEFAULT_EVAL_SET_PATH,
        limit=2,
        output_csv=csv_out,
        output_json=json_out,
    )

    # 1. Output files exist
    assert csv_out.exists(), "Results CSV was not created"
    assert json_out.exists(), "Results JSON was not created"

    # 2. Side-by-side comparison CSV contains required metrics
    df = pd.read_csv(csv_out)
    metrics_present = set(df["Metric"].tolist())
    required_metrics = {"Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"}
    assert required_metrics.issubset(metrics_present), f"Missing metrics in CSV: {required_metrics - metrics_present}"

    # 3. Faithfulness asserted >= baseline
    assert summary["corrective_rag"]["faithfulness"] >= summary["naive_rag"]["faithfulness"]
