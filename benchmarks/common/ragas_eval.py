#!/usr/bin/env python3
"""
Shared benchmark utilities for every benchmark runner.

run_benchmark.py (multi-agent) and run_benchmark_rag.py (RAG baseline)
generate answers with different retrieval stacks but share the same data I/O
and the same ragas evaluation pipeline.  Everything evaluation-related lives
here so the two runners cannot drift apart.

ragas 0.2.x API: metrics are configured with the evaluation LLM/embeddings
passed to evaluate().
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# BENCH_OUTPUT_DIR overrides the results directory, so a new run can write to its
# own directory and leave the archived results untouched. See paths.py.
from paths import OUTPUT_DIR  # noqa: E402

METRICS = ["faithfulness", "answer_relevancy", "answer_correctness"]

# RAGAS evaluation concurrency / timeout / retry settings, configurable through
# the environment (used to relax them for long benchmark runs).
RAGAS_TIMEOUT_SEC = float(os.environ.get("RAGAS_TIMEOUT_SEC", "300"))
RAGAS_MAX_RETRIES = int(os.environ.get("RAGAS_MAX_RETRIES", "5"))
RAGAS_MAX_WORKERS = int(os.environ.get("RAGAS_MAX_WORKERS", "2"))


def load_benchmark_questions(file_path: Path, max_questions: int = None) -> List[Dict[str, Any]]:
    """Load benchmark questions from a JSONL file."""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_questions and i >= max_questions:
                break
            question = json.loads(line.strip())
            questions.append(question)
    return questions


def is_nan(value) -> bool:
    """Check whether a ragas score is NaN/None."""
    if value is None:
        return True
    try:
        return float(value) != float(value)  # NaN check
    except (TypeError, ValueError):
        return True


def has_valid_scores(record: Dict[str, Any]) -> bool:
    """Check if a result record has valid (non-NaN) ragas scores."""
    if "ragas_scores" not in record:
        return False
    scores = record["ragas_scores"]
    return not any(is_nan(scores.get(m)) for m in METRICS)


def calculate_avg_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate average ragas scores from results, handling NaN values."""
    all_results_scores = [r["ragas_scores"] for r in results if "ragas_scores" in r]

    if not all_results_scores:
        return {m: 0 for m in METRICS}

    avg = {}
    for metric in METRICS:
        valid = [s[metric] for s in all_results_scores if s[metric] == s[metric]]
        avg[metric] = sum(valid) / len(valid) if valid else 0
    return avg


def _build_eval_data(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a result record into a ragas evaluation row."""
    return {
        "question": record["question"],
        "answer": record.get("generated_answer", ""),
        "retrieved_contexts": record.get("retrieved_context", []) or ["No context retrieved"],
        "ground_truth": record.get("reference_answer", "")
    }


def _run_ragas(eval_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, float]]]:
    """Run one ragas evaluation over the given rows.

    Returns one score dict per row (same order as eval_data) on success, or
    None if the evaluation failed entirely.
    """
    try:
        from ragas import evaluate, RunConfig
        from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
        from datasets import Dataset
        from moderag.models.llm import get_llm
        from moderag.models.embedding import get_embedding
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # Use temperature=0 for deterministic evaluation scores
        evaluation_llm = get_llm(temperature=0)
        # Wrap LangChain embedding for ragas compatibility
        evaluation_embedding = LangchainEmbeddingsWrapper(get_embedding(wrapped=False))

        print("\n" + "=" * 60)
        print(f"Running ragas evaluation on {len(eval_data)} records...")
        print(f"Configuration: timeout={RAGAS_TIMEOUT_SEC:.0f}, "
              f"max_retries={RAGAS_MAX_RETRIES}, max_workers={RAGAS_MAX_WORKERS}")
        print("=" * 60)

        dataset = Dataset.from_list(eval_data)
        score = evaluate(
            dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                AnswerCorrectness(embeddings=evaluation_embedding),
            ],
            llm=evaluation_llm,
            embeddings=evaluation_embedding,
            raise_exceptions=False,
            run_config=RunConfig(timeout=RAGAS_TIMEOUT_SEC, max_retries=RAGAS_MAX_RETRIES,
                                 max_workers=RAGAS_MAX_WORKERS),
        )

        # Extract per-question scores from score.scores (list of dicts)
        all_scores = []
        try:
            score_rows = score.scores if hasattr(score, 'scores') and isinstance(score.scores, list) else []
            for row in score_rows:
                if isinstance(row, dict):
                    all_scores.append({m: row.get(m, 0) for m in METRICS})
        except Exception as e:
            print(f"  Error extracting per-question scores: {e}")

        # Report average over the evaluated batch
        print("\nRagas Evaluation Scores (Averages over evaluated rows):")
        for metric in METRICS:
            valid = [s[metric] for s in all_scores if s[metric] == s[metric]]
            avg = sum(valid) / len(valid) if valid else 0
            print(f"  {metric}: {avg:.4f} ({len(valid)}/{len(all_scores)} valid)")

        return all_scores
    except Exception as e:
        print(f"Ragas evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _report_nan(results: List[Dict[str, Any]], tag: str) -> None:
    """Report questions the judge scored NaN.

    No separate file is written: the runner records these in this
    configuration's ``nan_scores`` entry of the results header, so a run
    produces one results file and nothing else.
    """
    nan_records = [r for r in results
                   if any(is_nan((r.get("ragas_scores") or {}).get(m)) for m in METRICS)]
    if nan_records:
        print(f"\n{len(nan_records)} record(s) of '{tag}' have a NaN metric; "
              f"recorded under nan_scores in the results file")
    else:
        print("\nNo NaN records found - all evaluations succeeded")


def run_ragas_eval_with_skip(results: List[Dict[str, Any]],
                             eval_data: List[Dict[str, Any]],
                             tag: str,
                             persist=None) -> Optional[Dict[str, float]]:
    """Ensure every result record carries a valid ragas_scores entry.

    Only records with missing/NaN scores are (re-)evaluated; records that
    already have valid scores are left untouched.  Records are updated in
    place.  On success the average scores over ALL records are returned
    (None if the evaluation itself failed).

    When an evaluation actually runs, ``persist(results)`` is called - the
    runner passes a callback that upserts this configuration into the shared
    results file - and any NaN metrics are reported (recorded by that callback
    under ``nan_scores`` in the results header, not in a file of their own).
    """
    total = len(results)
    valid_count = sum(1 for r in results if has_valid_scores(r))

    print(f"\nValid scores: {valid_count}/{total}")

    # If all have valid scores, skip evaluation
    if valid_count == total:
        print("All results already have valid ragas scores. Skipping evaluation.")
        return calculate_avg_scores(results)

    need_re_eval = [i for i, r in enumerate(results) if not has_valid_scores(r)]
    print(f"Need re-evaluation: {len(need_re_eval)}")

    subset_eval_data = [_build_eval_data(results[i]) for i in need_re_eval]
    print(f"\nRe-evaluating {len(subset_eval_data)} records with NaN/missing scores...")

    if not subset_eval_data:
        return calculate_avg_scores(results)

    all_scores = _run_ragas(subset_eval_data)
    if all_scores is None:
        return None

    # Update the records that were (re-)evaluated
    for pos, idx in enumerate(need_re_eval):
        if pos < len(all_scores):
            results[idx]["ragas_scores"] = all_scores[pos]
        else:
            results[idx]["ragas_scores"] = {m: 0 for m in METRICS}

    _report_nan(results, tag)

    if persist is not None:
        persist(results)
        print("\nResults updated with ragas scores")

    return calculate_avg_scores(results)
