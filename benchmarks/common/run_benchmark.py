#!/usr/bin/env python3
"""
Run the multi-agent configurations on a benchmark, with ragas metrics.

Features:
- Load benchmark questions from <BENCH_ROOT>/benchmark.jsonl (or --questions-file)
- Use Orchestrator in concise mode to generate answers
- Evaluate using ragas metrics: faithfulness, answer_relevancy, answer_correctness
- Save retrieved context, subquery count, batch parallelism

Usage:
    BENCH_ROOT=<benchmark dir> python run_benchmark.py --tool-config all [--with-ragas]

    --max-questions N: Only run first N questions (default: all)
    --tool-config C: any subset of naive/local/global (e.g. naive+local), or all
    --questions-file F: question set (default: <BENCH_ROOT>/benchmark.jsonl)
    --with-ragas: Run ragas evaluation (requires OPENAI_API_KEY)
    BENCH_OUTPUT_DIR: write results somewhere other than <BENCH_ROOT>/results
"""

import json
import os
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

# Set environment variables for storage paths BEFORE importing moderag modules
from paths import BENCHMARK_FILE, BENCH_ROOT, INDEXES_DIR, OUTPUT_DIR  # noqa: E402

os.environ["GRAPH_RAG_STORAGE_PATH"] = str(INDEXES_DIR)

from moderag.agents.orchestrator import Orchestrator, OrchestratorResult
from moderag.agents.reporter import ReportMode
from moderag.agents.executor import Executor
from moderag.agents.tools.registry import ToolRegistry, SearchModeTool
from moderag.models.llm import usage_scope

# Shared benchmark utilities (question loading, results IO, ragas evaluation)
from ragas_eval import (
    load_benchmark_questions,
    calculate_avg_scores,
    run_ragas_eval_with_skip,
)
import results_file


def run_query(orchestrator, question: str) -> OrchestratorResult:
    """Run a single query through the orchestrator."""
    return orchestrator.process_query(question)


def persist(results: List[Dict[str, Any]], tool_config: str) -> None:
    """Upsert this configuration into the benchmark's single results file."""
    path = OUTPUT_DIR / results_file.FILENAME
    results_file.upsert(
        path, tool_config, results,
        benchmark=BENCH_ROOT.name,
        question_set=str(BENCHMARK_FILE.name),
        config_meta={"nan_scores": results_file.nan_report(results)},
    )
    print(f"Results written to {path}")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run EIC2023 benchmark")
    parser.add_argument("--max-questions", type=int, default=50,
                        help="Number of questions to process (default: 50)")
    parser.add_argument("--questions-file", type=Path, default=BENCHMARK_FILE,
                        help=f"Question set jsonl (default: {BENCHMARK_FILE})")
    parser.add_argument("--with-ragas", action="store_true",
                        help="Run ragas evaluation (requires OPENAI_API_KEY)")
    parser.add_argument("--tool-config", type=str, default="all",
                        choices=["all", "naive", "local", "global",
                                 "naive+local", "naive+global", "local+global"],
                        help="Tool set exposed to the Planner (default: all)")
    parser.add_argument("--kv", type=int, default=3,
                        help="Entity recall depth for local mode, k_v (default: 3)")
    parser.add_argument("--kr", type=int, default=10,
                        help="Relationship recall depth for local/global modes, k_r (default: 10)")
    parser.add_argument("--cap", type=int, default=10,
                        help="Per-mode chunk-collection cap before final selection (default: 10)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Maximum parallel workers (default: 4)")
    parser.add_argument("--outer-workers", type=int, default=4,
                        help="Parallel questions processed concurrently across the question loop "
                             "(default: 4; pass 1 to run questions strictly sequentially)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run ragas evaluation on existing results")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Force regenerate answers even if results file exists")
    args = parser.parse_args()

    print("=" * 60)
    print(f"EIC2023 Benchmark Runner (Concise Mode)")
    print(f"Tool Config: {args.tool_config}")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / results_file.FILENAME
    _, existing_all = results_file.read(results_path)
    existing = results_file.run_rows(existing_all, results_file.canonical_config(args.tool_config))

    # Check if this configuration already ran (skip answer generation)
    if not args.eval_only and not args.force_regenerate and existing:
        print(f"\nConfiguration '{args.tool_config}' already present in {results_path}")
        print("Skipping answer generation, loading existing results...")
        results = existing
        print(f"Loaded {len(results)} existing results")

        # Prepare eval data from existing results
        eval_data = []
        for record in results:
            eval_data.append({
                "question": record["question"],
                "answer": record.get("generated_answer", ""),
                "retrieved_contexts": record.get("retrieved_context", []) or ["No context retrieved"],
                "ground_truth": record.get("reference_answer", "")
            })

        # Run ragas evaluation (will skip valid scores internally)
        if args.with_ragas:
            run_ragas_eval_with_skip(results, eval_data, args.tool_config,
                                     lambda rows: persist(rows, args.tool_config))
        else:
            print("\nUse --with-ragas to run/continue ragas evaluation")

        return

    # Eval-only mode: load existing results and run ragas
    if args.eval_only:
        if not existing:
            print(f"Error: configuration '{args.tool_config}' not found in {results_path}")
            return

        print(f"\nLoading existing results from {results_path}...")
        results = existing
        print(f"Loaded {len(results)} results")

        # Prepare eval data
        eval_data = []
        for record in results:
            eval_data.append({
                "question": record["question"],
                "answer": record.get("generated_answer", ""),
                "retrieved_contexts": record.get("retrieved_context", []) or ["No context retrieved"],
                "ground_truth": record.get("reference_answer", "")
            })

        # Run ragas evaluation with skip logic
        run_ragas_eval_with_skip(results, eval_data, args.tool_config,
                                 lambda rows: persist(rows, args.tool_config))
        return

    # Load benchmark questions
    print(f"\nLoading benchmark questions from {args.questions_file}...")
    questions = load_benchmark_questions(args.questions_file, max_questions=args.max_questions)
    print(f"Loaded {len(questions)} questions")

    # Get storage paths
    graphrag_storage_path = os.environ.get("GRAPH_RAG_STORAGE_PATH", str(INDEXES_DIR))
    print(f"\nGraphRAG storage path: {graphrag_storage_path}")

    # Create executor with storage path
    print(f"\nInitializing orchestrator (concise mode)...")
    executor = Executor(max_parallel_workers=args.workers)

    # Register tools based on tool-config. Retrieval hyper-parameters are
    # injected through the tool constructor (benchmark-harness values k_v, k_r,
    # and the per-mode chunk-collection cap), so ablation runs go through the
    # same code path as any GraphRAGTool usage (no monkey-patching).
    ToolRegistry.unregister("naive")
    ToolRegistry.unregister("local")
    ToolRegistry.unregister("global")

    # Every subset of the three retrieval modes, so the ablation can vary one
    # component at a time; "all" is the full system.
    tool_map = {
        "all": ["naive", "local", "global"],
        "naive": ["naive"],
        "local": ["local"],
        "global": ["global"],
        "naive+local": ["naive", "local"],
        "naive+global": ["naive", "global"],
        "local+global": ["local", "global"],
    }

    # Create factory functions for each tool mode
    def make_tool_factory(mode: str):
        def tool_factory(**kwargs):
            return SearchModeTool(
                mode=mode,
                storage_path=graphrag_storage_path,
                top_k_entities=args.kv,
                top_k_relationships=args.kr,
                max_recall_chunks=args.cap,
            )
        return tool_factory

    # Register factory functions for selected modes
    for name in tool_map.get(args.tool_config, []):
        ToolRegistry.register(name, make_tool_factory(name))

    orchestrator = Orchestrator(
        executor=executor,
        report_mode=ReportMode.CONCISE
    )

    def make_result_record(arxiv_id: str, question: str, reference_answer: str,
                           answer: str, retrieved_context: list, plan: dict = None,
                           stats: dict = None, success: bool = True, error: str = None) -> dict:
        """Build a result record."""
        return {
            "arxiv_id": arxiv_id, "question": question, "reference_answer": reference_answer,
            "generated_answer": answer, "retrieved_context": retrieved_context,
            "plan": plan, "statistics": stats or {}, "success": success, "error": error
        }

    results: List[Dict[str, Any]] = [None] * len(questions)
    eval_data: List[Dict[str, Any]] = [None] * len(questions)
    n_questions = len(questions)
    print_lock = threading.Lock()

    print(f"\nRunning queries (outer_workers={args.outer_workers})...")
    pbar = tqdm(total=n_questions, desc="Processing")

    def process_one(i: int, q: Dict[str, Any]):
        arxiv_id = q.get("arxiv_id", f"unknown_{i}")
        question_text = q["question"]
        reference_answer = q.get("reference_answer", "")

        with print_lock:
            print(f"\n[{i+1}/{n_questions}] Processing: {arxiv_id}")
            print(f"Question: {question_text[:100]}...")

        t0 = time.perf_counter()
        usage = None
        try:
            with usage_scope() as usage:
                result = run_query(orchestrator, question_text)
            elapsed_s = time.perf_counter() - t0
            usage_stats = usage.to_dict() if usage else {}

            if result.success:
                retrieved_context = []
                if result.report and result.report.structured_evidence:
                    for ev in result.report.structured_evidence.get("top_evidence", []):
                        retrieved_context.append(ev.get("content", "") if isinstance(ev, dict) else ev)

                result_record = make_result_record(
                    arxiv_id, question_text, reference_answer, result.answer,
                    retrieved_context, result.plan.to_dict() if result.plan else None,
                    {"num_subqueries": result.num_subqueries, "max_parallelism": result.max_parallelism,
                     "num_batches": result.num_batches, "planner_tokens": result.planner_tokens,
                     "reporter_tokens": result.reporter_tokens, "time_s": round(elapsed_s, 3),
                     **usage_stats}, True, None
                )
            else:
                result_record = make_result_record(arxiv_id, question_text, reference_answer, "", [],
                                                   None, {"time_s": round(elapsed_s, 3), **usage_stats},
                                                   False, result.error)
        except Exception as e:
            elapsed_s = time.perf_counter() - t0
            usage_stats = usage.to_dict() if usage is not None else {}
            with print_lock:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            result_record = make_result_record(arxiv_id, question_text, reference_answer, "", [],
                                               None, {"time_s": round(elapsed_s, 3), **usage_stats},
                                               False, str(e))

        return i, result_record

    def store_result(i: int, result_record: Dict[str, Any]) -> None:
        results[i] = result_record
        eval_data[i] = {
            "question": result_record["question"],
            "answer": result_record.get("generated_answer", ""),
            "retrieved_contexts": result_record.get("retrieved_context", []) or ["No context retrieved"],
            "ground_truth": questions[i].get("reference_answer", "")
        }

    if args.outer_workers <= 1:
        for i, q in enumerate(questions):
            idx, result_record = process_one(i, q)
            store_result(idx, result_record)
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=args.outer_workers) as pool:
            futures = [pool.submit(process_one, i, q) for i, q in enumerate(questions)]
            for future in as_completed(futures):
                idx, result_record = future.result()
                store_result(idx, result_record)
                pbar.update(1)
    pbar.close()

    # Verify all questions produced a record (parallel completion is unordered).
    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        raise RuntimeError(f"{len(missing)} questions produced no result: {missing[:10]}")

    # Persist the answers first, so a failed or skipped ragas pass cannot lose
    # them; the evaluation upserts the same configuration again with scores.
    persist(results, args.tool_config)

    # Evaluate with ragas if requested
    if args.with_ragas and eval_data and any(d["answer"] for d in eval_data):
        run_ragas_eval_with_skip(results, eval_data, args.tool_config,
                                 lambda rows: persist(rows, args.tool_config))
    elif args.with_ragas:
        print("\nNo valid answers to evaluate with ragas")
    else:
        print("\nRagas evaluation skipped (run again with --with-ragas to add scores)")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print(f"Successful queries: {sum(1 for r in results if r['success'])}/{len(questions)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
