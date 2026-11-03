#!/usr/bin/env python3
"""
Run benchmark on EIC2023 dataset using pure RAG (vector search only).

This script:
- Loads benchmark questions from benchmark.jsonl
- Uses FAISS vectorstore to retrieve context (top-k similarity search)
- Uses LLM with reporter-style prompt to generate answers
- Evaluates using ragas metrics: faithfulness, answer_relevancy, answer_correctness

Usage:
    python run_benchmark_retriever.py [--max-questions N] [--with-ragas] [--top-k K]

    --max-questions N: Only run first N questions (default: all)
    --with-ragas: Run ragas evaluation
    --top-k K: Number of documents to retrieve (default: 5)
"""

import json
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from langchain_core.messages import HumanMessage, SystemMessage

from moderag.models.llm import usage_scope

# Shared benchmark utilities (question loading, results IO, ragas evaluation)
from ragas_eval import (
    load_benchmark_questions,
    calculate_avg_scores,
    run_ragas_eval_with_skip,
)
import results_file


# Configuration (shared with run_benchmark.py; see paths.py)
from paths import BENCHMARK_FILE, BENCH_ROOT, INDEXES_DIR, OUTPUT_DIR  # noqa: E402

# The vector baseline is one configuration in the shared results file.
RAG_CONFIG = "rag"


# Reporter-style concise prompt (from reporter.py)
REPORTER_SYSTEM_PROMPT = """You are a report generation expert. Your task is to generate structured answers based on retrieved evidence."""

REPORTER_CONCISE_PROMPT = """You are an expert at extracting key information from retrieved scientific documents.

Query: {query}

Retrieved Context:
{evidence_context}

Instructions:
1. Carefully analyze ALL retrieved context fragments to find relevant information
2. Look for conclusions, summaries, and key findings - they often appear in section endings or "Conclusions" sections
3. Extract and synthesize the key points that directly answer the query
4. Technical content, formulas, and equations ARE valid evidence - use them to answer
5. If multiple context fragments contain complementary information, combine them
6. **CRITICAL - Numeric Values**: When the query asks for specific numeric values (e.g., "lowest transverse momentum", "dimension", "range"), ONLY use values that are EXPLICITLY stated in the retrieved context. Do NOT infer or assume values from unrelated contexts.
7. **CRITICAL - Source Attribution**: For each numeric value in your answer, verify it comes from the retrieved context. If a value appears in one context but the query is about a different entity, do NOT use that value.
8. **CRITICAL - Context Relevance**: Vector Search Results are typically more reliable for specific numeric queries. Graph-Based Results may contain information from multiple documents - verify each value is relevant to the specific query.
9. Do not include citations in the output
10. Keep the answer concise but complete (3-6 sentences)
11. Only state "no relevant information" if you have thoroughly checked ALL context and found nothing

Answer:
"""


def load_vectorstore(storage_path: str):
    """Load FAISS vectorstore from storage path."""
    from langchain_community.vectorstores import FAISS
    from moderag.models.embedding import get_embedding

    # Load embedding model (wrapped=False for FAISS compatibility)
    # device falls back to cpu (no GPU on this machine)
    import os
    embedding = get_embedding(model="BAAI/bge-m3", device=os.getenv("EMBEDDING_DEVICE", "cpu"), wrapped=False)

    # Load vectorstore (FAISS format in directory)
    vectorstore_path = Path(storage_path) / "vectorstore"
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embedding,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def generate_answer(query: str, retrieved_docs: List) -> str:
    """Generate answer using LLM with reporter-style prompt."""
    from moderag.models.llm import get_llm

    llm = get_llm(temperature=0)

    # Build evidence context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        metadata_str = ", ".join(f"{k}={v}" for k, v in doc.metadata.items() if k not in ["retrieval_type", "source"])
        context_parts.append(f"[Document {i}] ({metadata_str}):\n{doc.page_content}")

    evidence_context = "\n\n".join(context_parts)

    # Generate prompt
    prompt = REPORTER_CONCISE_PROMPT.format(
        query=query,
        evidence_context=evidence_context
    )

    messages = [
        SystemMessage(content=REPORTER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content if hasattr(response, 'content') else str(response)
    return answer.strip()


def run_query(vectorstore, query: str, recall_top_k: int, final_top_k: int, use_reranker: bool, reranker=None) -> tuple:
    """Run a single query using vector search with optional reranking.

    Flow:
    1. Recall: Retrieve recall_top_k documents (e.g., 20)
    2. Rerank (optional): Re-rank recalled documents
    3. Return final_top_k documents (e.g., 5)

    Args:
        reranker: Pre-created reranker instance (shared across queries to save memory)
    """
    from moderag.rag.retriever import Retriever
    import torch

    # Create retriever
    retriever = Retriever(
        vectorstore=vectorstore,
        reranker=reranker,
        top_k=final_top_k,
        recall_top_k=recall_top_k,
        rerank_top_k=final_top_k,
    )

    # Vector search (with optional reranking)
    docs = retriever.vector_search(query)

    # Clear GPU cache after each query if using reranker
    if use_reranker and reranker is not None:
        torch.cuda.empty_cache()

    # Generate answer
    answer = generate_answer(query, docs)

    return answer, docs


def persist_rag(results: List[Dict[str, Any]], args) -> None:
    """Upsert the vector baseline into the benchmark's single results file."""
    n_docs = [r.get("num_docs_retrieved", 0) or 0 for r in results]
    meta = {
        "retrieval_mode": "rag_with_reranker" if args.use_reranker else "rag_no_reranker",
        "recall_top_k": args.recall_top_k,
        "final_top_k": args.top_k,
        "use_reranker": bool(args.use_reranker),
        "avg_docs_retrieved": (sum(n_docs) / len(n_docs)) if n_docs else 0.0,
        "nan_scores": results_file.nan_report(results),
    }
    path = OUTPUT_DIR / results_file.FILENAME
    results_file.upsert(path, RAG_CONFIG, results,
                        benchmark=BENCH_ROOT.name,
                        question_set=BENCHMARK_FILE.name,
                        config_meta=meta)
    print(f"Results written to {path}")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run EIC2023 benchmark using pure RAG (vector search)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Number of questions to process (default: all)")
    parser.add_argument("--questions-file", type=Path, default=BENCHMARK_FILE,
                        help=f"Question set jsonl (default: {BENCHMARK_FILE})")
    parser.add_argument("--with-ragas", action="store_true",
                        help="Run ragas evaluation")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of documents to retrieve (final, default: 5)")
    parser.add_argument("--recall-top-k", type=int, default=20,
                        help="Number of documents to recall in first stage (default: 20)")
    parser.add_argument("--use-reranker", action="store_true",
                        help="Use reranker for re-ranking (recall 20 -> rerank -> top 5)")
    parser.add_argument("--reranker-device", type=str, default="cuda:0",
                        help="Device for reranker model (default: cuda:0)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run ragas evaluation on existing results")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Force regenerate answers even if results file exists")
    args = parser.parse_args()

    print("=" * 60)
    print(f"EIC2023 Benchmark Runner (Pure RAG - Vector Search)")
    print(f"Recall top-k: {args.recall_top_k}, Final top-k: {args.top_k}")
    print(f"Use reranker: {args.use_reranker}")
    if args.use_reranker:
        print(f"Reranker device: {args.reranker_device}")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # The vector baseline is one configuration of the shared results file
    results_path = OUTPUT_DIR / results_file.FILENAME
    _, existing_all = results_file.read(results_path)
    existing = results_file.run_rows(existing_all, RAG_CONFIG)

    # Check if this configuration already ran (skip answer generation)
    if not args.eval_only and not args.force_regenerate and existing:
        print(f"\nConfiguration '{RAG_CONFIG}' already present in {results_path}")
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
            if args.use_reranker:
                retrieval_mode = f"rag_with_reranker_recall{args.recall_top_k}"
            else:
                retrieval_mode = "rag_no_reranker"
            run_ragas_eval_with_skip(results, eval_data, retrieval_mode,
                                     lambda rows: persist_rag(rows, args))
        else:
            print("\nUse --with-ragas to run/continue ragas evaluation")
            persist_rag(results, args)
        return

    # Eval-only mode: load existing results and run ragas
    if args.eval_only:
        if not existing:
            print(f"Error: configuration '{RAG_CONFIG}' not found in {results_path}")
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

        # Determine retrieval mode
        if args.use_reranker:
            retrieval_mode = f"rag_with_reranker_recall{args.recall_top_k}"
        else:
            retrieval_mode = "rag_no_reranker"

        # Run ragas evaluation with skip logic
        run_ragas_eval_with_skip(results, eval_data, retrieval_mode,
                                 lambda rows: persist_rag(rows, args))
        return

    # Load benchmark questions
    print(f"\nLoading benchmark questions from {args.questions_file}...")
    questions = load_benchmark_questions(args.questions_file, max_questions=args.max_questions)
    print(f"Loaded {len(questions)} questions")

    # Load vectorstore
    print(f"\nLoading vectorstore from {INDEXES_DIR}...")
    vectorstore = load_vectorstore(str(INDEXES_DIR))
    print("Vectorstore loaded successfully")

    # Create reranker once (shared across queries to save memory)
    reranker = None
    if args.use_reranker:
        from moderag.models.reranker import get_reranker
        print(f"\nLoading reranker model on {args.reranker_device}...")
        reranker = get_reranker(model="BAAI/bge-reranker-v2-m3", device=args.reranker_device, top_k=args.top_k)
        print("Reranker loaded successfully")

    results = []
    eval_data = []

    print(f"\nRunning queries...")
    for i, q in enumerate(tqdm(questions, desc="Processing")):
        arxiv_id = q.get("arxiv_id", f"unknown_{i}")
        question_text = q["question"]
        reference_answer = q.get("reference_answer", "")

        print(f"\n[{i+1}/{len(questions)}] Processing: {arxiv_id}")
        print(f"Question: {question_text[:100]}...")

        t0 = time.perf_counter()
        usage = None
        try:
            with usage_scope() as usage:
                answer, docs = run_query(
                    vectorstore, question_text,
                    args.recall_top_k, args.top_k,
                    args.use_reranker, reranker
                )
            elapsed_s = time.perf_counter() - t0
            usage_stats = usage.to_dict() if usage else {}

            # Build retrieved context
            retrieved_context = [doc.page_content for doc in docs]

            # Determine retrieval mode name
            if args.use_reranker:
                retrieval_mode = f"rag_with_reranker_recall{args.recall_top_k}"
            else:
                retrieval_mode = "rag_no_reranker"

            result_record = {
                "arxiv_id": arxiv_id,
                "question": question_text,
                "reference_answer": reference_answer,
                "generated_answer": answer,
                "retrieved_context": retrieved_context,
                "retrieval_mode": retrieval_mode,
                "recall_top_k": args.recall_top_k,
                "final_top_k": args.top_k,
                "use_reranker": args.use_reranker,
                "num_docs_retrieved": len(docs),
                "statistics": {"time_s": round(elapsed_s, 3), **usage_stats},
                "success": True,
                "error": None
            }
        except Exception as e:
            elapsed_s = time.perf_counter() - t0
            usage_stats = usage.to_dict() if usage is not None else {}
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            if args.use_reranker:
                retrieval_mode = f"rag_with_reranker_recall{args.recall_top_k}"
            else:
                retrieval_mode = "rag_no_reranker"
            result_record = {
                "arxiv_id": arxiv_id,
                "question": question_text,
                "reference_answer": reference_answer,
                "generated_answer": "",
                "retrieved_context": [],
                "retrieval_mode": retrieval_mode,
                "recall_top_k": args.recall_top_k,
                "final_top_k": args.top_k,
                "use_reranker": args.use_reranker,
                "num_docs_retrieved": 0,
                "statistics": {"time_s": round(elapsed_s, 3), **usage_stats},
                "success": False,
                "error": str(e)
            }

        results.append(result_record)
        eval_data.append({
            "question": question_text,
            "answer": result_record["generated_answer"],
            "retrieved_contexts": result_record["retrieved_context"] or ["No context retrieved"],
            "ground_truth": reference_answer
        })

    # Persist the answers first, so a failed or skipped ragas pass cannot lose
    # them; the evaluation upserts the same configuration again with scores.
    persist_rag(results, args)

    # Evaluate with ragas if requested
    if args.with_ragas and eval_data and any(d["answer"] for d in eval_data):
        # Determine retrieval mode for ragas evaluation
        if args.use_reranker:
            retrieval_mode = f"rag_with_reranker_recall{args.recall_top_k}"
        else:
            retrieval_mode = "rag_no_reranker"
        # Run ragas evaluation with skip logic
        run_ragas_eval_with_skip(results, eval_data, retrieval_mode,
                                 lambda rows: persist_rag(rows, args))
    else:
        if args.with_ragas:
            print("\nNo valid answers to evaluate with ragas")
        else:
            print("\nRagas evaluation skipped (run again with --with-ragas to add scores)")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print(f"Successful queries: {sum(1 for r in results if r.get('success', False))}/{len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# --- keep the vanilla-RAG baseline answer prompt identical to the agentic reporter ---
# run_benchmark_rag.py used to carry its own copy of the reporter prompt, which
# silently drifted from moderag/agents/reporter.py (the partial-answer /
# "scan all context" rule was missing).  Import it so both paths stay in sync.
from moderag.agents.reporter import (  # noqa: E402
    REPORTER_SYSTEM_PROMPT as _REPORTER_SYSTEM_PROMPT,
    REPORTER_CONCISE_PROMPT as _REPORTER_CONCISE_PROMPT,
)

REPORTER_SYSTEM_PROMPT = _REPORTER_SYSTEM_PROMPT
REPORTER_CONCISE_PROMPT = _REPORTER_CONCISE_PROMPT
