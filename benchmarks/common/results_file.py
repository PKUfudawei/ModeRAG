"""One table, one file.

Every result set - the released scores and each archived run - is a single
``results.jsonl`` that holds the whole table:

    {"record": "header", "benchmark": ..., "question_set": ...,
     "metrics": [...], "configs": [...], "config_meta": {...},
     "summary": {config: {"n": N, <metric>: mean}}}
    {"record": "question", "question": ..., "type": ..., "reference_answer": ...,
     "runs": {config: {"generated_answer": ..., "retrieved_context": [...],
                       "ragas_scores": {<metric>: value}, ...}}}

The header line is the table's per-configuration mean row; each later line is one
question carrying every configuration's answer and scores for it. Configurations
are *upserted*, so running one configuration updates the shared file in place and
re-running anything converges on the same single artifact instead of adding
another ``results_<config>.jsonl`` beside it.

Only configurations that produced a score for a question appear in its ``runs``,
which is why the graph-only rows of the LHC benchmark cover 88 questions while
the others cover 100.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FILENAME = "results.jsonl"
METRICS = ["faithfulness", "answer_relevancy", "answer_correctness"]

# Per-question execution statistics averaged into the header. The paper's
# efficiency table reads these back (sub-tasks, batches, tokens, wall clock).
STAT_KEYS = [
    "num_subqueries", "max_parallelism", "num_batches",
    "planner_tokens", "reporter_tokens", "time_s",
    "llm_calls", "llm_attempts", "llm_prompt_tokens",
    "llm_completion_tokens", "llm_total_tokens", "llm_seconds",
]

# Archived runs wrote the vector baseline under a longer name; canonicalise it.
CONFIG_ALIASES = {
    "rag": "rag",
    "rag_no_reranker": "rag",
    "rag_with_reranker": "rag",
    "rag_no_reranker_recall20_topk5": "rag",
    "rag_reranker_recall20_topk5": "rag",
}


def canonical_config(name: str) -> str:
    return CONFIG_ALIASES.get(name, name)


def is_nan(value: Any) -> bool:
    if value is None:
        return True
    try:
        return float(value) != float(value)
    except (TypeError, ValueError):
        return True


def valid_scores(scores: Optional[Dict[str, Any]]) -> bool:
    """True when every metric has a non-NaN value."""
    if not scores:
        return False
    return not any(is_nan(scores.get(m)) for m in METRICS)


def nan_report(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Questions whose Ragas judge returned a NaN for at least one metric.

    These are recorded in the results header instead of in a separate
    ``nan_scores_<config>.json`` beside it: they are a property of this
    configuration's run, and the header is where a run's metadata belongs.
    """
    out = []
    for row in rows:
        scores = row.get("ragas_scores") or {}
        bad = [m for m in METRICS if is_nan(scores.get(m))]
        if bad:
            out.append({"question": row.get("question"), "nan_metrics": bad})
    return out


def summarise(questions: Dict[str, Dict[str, Any]],
              configs: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per-configuration mean row, NaN-safe, over the questions that have it."""
    out: Dict[str, Dict[str, float]] = {}
    for config in configs:
        values = {m: [] for m in METRICS}
        stats = {k: [] for k in STAT_KEYS}
        successes = 0
        runs_for_config = 0
        for record in questions.values():
            run = (record.get("runs") or {}).get(config)
            if run is None:
                continue
            runs_for_config += 1
            successes += 1 if run.get("success") else 0
            stats_row = run.get("statistics")
            if isinstance(stats_row, dict):
                for k in STAT_KEYS:
                    stats[k].append(float(stats_row.get(k, 0) or 0))
            scores = run.get("ragas_scores") or {}
            for m in METRICS:
                if not is_nan(scores.get(m)):
                    values[m].append(float(scores[m]))
        n = max((len(v) for v in values.values()), default=0)
        row = {
            "n": n,
            "runs": runs_for_config,
            "successful_queries": successes,
            "avg_statistics": {
                k: (sum(v) / len(v) if v else 0.0) for k, v in stats.items()
            },
        }
        for m in METRICS:
            row[m] = sum(values[m]) / len(values[m]) if values[m] else None
        out[config] = row
    return out


def read(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Load a results file -> (header, {question: record})."""
    header: Dict[str, Any] = {"record": "header", "configs": [], "metrics": METRICS}
    questions: Dict[str, Dict[str, Any]] = {}
    if not Path(path).exists():
        return header, questions
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record") == "header":
                header = row
            elif row.get("question") is not None:
                questions[row["question"]] = row
    header.setdefault("configs", sorted(
        {c for r in questions.values() for c in (r.get("runs") or {})}))
    return header, questions


def write(path: Path, header: Dict[str, Any],
          questions: Dict[str, Dict[str, Any]]) -> None:
    """Persist header + questions as one JSONL file, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = header.get("configs") or sorted(
        {c for r in questions.values() for c in (r.get("runs") or {})})
    header = dict(header)
    header.update(
        record="header",
        metrics=list(METRICS),
        configs=list(configs),
        summary=summarise(questions, configs),
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for record in questions.values():
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run_rows(questions: Dict[str, Dict[str, Any]], config: str) -> List[Dict[str, Any]]:
    """The rows this configuration produced, in question order."""
    rows = []
    for record in questions.values():
        run = (record.get("runs") or {}).get(config)
        if run is None:
            continue
        row = dict(run)
        row.setdefault("question", record["question"])
        if "reference_answer" in record and "reference_answer" not in row:
            row["reference_answer"] = record["reference_answer"]
        rows.append(row)
    return rows


def upsert(path: Path, config: str, rows: Iterable[Dict[str, Any]],
           benchmark: Optional[str] = None,
           question_set: Optional[str] = None,
           config_meta: Optional[Dict[str, Any]] = None,
           reset: bool = False) -> None:
    """Replace this configuration's rows in the shared results file."""
    config = canonical_config(config)
    path = Path(path)
    header, questions = ({"record": "header", "configs": []}, {}) if reset else read(path)
    if benchmark:
        header["benchmark"] = benchmark
    if question_set:
        header["question_set"] = question_set
    if config_meta:
        header.setdefault("config_meta", {}).setdefault(config, {}).update(config_meta)

    for row in rows:
        question = row["question"]
        record = questions.setdefault(question, {"record": "question", "question": question,
                                                 "runs": {}})
        for key in ("type", "reference_answer", "arxiv_id"):
            if key not in record and row.get(key) is not None:
                record[key] = row[key]
        record["runs"][config] = row

    configs = list(header.get("configs") or [])
    if config not in configs:
        configs.append(config)
    header["configs"] = configs
    write(path, header, questions)


def merge(paths: Sequence[Path], order: Optional[Iterable[str]] = None,
          benchmark: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Merge several result files; later files win per (question, configuration)."""
    header: Dict[str, Any] = {"record": "header", "configs": []}
    questions: Dict[str, Dict[str, Any]] = {}
    configs: List[str] = []
    for path in paths:
        part_header, part = read(Path(path))
        for config, meta in (part_header.get("config_meta") or {}).items():
            config = canonical_config(config)
            header.setdefault("config_meta", {}).setdefault(config, {}).update(meta)
        for question, record in part.items():
            target = questions.setdefault(question, {"record": "question", "question": question,
                                                     "runs": {}})
            for key, value in record.items():
                if key not in ("runs", "record", "question") and key not in target:
                    target[key] = value
            target["runs"].update(record.get("runs") or {})
            for config in (record.get("runs") or {}):
                if config not in configs:
                    configs.append(config)
    if order is not None:
        ordered = {q: questions[q] for q in order if q in questions}
        ordered.update({q: r for q, r in questions.items() if q not in ordered})
        questions = ordered
    header["configs"] = configs
    if benchmark:
        header["benchmark"] = benchmark
    return header, questions
