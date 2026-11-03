#!/usr/bin/env python3
"""Paired per-question bootstrap for the numbers reported in the paper.

Every reported gap is a mean over *questions*, so the unit of inference is the
question rather than the LLM call. For each comparison we take the paired
per-question difference

    d_i = score_i(config A) - score_i(config B),      i = 1..n

and resample the d_i with replacement `--resamples` times, recomputing the mean
each time. The reported interval is the percentile interval [alpha/2, 1-alpha/2]
of those means. Resampling paired differences (rather than the two marginal
means) cancels question difficulty, which is the dominant source of variance.

Usage:
    python analysis/bootstrap_ci.py                    # defaults: 10000 resamples
    python analysis/bootstrap_ci.py --resamples 50000 --seed 0

The output reproduces Table `tab:noise` and the genotype rows of
Table `tab:querytype`. It also prints the standard deviation / standard error of
each difference, which is where the paper's +-0.05 (LHC) and +-0.055 (AI4EIC)
noise bands come from.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import results_file  # noqa: E402

METRIC = "answer_correctness"

# (label, benchmark directory, question-set file or None)
BENCHMARKS = [
    ("LHCpublications", ROOT / "LHCpubs_run2", ROOT / "LHCpubs_run2" / "benchmark.jsonl"),
    ("AI4EIC2023", ROOT / "AI4EIC2023", None),
]

# (comparison label, A suffix, B suffix)
COMPARISONS = [
    ("All Modes - Naive Only", "all", "naive"),
    ("All Modes - legacy RAG", "all", "rag"),
]


def load_scores(bench_dir: Path, config: str) -> dict:
    """{question: ragas_scores} for one configuration of one benchmark."""
    path = bench_dir / "results" / results_file.FILENAME
    _, questions = results_file.read(path)
    out: dict[str, dict] = {}
    for question, record in questions.items():
        run = (record.get("runs") or {}).get(config)
        if run is not None:
            out[question] = run.get("ragas_scores") or {}
    return out


def score(scores: dict, metric: str = METRIC):
    value = scores.get(metric)
    if value is None or value != value:  # None or NaN
        return None
    return float(value)


def paired_diffs(a: dict, b: dict, metric: str, questions=None) -> list[float]:
    keys = questions if questions is not None else a.keys()
    diffs = []
    for q in keys:
        if q not in a or q not in b:
            continue
        x, y = score(a[q], metric), score(b[q], metric)
        if x is not None and y is not None:
            diffs.append(x - y)
    return diffs


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile, matching numpy's default."""
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(diffs: list[float], resamples: int, alpha: float, seed: int):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return percentile(means, alpha / 2), percentile(means, 1 - alpha / 2)


def report(name: str, diffs: list[float], resamples: int, alpha: float, seed: int) -> None:
    n = len(diffs)
    if n == 0:
        print(f"  {name:26s} (no paired questions)")
        return
    delta = sum(diffs) / n
    lo, hi = bootstrap_ci(diffs, resamples, alpha, seed)
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    flag = "resolved" if lo > 0 or hi < 0 else "unresolved (CI contains 0)"
    print(
        f"  {name:26s} d={delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
        f"n={n:3d}  W/L={wins}/{losses}  sd={sd:.3f} se={se:.3f}  {flag}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--metric", default=METRIC)
    args = ap.parse_args()

    for label, bench_dir, question_file in BENCHMARKS:
        print(f"\n=== {label}  ({(bench_dir / 'results').relative_to(ROOT)}) ===")
        if not (bench_dir / "results" / results_file.FILENAME).is_file():
            print("  results file missing")
            continue

        pools = {}
        for _, a_cfg, b_cfg in COMPARISONS:
            pools.setdefault(a_cfg, load_scores(bench_dir, a_cfg))
            pools.setdefault(b_cfg, load_scores(bench_dir, b_cfg))

        for name, a_cfg, b_cfg in COMPARISONS:
            diffs = paired_diffs(pools[a_cfg], pools[b_cfg], args.metric)
            report(name, diffs, args.resamples, args.alpha, args.seed)

        if question_file and question_file.is_file():
            types = {}
            for line in open(question_file):
                if line.strip():
                    row = json.loads(line)
                    types[row["question"]] = row.get("type")
            print(f"  --- by genotype ({question_file.name}) ---")
            all_pool, naive_pool = pools["all"], pools["naive"]
            for genotype in sorted({t for t in types.values() if t}):
                questions = [q for q, t in types.items() if t == genotype]
                diffs = paired_diffs(all_pool, naive_pool, args.metric, questions)
                report(genotype, diffs, args.resamples, args.alpha, args.seed)


if __name__ == "__main__":
    main()
