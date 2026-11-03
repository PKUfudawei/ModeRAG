#!/usr/bin/env python3
"""Generate LHCpubs composite-heavy benchmark questions (target N questions).

Evidence-first QA generation on top of `generate_qa.gen_for_cluster`:

1. Base round : for every cluster in the clusters file, ask the LLM for a fixed
   per-genotype count (BASE_COUNTS), composite-heavy by default.
2. Extra round : if the validated total falls short of --target, ask for one
   extra composite question (T3/T4 cycle, one cluster per round) until the
   target is reached or MAX_EXTRA_ROUNDS runs out.
3. Normalisation: every row is written with BOTH `gold_answer` and
   `reference_answer` (run_benchmark.py / run_benchmark_rag.py read
   `reference_answer` as the ragas ground truth), plus `arxiv_id` (first),
   `arxiv_ids`, `cluster` (seed) and `evidence`.

Valid rows must carry: question, a known type, gold_answer, a non-empty
evidence list, and arxiv_ids that all belong to the cluster that generated
them. Duplicate questions (casefold-normalised) are dropped.

Usage:
    python scripts/gen_qa.py [--clusters-file questions/selected_clusters.json]
        [--out questions/v2.jsonl] [--target 104]
        [--base-counts '{"T1":2,"T2":2,"T3":2}']
        [--max-extra-rounds 64]

Env: same as moderag (LLM_MODE/LLM_MODEL/...) — reuse the run script env.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                     # LHCpubs_run2/
QUESTIONS = ROOT / "questions"
DEFAULT_CLUSTERS = QUESTIONS / "selected_clusters.json"
DEFAULT_OUT = QUESTIONS / "v2.jsonl"
# Genotypes are numbered T1..T4 (renumbered 2026-09): bridge, cross-document
# aggregation, heterogeneous multi-hop, three-retriever. The old single-paper,
# within-paper-synthesis and no-evidence genotypes are no longer generated.
KNOWN_TYPES = {"T1", "T2", "T3", "T4"}

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent.parent))   # repo root, so `moderag` imports
import generate_qa  # noqa: E402


def norm_question(q: str) -> str:
    return " ".join(q.casefold().split())


def load_clusters(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    clusters = []
    for c in data:
        if isinstance(c, dict) and c.get("papers"):
            clusters.append(c)
    return clusters


def validate_rows(rows: list, papers_ids: set) -> list:
    """Keep rows that are structurally usable by run_benchmark."""
    out = []
    for q in rows:
        if not isinstance(q, dict):
            continue
        if not q.get("question") or not q.get("gold_answer"):
            continue
        typ = q.get("type", "")
        if typ not in KNOWN_TYPES:
            continue
        ev = q.get("evidence")
        if not isinstance(ev, list) or not ev:
            continue
        ids = q.get("arxiv_ids")
        if not isinstance(ids, list) or not ids:
            ids = [e.get("arxiv_id") for e in ev if e.get("arxiv_id")]
        ids = [str(x).strip() for x in ids if str(x).strip()]
        if not ids:
            continue
        if not set(ids) <= papers_ids:
            # Evidence must stay inside the closed cluster (the index subset).
            continue
        q = dict(q)
        q["arxiv_ids"] = ids
        q["arxiv_id"] = q.get("arxiv_id") or ids[0]
        q["reference_answer"] = q["gold_answer"]
        out.append(q)
    return out


def call_cluster(papers, counts: dict, papers_ids: set) -> list:
    """One LLM call for a cluster; returns validated rows."""
    raw = generate_qa.gen_for_cluster(papers, dict(counts))
    if not isinstance(raw, list):
        return []
    return validate_rows(raw, papers_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters-file", type=Path, default=DEFAULT_CLUSTERS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--target", type=int, default=104)
    ap.add_argument("--base-counts", type=str,
                    default='{"T1":2,"T2":2,"T3":2}')
    ap.add_argument("--max-extra-rounds", type=int, default=96)
    ap.add_argument("--extra-types", type=str, default="T3,T4",
                    help="comma-separated genotypes used to fill the deficit")
    ap.add_argument("--max-tries", type=int, default=2,
                    help="retries per cluster call when the LLM output is unusable")
    args = ap.parse_args()

    clusters = load_clusters(args.clusters_file)
    base_counts = json.loads(args.base_counts)
    # drop zero-count / unknown genotypes
    base_counts = {t: n for t, n in base_counts.items()
                   if t in KNOWN_TYPES and n > 0}
    base_per_cluster = sum(base_counts.values())
    print(f"[gen] clusters={len(clusters)} base_counts={base_counts} "
          f"({base_per_cluster}/cluster) target={args.target}")

    all_rows: list = []
    seen = set()
    per_cluster = {}

    def absorb(rows: list, cluster_name: str) -> None:
        kept = 0
        for q in rows:
            key = norm_question(q["question"])
            if key in seen:
                continue
            seen.add(key)
            q["cluster"] = cluster_name
            all_rows.append(q)
            kept += 1
        return kept

    for c in clusters:
        name = c["seed"]
        pids = c["papers"]
        papers = generate_qa.load_papers(pids)
        if len(papers) < 2:
            print(f"[gen] skip {name}: usable papers={len(papers)}")
            continue
        papers_ids = set(p["arxiv_id"] for p in papers)
        got = []
        for attempt in range(1, args.max_tries + 1):
            try:
                got = call_cluster(papers, base_counts, papers_ids)
            except Exception as exc:  # JSON / invoke errors
                print(f"[gen] {name} attempt {attempt} failed: {type(exc).__name__}: {exc}")
                got = []
            if got:
                break
        added = absorb(got, name)
        per_cluster[name] = added
        print(f"[gen] {name:22s} base +{added} "
              f"(raw {len(got)})  running total {len(all_rows)}", flush=True)
        if args.target and len(all_rows) >= args.target:
            break

    # ---- extra rounds to reach the target --------------------------------
    extra_cycle = [t for t in args.extra_types.split(",") if t in KNOWN_TYPES] or ["T3"]
    rounds = 0
    while args.target and len(all_rows) < args.target and rounds < args.max_extra_rounds:
        ci = rounds % len(clusters)
        c = clusters[ci]
        papers = generate_qa.load_papers(c["papers"])
        papers_ids = set(p["arxiv_id"] for p in papers)
        typ = extra_cycle[(rounds // len(clusters)) % len(extra_cycle)]
        if len(papers) < 2:
            rounds += 1
            continue
        got = []
        for _ in range(args.max_tries):
            try:
                got = call_cluster(papers, {typ: 1}, papers_ids)
            except Exception as exc:
                print(f"[gen] {c['seed']} extra {typ} failed: {exc}")
                got = []
            if got:
                break
        # only keep rows of the requested genotype to preserve the target mix
        got = [q for q in got if q["type"] == typ]
        before = len(all_rows)
        absorb(got, c["seed"])
        per_cluster[c["seed"]] = per_cluster.get(c["seed"], 0) + (len(all_rows) - before)
        rounds += 1
        if len(all_rows) != before:
            print(f"[gen] extra #{rounds} {c['seed']:22s} +{typ} "
                  f"(total {len(all_rows)})", flush=True)
        elif rounds % len(clusters) == 0:
            print(f"[gen] extra pass {rounds // len(clusters)} done, "
                  f"total {len(all_rows)}, stopping early (LLM not yielding new valid rows)")
            break

    # ---- cap at target and write -----------------------------------------
    if args.target and len(all_rows) > args.target:
        print(f"[gen] capping {len(all_rows)} -> {args.target}")
        all_rows = all_rows[:args.target]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for q in all_rows:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    types = Counter(q["type"] for q in all_rows)
    multi = sum(1 for q in all_rows if len(q["arxiv_ids"]) >= 2)
    papers = set(pid for q in all_rows for pid in q["arxiv_ids"])
    print("[gen] done")
    print(f"[gen] wrote {len(all_rows)} questions -> {args.out}")
    print(f"[gen] by_type={dict(types)}  multi_paper={multi}  "
          f"distinct_papers={len(papers)}")
    (args.out.with_suffix(".composition.json")).write_text(
        json.dumps({
            "file": str(args.out), "n": len(all_rows),
            "by_type": dict(types), "multi_paper": multi,
            "distinct_papers": sorted(papers),
            "per_cluster": per_cluster,
            "base_counts": base_counts, "target": args.target,
        }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
