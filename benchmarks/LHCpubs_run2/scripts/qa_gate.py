#!/usr/bin/env python3
"""QA retrievability gate + re-scored comparison.

For every probe question: embed it with the local bge-m3 model, retrieve the
vectorstore top-20 chunks, and check whether the question's gold evidence can
actually be retrieved (normalised snippet containment, or the cited paper
showing up in the retrieved chunks).  Questions that cannot be retrieved are
"unanswerable by retrieval" — any correct answer on them is prior-knowledge
guessing, not a retrieval win.  The existing naive/all ragas scores (already
produced with deepseek-v4-flash-ioa) are then regrouped by PASS/FAIL.

Usage: EMBEDDING_DEVICE=cpu python scripts/qa_gate.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root

ROOT = Path(__file__).resolve().parents[1]     # LHCpubs_run2/
BENCH = ROOT / "questions"
TOPK = int(os.getenv("GATE_TOPK", "20"))


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", nargs="+", required=True,
                    help="question files to gate, relative to questions/ unless absolute")
    ap.add_argument("--out", default=None,
                    help="write the passing questions to this jsonl (benchmark file)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the written file at N passing questions (0 = all)")
    return ap.parse_args()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", "", s)


def load_vectorstore(storage_path: str):
    from langchain_community.vectorstores import FAISS
    from moderag.models.embedding import get_embedding

    embedding = get_embedding(model="BAAI/bge-m3",
                              device=os.getenv("EMBEDDING_DEVICE", "cpu"),
                              wrapped=False)
    return FAISS.load_local(str(Path(storage_path) / "vectorstore"), embedding,
                            allow_dangerous_deserialization=True)


def main() -> None:
    args = parse_args()
    questions = []
    for f in args.qa:
        p = BENCH / f if not str(f).startswith("/") else Path(f)
        if not p.exists():
            print(f"[gate] missing {p}")
            continue
        for line in open(p):
            o = json.loads(line)
            questions.append(o)

    vs = load_vectorstore(str(BENCH / "indexes"))
    print(f"[gate] retrieved {len(questions)} questions, topk={TOPK}")

    gate = {}
    for i, q in enumerate(questions, 1):
        docs = vs.similarity_search(q["question"], k=TOPK)
        texts = [norm(d.page_content) for d in docs]
        metas = [str(d.metadata or {}) for d in docs]
        hits = 0
        total = 0
        for e in q.get("evidence", []):
            snip = norm(e.get("snippet", ""))
            if len(snip) < 40:
                continue
            total += 1
            # distinctive fragment: middle 80 chars of the snippet
            frag = snip[max(0, len(snip) // 4):][:80]
            pid = str(e.get("arxiv_id", ""))
            if any(frag in t for t in texts) or (pid and any(pid in m for m in metas)):
                hits += 1
        cov = hits / total if total else 0.0
        gate[q["question"]] = {"hits": hits, "total": total, "coverage": round(cov, 3),
                               "pass": cov >= 0.5}
        print(f"[gate] {i:2d}/{len(questions)} type={q.get('type')} "
              f"cov={cov:.2f} {'PASS' if cov >= 0.5 else 'FAIL'}  {q['question'][:60]}",
              flush=True)

    if args.out:
        from collections import Counter
        keep = [q for q in questions if gate[q["question"]]["pass"]]
        if args.limit:
            keep = keep[:args.limit]
        outp = BENCH / args.out if not str(args.out).startswith("/") else Path(args.out)
        with open(outp, "w") as f:
            for q in keep:
                q = dict(q)
                q["reference_answer"] = q.get("reference_answer") or q.get("gold_answer", "")
                q["arxiv_id"] = q.get("arxiv_id") or (q.get("arxiv_ids") or ["?"])[0]
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"[gate] wrote {len(keep)} PASS questions -> {outp} "
              f"({dict(Counter(q.get('type') for q in keep))})", flush=True)

    passed = sum(1 for g in gate.values() if g["pass"])
    print(f"\n=== gate summary ===\nPASS {passed} / FAIL {len(gate) - passed} / total {len(gate)}")

    # The per-question verdicts go to stdout: the gate is a filter, and writing a
    # second file beside the question set only leaves an unread artifact behind.
    if args.out:
        print(f"\n[gate] wrote {outp} ({len(keep)} passing of {len(gate)})")
    print("[gate] per-question coverage:")
    for q, g in gate.items():
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['hits']}/{g['total']}  {q[:80]}")


if __name__ == "__main__":
    main()
