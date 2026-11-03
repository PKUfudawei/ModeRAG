#!/usr/bin/env python3
"""Step 3: evidence-first QA generation for a chosen set of LHCpubs clusters.

Input  : a clusters.json whose entries carry selected paper arxiv_ids that
         genuinely share a measurable/object (a short curated list; the
         heuristic draft clusters need human trimming first).
Method  : for each cluster, load each paper's full markdown, ask the LLM to
          (1) identify the shared bridge observable(s), (2) locate the per-paper
          evidence snippets, and (3) emit questions of a requested genotype
          that REQUIRE >=2 papers to answer, with a gold answer and the
          per-hop evidence (paper_id + section + snippet).
Output  : questions/qa_benchmark.jsonl  (one record per question)
Schema  : question / type(T1..T4) / arxiv_ids / gold_answer /
          evidence:[{arxiv_id, section, snippet}]
LLM     : reuse the ModeRAG chat factory (Tencent gateway default), same
          model family as the RAG evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
try:
    from moderag.models.llm import get_llm  # type: ignore
except Exception:  # pragma: no cover - env may lack langchain
    get_chat = None

ROOT = Path(__file__).resolve().parents[1]     # LHCpubs_run2/
OUT = ROOT / "questions"

# Genotypes are numbered T1..T4 (renumbered 2026-09 from the old seven-type
# taxonomy, which left the gaps T3/T4/T5/T7). T1 is the complementary bridge; it
# used to be generated under the label T3A to separate it from the old broad
# "cross-paper entity" T3, whose same-term-different-meaning variant was dropped
# from the benchmark. The old single-paper (T1), within-paper-synthesis (T2) and
# no-evidence (T6) genotypes are no longer generated.
_T1_BRIDGE = (
    "complementary bridge (NOT same-term-different-meaning): the two/three papers each hold a "
    "DISJOINT part of the answer (e.g. paper A defines/identifies an object referenced only as "
    "'[14]' in paper B, or paper A gives the translation formula while paper B gives the physical "
    "interpretation). Answering REQUIRES combining the parts; no single paper is sufficient. Do NOT "
    "ask 'what does term X mean in each paper' or 'is X the same object in both' — that is the "
    "distractor pattern which vector search already answers well."
)

TYPE_INSTR = {
    "T1": _T1_BRIDGE,
    "T2": "cross-document aggregation/comparison of the same measured quantity reported in >=2 papers",
    "T3": "heterogeneous multi-hop: different hops need vector vs graph vs aggregate evidence",
    "T4": "three-retriever mandatory: the question MUST need (a) a concrete numeric/detail value "
          "retrievable only by vector search, AND (b) an entity/alias defined in one paper that must be "
          "resolved through another paper (graph/entity relation), AND (c) a cross-document aggregation or "
          "comparison of the same quantity across >=2 papers. Answering requires all three; omitting any one "
          "hop makes the question unanswerable.",
}

SYSTEM = (
    "You design closed-book retrieval questions over real LHC papers. "
    "You are given papers with ids and their full text. Rules:\n"
    "- Compose the question so that answering it REQUIRES evidence from >=2 different papers. "
    "  Every genotype is multi-paper; no single listed paper may be sufficient.\n"
    "- Never leak the paper identity (no 'ATLAS in arXiv 1902.xxxxx finds ...'); phrase "
    "  like a user asking about physics results.\n"
    "- Include specific quantities/numbers so it cannot be guessed.\n"
    "- For every question provide gold_answer and evidence list: for each needed hop, "
    "  the arxiv_id, approximate section, and a verbatim snippet.\n"
    "- Output STRICT JSON array only.\n"
    "- For T4 the three hops MUST be genuinely different in kind: one verbatim numeric/detail\n"
    "  snippet (vector), one entity/alias resolution across papers (graph relation), one\n"
    "  cross-paper aggregation/comparison of the same quantity. Give a separate evidence entry\n"
    "  per hop and make the gold answer combine all three."
)


def load_papers(ids):
    idx = {}
    for p in json.loads((OUT / "run2_run3_papers.json").read_text()):
        idx[p["arxiv_id"]] = p
    out = []
    for pid in ids:
        p = idx.get(pid)
        if not p:
            continue
        md = Path(p["path"])
        mds = list(md.glob("*.md"))
        text = mds[0].read_text(encoding="utf-8", errors="ignore") if mds else ""
        out.append({"arxiv_id": pid, "title": p["title"], "text": text[:24000]})
    return out


def gen_for_cluster(papers, counts=None) -> list[dict]:
    counts = counts or {"T1": 2, "T2": 2, "T3": 2}
    docs = "\n\n".join(
        f"@@PAPER {p['arxiv_id']}@@\nTITLE: {p['title']}\n{p['text']}"
        for p in papers
    )
    instr = "\n".join(f"- {t}: {d} (make {c})" for t, c in counts.items()
                      if t in TYPE_INSTR and c > 0 for d in [TYPE_INSTR[t]])
    user = (
        f"Papers:\n{docs}\n\nGenerate the requested number of questions per type:\n{instr}\n"
        "Return only JSON: [{\"question\",\"type\",\"arxiv_ids\":[...],\"gold_answer\","
        "\"evidence\":[{\"arxiv_id\",\"section\",\"snippet\"}]}]"
    )
    llm = get_llm(temperature=0) if get_llm else None
    if llm is None:
        raise SystemExit("no LLM chat factory available (need moderag.models.llm.get_llm)")
    resp = llm.invoke([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user}])
    text = resp.content if hasattr(resp, "content") else str(resp)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    return json.loads(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True,
                    help="comma-separated paper arxiv_ids of one cluster")
    ap.add_argument("--counts", default='{"T1":2,"T2":2,"T3":2}',
                    help='JSON dict {type: n} of questions to generate per type')
    ap.add_argument("--out", default=str(OUT / "qa_benchmark.jsonl"),
                    help="output jsonl (appended)")
    args = ap.parse_args()

    ids = [x.strip() for x in args.cluster.split(",") if x.strip()]
    papers = load_papers(ids)
    if len(papers) < 2:
        raise SystemExit("cluster needs >=2 papers present in the run2/3 pool")
    counts = json.loads(args.counts)
    print(f"[qa] generating from {len(papers)} papers: {[p['arxiv_id'] for p in papers]}")
    qas = gen_for_cluster(papers, counts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for q in qas:
            q.setdefault("cluster", ",".join(ids))
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[qa] wrote {len(qas)} questions -> {out_path}")
    for q in qas[:3]:
        print("  -", q["type"], "|", q["question"][:100])


if __name__ == "__main__":
    main()
