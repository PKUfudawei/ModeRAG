---
license: mit
configs:
- config_name: default
  data_files: benchmark.jsonl
---

# AI4EIC2023

50 question-answer pairs over 38 papers accepted to the AI4EIC2023 workshop on
artificial intelligence for the Electron-Ion Collider, covering jet physics,
detector design, track reconstruction, particle identification and ML
applications. Most questions are single-paper and expository, which is why the
Planner decomposes only 6 of the 50 into more than two sub-tasks. That contrast
with LHCpublications is the point of having a second benchmark.

Credit to [AI4EIC2023_DATASETS](https://smith.langchain.com/public/a0fb3ae5-c878-4626-b8e4-1c45cc5fa566/d).

## Contents

| Path | What it is |
|---|---|
| `benchmark.jsonl` | the 50 questions: `question`, `reference_answer`, `arxiv_id` |
| `results/results.jsonl` | the released table in one file (see below) |
| `run.sh` | scores every configuration into `results/` |
| `corpus/`, `indexes/` | markdown corpus and graph/vector index (downloaded, not committed) |

## Reported configurations

`results/results.jsonl` holds the whole table in one file. Line 1 is the header:
the metrics, the configuration ids, and one mean row per configuration (the
paper's main table). Every later line is one question with an entry per
configuration holding that configuration's answer, retrieved context and Ragas
scores. Configurations only appear on questions they scored, which is why the
graph-only rows cover 88 questions while the others cover 100.

| Config | Meaning |
|---|---|
| `naive` | full multi-agent pipeline restricted to dense passage retrieval |
| `local` | restricted to the entity-centric graph mode |
| `global` | restricted to the relationship-centric graph mode |
| `local+global` | both graph modes, with no dense tool registered |
| `all` | the complete tool set; graph tasks also get a vector-fusion pass |
| `rag_no_reranker_recall20_topk5` | single-call Vector RAG baseline (legacy Reporter prompt) |

The one question whose Ragas faithfulness score came back NaN is recorded under
`nan_scores` in the header's `all` entry, and is excluded from that metric's mean.

## Reproducing

```bash
./run.sh --rebuild-index    # rebuild indexes/ from corpus/ first
./run.sh                    # 5 agent configs + the RAG baseline
python ../analysis/bootstrap_ci.py   # paired confidence intervals
```

`run.sh` reads `MAX_QUESTIONS`, `WORKERS`, `LLM_MODEL`, `LLM_BASE_URL` and
`LLM_API_KEY`. The LLM defaults point at an internal Tencent tt-switch proxy;
point them at any OpenAI-compatible endpoint to run elsewhere. Embeddings are
fixed to a local BAAI/bge-m3 model.

`--no-rag` skips the RAG baseline; positional config names (`./run.sh all naive`)
run a subset.
