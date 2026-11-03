# Benchmarks

Two question sets, one harness. Both are evaluated with the same code path, the
same indexer, the same DeepSeek-V4-Flash backend and the same Ragas metrics; they
differ only in corpus and question composition.

| | `AI4EIC2023` | `LHCpubs_run2` |
|---|---|---|
| Corpus | 38 workshop papers | 96 CMS/ATLAS publications |
| Questions | 50 | 100 (25 per genotype) |
| Question file | `benchmark.jsonl` | `benchmark.jsonl` |
| Released results | `results/` | `results/` |
| Genotypes | mostly single-paper | T1 bridge / T2 aggregation / T3 heterogeneous / T4 three-retriever |
| Entry point | `run.sh` | `run.sh` |

## Layout convention

Every benchmark directory follows the same shape, so reproduction steps transfer
between them:

```
<benchmark>/
├── README.md            corpus + questions + how to reproduce
├── benchmark.jsonl      the released question set (also published as a dataset)
├── run.sh               driver: scores every configuration into results/
├── results/             the released table, as one results.jsonl
├── corpus/              markdown corpus      (download, not committed)
└── indexes/             graph + vector index (download, not committed)
```

`LHCpubs_run2/` carries two extra directories: the question set was written over
several generation batches, and the tools that produced it are kept for
provenance.

| Directory | Contents |
|---|---|
| `questions/` | corpus selection and cluster files - tool inputs, not question sets |
| `scripts/` | the question generators and the retrievability gate |

## Shared harness

`common/` holds the only copy of the evaluation code. It is shared rather than
duplicated so the two benchmarks cannot drift apart — they previously carried
near-identical copies that had already diverged (one had `--questions-file` and
`BENCH_OUTPUT_DIR`, the other did not).

| File | Role |
|---|---|
| `common/paths.py` | resolves `$BENCH_ROOT`; every other path hangs off it |
| `common/build_index.py` | builds the graph + vector index from `corpus/` |
| `common/run_benchmark.py` | the agent configurations (naive/local/global/local+global/all) |
| `common/run_benchmark_rag.py` | the single-call Vector RAG baseline |
| `common/ragas_eval.py` | shared question loading, results IO and Ragas scoring |
| `common/results_file.py` | the results file format: one file per table, read/upsert/merge |

The harness operates on whichever benchmark `$BENCH_ROOT` points at (each
`run.sh` exports it); run the scripts directly only if you set it yourself:

```bash
export BENCH_ROOT=benchmarks/AI4EIC2023
python benchmarks/common/run_benchmark.py --tool-config all --with-ragas
```

`BENCH_OUTPUT_DIR` and `--questions-file` override the output directory and the
question set per run; that is how the archived control runs were produced.

## One table, one file

A result set is a single `results.jsonl`: a header line carrying one mean row per
configuration (the paper's table), then one line per question carrying that
question's answer and Ragas scores in every configuration. The runners *upsert*
their own configuration into that file, so running one configuration updates the
shared table in place and re-running anything converges on the same single
artifact rather than adding a `results_<config>.jsonl` beside it. The format is
specified in `common/results_file.py`.

## Reproducing the paper

```bash
# score every configuration of both benchmarks (needs the LLM backend and indexes)
benchmarks/AI4EIC2023/run.sh
benchmarks/LHCpubs_run2/run.sh

# paired per-question bootstrap: the deltas, intervals and noise bands in the paper
python benchmarks/analysis/bootstrap_ci.py
```

`analysis/bootstrap_ci.py` reads `results/` from both benchmarks and reproduces
Table `tab:noise` and the genotype rows of Table `tab:querytype`.
