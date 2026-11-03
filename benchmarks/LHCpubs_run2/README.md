# LHCpublications (run 2)

100 questions over 96 CMS/ATLAS publications, 25 per genotype, each requiring
evidence from at least two papers. Published as the `LHCpublications` dataset;
`benchmark.jsonl` here is the released question set.

| Genotype | What it requires | Was |
|---|---|---|
| T1 | complementary bridge: one paper supplies a value or entity the other names but does not resolve | T3 / T3A |
| T2 | cross-document aggregation: the same measured quantity reported in two or more papers | T4 |
| T3 | heterogeneous multi-hop: hops that genuinely need different retrieval strategies | T5 |
| T4 | three-retriever: a concrete value plus two further hops of a different kind | T7 |

The old ids were gaps left by a seven-type generation taxonomy. The three
genotypes that were never released (single-paper factual, within-paper synthesis,
no-evidence) have been removed from the question sets and the generation pools,
so every file in this directory uses T1-T4.

## Layout

| Path | Role |
|---|---|
| `benchmark.jsonl` | released question set (100 questions) |
| `benchmark.jsonl.README.md` | composition, and the disclosure of how the T2 subset was chosen |
| `results/results.jsonl` | the released table: one header line with a mean row per configuration, then one line per question carrying its answers and scores in every configuration |
| `questions/` | corpus selection and cluster files, i.e. tool inputs rather than question sets |
| `scripts/` | question generators and the retrievability gate |
| `run.sh` | scores `benchmark.jsonl` into `results/` and prints the table |
| `corpus/`, `indexes/` | markdown corpus and graph/vector index (downloaded, not committed) |

`results/results.jsonl` is the whole score artifact: it holds, for each of the
100 released questions, the answer and Ragas scores of every configuration.
There is no second, raw layer - the per-batch run directories were copies of
rows already in this file, so they are gone. Scoring a configuration again
upserts its rows here rather than adding a file.

## `questions/`

`benchmark.jsonl` at the root of this directory is the only question set here:
100 questions, 25 per genotype. `questions/` holds no questions at all - just the
two files the tooling reads:

| File | Role |
|---|---|
| `corpus_papers.json` | the 96 papers the index is built over (`scripts/build_corpus.py`) |
| `selected_clusters.json` | default cluster file for `scripts/gen_qa.py` |

Generation is `scripts/gen_qa.py` (batched; writes a `<out>.composition.json` run
log beside its output) and the gate is `scripts/qa_gate.py`.

Every question in `benchmark.jsonl` carries its generation batch in the
`qa_source` field (e.g. `newgen:v2raw-20260909`) as provenance.
`benchmark.jsonl` is byte-identical to the copy published with the dataset.

**Questions that are not in `benchmark.jsonl` are not released.** In particular
the 8 T2 questions that the outcome-based trim removed, and the question sets
behind the untrimmed and file-order comparisons, are not part of this
repository, so those two comparison figures cannot be recomputed from it. They
are reported in the paper, which states how the released subset was chosen.

## Sensitivities the paper reports

Two construction choices could inflate the result, and both are disclosed in the
paper and in `benchmark.jsonl.README.md`. The paper reports the figures below,
but they cannot be recomputed from this repository: the questions they compare
are not released, and the per-batch score files are gone.

1. **T2 was trimmed 33 to 25 by outcome** - the 8 questions with the worst
   `all - naive` were dropped, which raises the reported T2 advantage from
   `+0.065` to `+0.130`. On the untrimmed 33 T2 questions the overall gap is
   `+0.061`; trimming the same 8 by file order instead of by outcome gives
   `+0.065` overall and `+0.083` on T2.
2. **The 11 T1 top-up questions were selected with `all >= naive`.** Two
   independently generated T1 batches bound that, at `+0.21` and `+0.05`. Those
   batches' questions are in the release, but one of them was re-scored inside
   the released run, so the released rows do not reproduce the per-batch
   figures either.

## Reproducing

```bash
./run.sh                            # score the released questions into results/
python ../analysis/bootstrap_ci.py  # all deltas, intervals and noise bands
```

`run.sh` exports `BENCH_ROOT` and drives the shared harness in `../common/`.
Scoring a configuration from scratch, e.g. the full tool set over the released
questions, is:

```bash
export BENCH_ROOT=$PWD LLM_MODEL=... LLM_BASE_URL=... LLM_API_KEY=...
python ../common/run_benchmark.py --tool-config all \
    --questions-file benchmark.jsonl --workers 16 --with-ragas
python ../common/run_benchmark_rag.py --top-k 5 --recall-top-k 20 --with-ragas
```
