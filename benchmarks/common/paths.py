"""Resolve the benchmark root that the shared harness operates on.

These runners live in benchmarks/common/ and are shared by every benchmark, so
they cannot derive their data locations from ``__file__``. The benchmark under
test is instead:

    1. ``$BENCH_ROOT`` when set - this is what each benchmark's ``run.sh``
       exports, and what you should set if you invoke the harness directly, else
    2. the current working directory.

Everything else hangs off that root: the released question set is
``<root>/benchmark.jsonl``, the prebuilt index is ``<root>/indexes``, the corpus
markdown is ``<root>/corpus``, and runs write to ``<root>/results``.
``BENCH_OUTPUT_DIR`` and ``--questions-file`` override the last two per run,
which is how the archived control runs were produced.
"""
from __future__ import annotations

import os
from pathlib import Path

BENCH_ROOT = Path(os.environ.get("BENCH_ROOT") or Path.cwd()).resolve()
BENCHMARK_FILE = BENCH_ROOT / "benchmark.jsonl"
INDEXES_DIR = BENCH_ROOT / "indexes"
CORPUS_DIR = BENCH_ROOT / "corpus"
# BENCH_OUTPUT_DIR lets a run write elsewhere so archived results stay untouched.
OUTPUT_DIR = Path(os.environ.get("BENCH_OUTPUT_DIR", str(BENCH_ROOT / "results")))
