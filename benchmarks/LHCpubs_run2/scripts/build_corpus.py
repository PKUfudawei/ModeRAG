#!/usr/bin/env python3
"""
Materialise the LHCpublications benchmark corpus.

build_index.py reads markdown from `corpus/<arxiv_id>/<arxiv_id>.md`, but the
corpus is not tracked in git: `corpus/` is ignored because the documents belong
to the LHCpublications dataset repository, not to this one. This script rebuilds
the directory from the published selection list, `questions/corpus_papers.json`
(96 papers, each with its arxiv_id).

    # from a clone of the dataset repository: link to the markdown in place
    python benchmarks/LHCpubs_run2/scripts/build_corpus.py \
        --source benchmarks/LHCpubs/arxiv

    # or fetch just the 96 markdown files (16 MB) and copy them in
    python benchmarks/LHCpubs_run2/scripts/build_corpus.py --download

`--source` is the cheaper option if you already have the dataset clone; it needs
9.7 GB on disk (the papers carry figures and PDFs) but no second copy of the
markdown. `--download` keeps the checkout self-contained and pulls 16 MB.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]     # LHCpubs_run2/
DEFAULT_PAPERS = ROOT / "questions" / "corpus_papers.json"
DATASET_REPO = "PKUfudawei/LHCpublications"


def load_paper_ids(papers_path: Path) -> list[str]:
    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    return [p["arxiv_id"] for p in papers]


def download_markdown(ids: list[str]) -> Path:
    """Fetch <id>/<id>.md for the selected papers into a temporary directory."""
    staging = Path(tempfile.mkdtemp(prefix="lhcpubs_md_"))
    cmd = ["hf", "download", DATASET_REPO, "--repo-type", "dataset",
           "--local-dir", str(staging)]
    for arxiv_id in ids:
        cmd += ["--include", f"arxiv/{arxiv_id}/{arxiv_id}.md"]
    print(f"[corpus] fetching {len(ids)} markdown files from {DATASET_REPO}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise SystemExit("hf download failed; pass --source to use a local clone instead")
    return staging / "arxiv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", type=Path, default=DEFAULT_PAPERS,
                    help="selection list (default: questions/corpus_papers.json)")
    ap.add_argument("--source", type=Path,
                    help="directory holding <arxiv_id>/<arxiv_id>.md, typically "
                         "<dataset clone>/arxiv")
    ap.add_argument("--download", action="store_true",
                    help="fetch the 96 markdown files (16 MB) instead of reading a clone")
    ap.add_argument("--out", type=Path, default=ROOT / "corpus",
                    help="corpus directory to create (default: <LHCpubs_run2>/corpus)")
    args = ap.parse_args()

    if not args.papers.is_file():
        raise SystemExit(f"selection list not found: {args.papers}")
    if args.download == (args.source is not None):
        raise SystemExit("pass exactly one of --source <clone>/arxiv or --download")
    ids = load_paper_ids(args.papers)

    staging = None
    if args.download:
        staging = download_markdown(ids)
        source, copy = staging, True
    else:
        source, copy = args.source, False

    created = skipped = missing = 0
    for arxiv_id in ids:
        target = source / arxiv_id / f"{arxiv_id}.md"
        link_dir = args.out / arxiv_id
        dest = link_dir / f"{arxiv_id}.md"
        if not target.is_file():
            print(f"[corpus] missing source: {target}")
            missing += 1
            continue
        if dest.exists() or dest.is_symlink():
            skipped += 1
            continue
        link_dir.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(target, dest)
        else:
            dest.symlink_to(target.resolve())
        created += 1

    if staging is not None:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"[corpus] {created} created, {skipped} already present, "
          f"{missing} missing at the source -> {args.out}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
