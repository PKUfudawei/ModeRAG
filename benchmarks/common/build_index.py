#!/usr/bin/env python3
"""
Build GraphRAG index for EIC2023 corpus.

Features:
- Load markdown documents from corpus directory
- Chunk documents with truncation at acknowledgement sections
- Save vector store first, then build knowledge graph
- Record chunks that failed during entity extraction

The index behind the reported numbers is published at
https://huggingface.co/datasets/PKUfudawei/LHCpublications under indexes/, so
this script is only needed to rebuild from scratch. Entity and relationship
extraction is an LLM pass and will not reproduce the published index bit for
bit.
"""

import json
import hashlib
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from tqdm import tqdm

from moderag.models.chunker import get_chunker
from moderag.graphrag.indexer import get_graphrag_indexer


# Configuration (shared with the runners; see paths.py)
from paths import CORPUS_DIR, INDEXES_DIR  # noqa: E402
TRUNCATIONS = ['acknowledgement', 'acknowledgment', 'Acknowledgement', 'Acknowledgment']
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_WORKERS = 16


def load_markdown_documents(corpus_dir: Path) -> List[Document]:
    """Load all markdown files from corpus directory."""
    documents = []
    md_files = list(corpus_dir.glob("*/*.md"))

    print(f"Found {len(md_files)} markdown files")

    for md_path in tqdm(md_files, desc="Loading documents"):
        try:
            content = md_path.read_text(encoding="utf-8")

            # Calculate file hash for tracking
            file_hash = hashlib.md5(content.encode()).hexdigest()[:16]

            document = Document(
                page_content=content,
                metadata={
                    "source": str(md_path),
                    "file_name": md_path.name,
                    "file_hash": file_hash,
                    "relative_path": str(md_path.relative_to(corpus_dir))
                }
            )
            documents.append(document)
        except Exception as e:
            print(f"Error loading {md_path}: {e}")

    print(f"Loaded {len(documents)} documents")
    return documents


def _chunk_meta(doc) -> dict:
    """Common metadata block for failure / low-signal records."""
    return {
        "chunk_id": doc.metadata.get("chunk_id", "unknown"),
        "file_hash": doc.metadata.get("file_hash", "unknown"),
        "source": doc.metadata.get("source", "unknown"),
        "content_preview": doc.page_content[:200],
    }


def extract_batch_with_tracking(
    graph_builder,
    chunks: List[Document],
    mode: str = "thread",
    timeout: int = 300
) -> tuple:
    """
    Extract entities from chunks and track failures.

    Real extraction errors (which used to be swallowed by the extractor and
    reported as fake ``empty_result``) are re-raised here and recorded with the
    actual error string. Chunks that extract *successfully* but produce zero
    entities / relationships are not failures: they are flagged as low-signal
    (pure-vector) chunks and are not fed into the graph.

    Returns:
        tuple: (graph_docs, failed_chunks, low_signal_chunks)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
    import asyncio

    extractor = graph_builder.extractor
    # Surface the real per-chunk error instead of swallowing it, and retry each
    # failing chunk once (transient structured-output / parse failures).
    extractor.raise_on_error = True
    extractor.max_retries = 1

    failed_chunks = []
    low_signal_chunks = []
    graph_docs = []

    def handle_result(doc, result):
        """Returns the doc to add to the graph, or None for empty results."""
        if len(result.nodes) == 0 and len(result.relationships) == 0:
            meta = _chunk_meta(doc)
            meta["reason"] = "no_entities_low_signal"
            low_signal_chunks.append(meta)
            return None
        return result

    def handle_failure(doc, error: str):
        meta = _chunk_meta(doc)
        meta["error"] = error
        failed_chunks.append(meta)

    if mode == "async":
        # Async extraction with tracking and timeout
        pbar = tqdm(total=len(chunks), desc="Extracting (async)", unit="chunk")

        async def extract_with_tracking(doc: Document):
            try:
                result = await asyncio.wait_for(extractor.aextract(doc), timeout=timeout)
                good = handle_result(doc, result)
                if good is not None:
                    graph_docs.append(good)
            except asyncio.TimeoutError:
                handle_failure(doc, f"Timeout after {timeout}s")
            except Exception as e:
                handle_failure(doc, str(e))
            finally:
                pbar.update(1)

        async def batch_extract():
            tasks = [extract_with_tracking(doc) for doc in chunks]
            return await asyncio.gather(*tasks)

        asyncio.run(batch_extract())
        pbar.close()

    elif mode == "thread":
        # Thread pool extraction with tracking and timeout
        with ThreadPoolExecutor(max_workers=extractor.max_workers) as executor:
            futures = {executor.submit(extractor.extract, doc): doc for doc in chunks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
                doc = futures[future]
                try:
                    result = future.result(timeout=timeout)
                    good = handle_result(doc, result)
                    if good is not None:
                        graph_docs.append(good)
                except FuturesTimeoutError:
                    handle_failure(doc, f"Timeout after {timeout}s")
                except Exception as e:
                    handle_failure(doc, str(e))
    else:
        # Sequential extraction with timeout
        for doc in tqdm(chunks, desc="Extracting"):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(extractor.extract, doc)
                    result = future.result(timeout=timeout)
                good = handle_result(doc, result)
                if good is not None:
                    graph_docs.append(good)
            except FuturesTimeoutError:
                handle_failure(doc, f"Timeout after {timeout}s")
            except Exception as e:
                handle_failure(doc, str(e))

    return graph_docs, failed_chunks, low_signal_chunks


def build_index():
    """Main function to build the GraphRAG index."""
    print("=" * 60)
    print("Building GraphRAG Index for EIC2023 Corpus")
    print("=" * 60)

    # Create output directory
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load documents
    print("\n[Step 1] Loading markdown documents...")
    documents = load_markdown_documents(CORPUS_DIR)

    if not documents:
        print("No documents found. Exiting.")
        return

    # Step 2: Create indexer with custom chunker
    print("\n[Step 2] Creating GraphRAG indexer...")
    chunker = get_chunker(
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        truncations=TRUNCATIONS
    )
    indexer = get_graphrag_indexer(
        chunker=chunker,
        max_workers=MAX_WORKERS,
        extract_mode="thread"
    )

    # Step 3: Chunk documents
    print("\n[Step 3] Chunking documents...")
    chunks = indexer.chunker.split_documents(documents)
    print(f"Generated {len(chunks)} chunks")

    # Step 4: Build and save vectorstore FIRST
    print("\n[Step 4] Building and saving vectorstore...")
    vectorstore_path = INDEXES_DIR / "vectorstore"
    vectorstore = indexer.build_vectorstore(chunks)
    indexer.save_vectorstore(vectorstore, str(vectorstore_path))
    print(f"Vectorstore saved to {vectorstore_path}")

    # Step 5: Extract entities with failure tracking
    print("\n[Step 5] Extracting entities and relationships...")
    graph_docs, failed_chunks, low_signal_chunks = extract_batch_with_tracking(
        indexer.graph_builder,
        chunks,
        mode="thread",
        timeout=300  # 5 minute timeout per chunk
    )

    # Step 6: Build graph from extracted data
    print("\n[Step 6] Building knowledge graph...")
    stats = indexer.graph_builder.align_and_build(graph_docs, incremental=False)
    graph = indexer.graph_builder.graph
    print(f"Graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships")

    # Step 7: Save graph
    print("\n[Step 7] Saving graph...")
    graph_path = INDEXES_DIR / "graph.pkl"
    indexer.save_graph(str(graph_path))

    # Step 8: Save entity and relationship indexes
    print("\n[Step 8] Saving entity and relationship indexes...")
    entities_path = INDEXES_DIR / "entities.pkl"
    indexer.save_entities(str(entities_path))

    relationships_path = INDEXES_DIR / "relationships.pkl"
    indexer.save_relationships(str(relationships_path))

    # Step 9: Save failed-chunks and low-signal reports
    print("\n[Step 9] Saving failed chunks & low-signal reports...")
    failed_chunks_path = INDEXES_DIR / "failed_chunks.json"
    low_signal_path = INDEXES_DIR / "low_signal_chunks.json"

    # Group failed chunks by file (reason carries the real extraction error)
    failed_by_file = {}
    for fc in failed_chunks:
        file_hash = fc.get("file_hash", "unknown")
        if file_hash not in failed_by_file:
            failed_by_file[file_hash] = {
                "file_hash": file_hash,
                "source": fc.get("source", "unknown"),
                "failed_chunks": []
            }
        failed_by_file[file_hash]["failed_chunks"].append({
            "chunk_id": fc.get("chunk_id"),
            "reason": fc.get("reason", fc.get("error", "unknown")),
            "content_preview": fc.get("content_preview", "")
        })

    report = {
        "total_chunks": len(chunks),
        "total_failed": len(failed_chunks),
        "failure_rate": len(failed_chunks) / len(chunks) * 100 if chunks else 0,
        "failed_by_file": list(failed_by_file.values()),
        "failed_chunks_detail": failed_chunks
    }

    with open(failed_chunks_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Low-signal chunks: extraction succeeded but produced no entities; they are
    # still indexed in the vectorstore (pure-vector chunks) and only absent from
    # the graph.
    low_signal_report = {
        "total_chunks": len(chunks),
        "total_low_signal": len(low_signal_chunks),
        "low_signal_rate": len(low_signal_chunks) / len(chunks) * 100 if chunks else 0,
        "low_signal_chunks": low_signal_chunks
    }
    with open(low_signal_path, "w", encoding="utf-8") as f:
        json.dump(low_signal_report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("Indexing Complete!")
    print("=" * 60)
    print(f"Documents processed: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Failed extractions: {len(failed_chunks)} ({report['failure_rate']:.2f}%)")
    print(f"Low-signal (pure-vector) chunks: {len(low_signal_chunks)} "
          f"({low_signal_report['low_signal_rate']:.2f}%)")
    print(f"Graph entities: {stats.get('entities', 0)}")
    print(f"Graph relationships: {stats.get('relationships', 0)}")
    print(f"\nIndexes directory: {INDEXES_DIR}")
    print(f"Failed chunks report: {failed_chunks_path}")
    print(f"Low-signal chunks report: {low_signal_path}")

    if failed_chunks:
        print(f"\nFiles with failed extractions:")
        for file_info in failed_by_file.values():
            print(f"  - {file_info['source']}: {len(file_info['failed_chunks'])} failed chunks")


if __name__ == "__main__":
    build_index()
