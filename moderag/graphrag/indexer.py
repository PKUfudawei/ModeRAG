"""
GraphRAG Indexer - integrates vector index and knowledge graph index
"""
from typing import List, Optional
import faiss
import os
from tqdm import tqdm

from langchain_core.documents import Document

from moderag.graphrag.graph.builder import get_graph_builder
from moderag.rag.indexer import Indexer


class GraphRAGIndexer(Indexer):
    """GraphRAG indexer - indexes both vector database and knowledge graph

    Workflow:
    1. index_documents(): chunk documents (inherited from Indexer)
    2. build_graph_from_chunks(): extract entities from chunks -> align -> build graph
    3. index_entities(): generate embeddings for entities and build vector index
    4. build_vectorstore(): build vector index for chunks (inherited from Indexer)
    """

    def __init__(
        self,
        chunker=None,
        embedding=None,
        graph_builder=None,
        max_workers: int = 16,
        enable_thinking: bool = False,
        extract_mode: str = "thread",
    ):
        super().__init__(
            chunker=chunker, embedding=embedding,
        )
        self.enable_thinking = enable_thinking
        self.max_workers = max_workers
        self.extract_mode = extract_mode
        # Create graph_builder with enable_thinking parameter
        if graph_builder is None:
            from moderag.models.llm import get_llm
            from moderag.graphrag.graph.extractor import get_extractor
            llm = get_llm(enable_thinking=enable_thinking)
            extractor = get_extractor(llm=llm, max_workers=max_workers)
            self.graph_builder = get_graph_builder(extractor=extractor, max_workers=max_workers, extract_mode=extract_mode)
        else:
            self.graph_builder = graph_builder

    def load_vectorstore(self, path):
        return super().load_vectorstore(path)

    def save_graph(self, path):
        self.graph_builder.storage_path = path
        self.graph_builder.save_graph()
        print(f"Graph saved to {path}")

    def clear_graph(self):
        """Clear graph data"""
        self.graph_builder.clear_graph()
        print("Graph cleared")

    def build_graph_from_chunks(self, chunks: List[Document]) -> dict:
        """Build knowledge graph from chunks.

        Args:
            chunks: List of chunked documents

        Returns:
            Statistics dictionary
        """
        return self.graph_builder.build_from_documents(chunks)

    def index_documents(
        self,
        documents,
        database_path,
        incremental: bool = False
    ) -> tuple:
        """Index documents: chunk + build vector index + build graph + build entity index + build relationship index.

        Args:
            documents: Input Document list.
            database_path: Database storage path.
            incremental: Whether to incrementally update. True only processes new documents.

        Returns:
            (chunks, vectorstore, graph, entity_index, relationship_index) tuple.
        """
        import pickle

        # Step 1: Chunk + build vector index (delegate to rag.indexer.Indexer)
        vectorstore_path = os.path.join(database_path, 'vectorstore')
        all_chunks, vectorstore = super().index_documents(
            documents,
            output_path=vectorstore_path
        )

        # Step 2: Build/update graph
        stats = self.graph_builder.build_from_documents(
            all_chunks,
            incremental=incremental
        )
        graph = self.graph_builder.graph
        print(f"Graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships")
        graph_path = os.path.join(database_path, 'graph.pkl')
        self.save_graph(graph_path)

        # Step 3: Save entity and relationship information
        entities_path = os.path.join(database_path, 'entities.pkl')
        entity_index = self.save_entities(entities_path)

        relationships_path = os.path.join(database_path, 'relationships.pkl')
        relationship_index = self.save_relationships(relationships_path)

        return all_chunks, vectorstore, graph, entity_index, relationship_index 

    def save_entities(self, path: str = None):
        """Save entity index to disk.

        Extract entity names and descriptions from graph, generate embeddings and save.
        Vectorization content format: "{entity_name}\n{description}" (reference LightRAG)

        Args:
            path: Storage path (file path)

        Returns:
            entity_index: FAISS index object
        """
        import os
        import pickle

        # Get all entities from graph
        entity_names = list(self.graph_builder.graph.nodes())
        if not entity_names:
            print("No entities to save.")
            return

        # Collect entity metadata and vectorization texts (with progress bar)
        entity_metadata = []
        entity_texts = []
        print(f"[save_entities] Processing {len(entity_names)} entities...")
        for entity in tqdm(entity_names, desc="  Collecting"):
            node_data = self.graph_builder.graph.nodes[entity]
            description = node_data.get("description", "")

            # LightRAG approach: entity name + description combined as vectorization content
            entity_content = f"{entity}\n{description}"

            entity_metadata.append({
                "name": entity,
                "type": node_data.get("type", "Entity"),
                "description": description,
                "chunk_ids": node_data.get("chunk_ids", []),
                "content": entity_content  # Save complete content for vectorization
            })
            entity_texts.append(entity_content)

        # Generate entity embeddings (using complete content not just entity name)
        print(f"  Generating embeddings...")
        embeddings = self.embedding.encode(list(tqdm(entity_texts, desc="  Embedding")))

        # Create FAISS index
        dim = embeddings.shape[1]
        entity_index = faiss.IndexFlatIP(dim)  # Inner product similarity
        entity_index.add(embeddings)

        # Save entity index to specified file path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "index": entity_index,
                "metadata": entity_metadata
            }, f)

        print(f"Saved {len(entity_names)} entities to {path}")
        return entity_index

    def save_relationships(self, path: str = None):
        """Save relationship index to disk.

        Extract relationship information from graph, generate embeddings and save.
        Vectorization content format: "{keywords}\t{src_id}\n{tgt_id}\n{description}" (reference LightRAG)

        Args:
            path: Storage path (file path)

        Returns:
            relationship_index: FAISS index object
        """
        import os
        import pickle

        # Get all relationships from graph
        relationships = list(self.graph_builder.graph.edges(data=True))
        if not relationships:
            print("No relationships to save.")
            return

        # Collect relationship metadata and vectorization texts (with progress bar)
        relationship_metadata = []
        relationship_texts = []
        print(f"[save_relationships] Processing {len(relationships)} relationships...")
        for src, tgt, data in tqdm(relationships, desc="  Collecting"):
            rel_description = data.get("description", f"{src} related to {tgt}")
            keywords = data.get("keywords", "")

            # LightRAG approach: keywords + source entity + target entity + description combined
            # Format: "{keywords}\t{src_id}\n{tgt_id}\n{description}"
            rel_content = f"{keywords}\t{src}\n{tgt}\n{rel_description}"

            relationship_metadata.append({
                "src_id": src,
                "tgt_id": tgt,
                "rel_type": data.get("rel_type", "RELATED_TO"),
                "description": rel_description,
                "keywords": keywords,
                "weight": data.get("weight", 1.0),
                "content": rel_content  # Save complete content for vectorization
            })
            relationship_texts.append(rel_content)

        # Generate relationship embeddings (using complete content not just description)
        print(f"  Generating embeddings...")
        embeddings = self.embedding.encode(list(tqdm(relationship_texts, desc="  Embedding")))

        # Create FAISS index
        dim = embeddings.shape[1]
        relationship_index = faiss.IndexFlatIP(dim)  # Inner product similarity
        relationship_index.add(embeddings)

        # Save relationship index to specified file path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "index": relationship_index,
                "metadata": relationship_metadata
            }, f)

        print(f"Saved {len(relationship_texts)} relationships to {path}")
        return relationship_index
    


def get_graphrag_indexer(
    chunker=None,
    embedding=None,
    max_workers: int = 16,
    enable_thinking: bool = False,
    extract_mode: str = "thread",
) -> GraphRAGIndexer:
    """Get a GraphRAGIndexer instance

    Args:
        chunker: Pre-created chunker instance
        embedding: Pre-created Embeddings instance
        max_workers: Maximum number of parallel workers for extraction
        enable_thinking: Whether to enable LLM thinking mode (default False)
        extract_mode: Extraction mode (thread/async/sync)

    Returns:
        GraphRAGIndexer instance
    """
    return GraphRAGIndexer(
        chunker=chunker,
        embedding=embedding,
        max_workers=max_workers,
        enable_thinking=enable_thinking,
        extract_mode=extract_mode,
    )


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    print("=" * 60)
    print("GraphRAGIndexer test: full index_documents workflow")
    print("=" * 60)

    # Test: create indexer and index documents
    print("\n[Test] Create indexer and index documents...")
    indexer = get_graphrag_indexer(max_workers=8)
    print("  ✓ Indexer created")

    # Prepare test documents
    texts = [
        "Beijing is the capital of China, located in the North China Plain. Beijing has rich historical and cultural heritage, including the Forbidden City and Temple of Heaven.",
        "Beijing's GDP exceeds 4 trillion yuan, making it one of China's four municipalities. Beijing has numerous universities and research institutes.",
        "Shanghai is China's largest city, located at the Yangtze River estuary. Shanghai is an international financial center.",
        "Shanghai Pudong New Area is the forefront of China's reform and opening up. Shanghai Port is the world's largest container port.",
    ]
    documents = [Document(page_content=t, metadata={"source": f"text{i+1}"}) for i, t in enumerate(texts)]

    # Execute indexing
    chunk_count = indexer.index_documents(documents)
    print(f"  ✓ Indexed {chunk_count} chunks")

    # Verify results
    print("\n[Verify] Checking results...")

    # Check vectorstore
    if Path('./database/vectorstore/index.faiss').exists():
        print("  ✓ Vectorstore saved")

    # Check graph
    if Path('./database/graph.pkl').exists():
        print("  ✓ Graph saved")

    # Check entities
    if Path('./database/entities.pkl').exists():
        import pickle
        with open('./database/entities.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"  ✓ Entities saved: {data['index'].ntotal} entities")
        print(f"    Sample: {data['metadata'][0] if data['metadata'] else 'None'}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
