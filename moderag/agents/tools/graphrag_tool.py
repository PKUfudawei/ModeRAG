"""GraphRAG tool wrapper"""
import os
import pickle
import threading
from typing import Optional, Dict, Any

from .base import BaseTool, ToolResult


# Process-level shared cache of loaded GraphRAG resources keyed by storage_path.
# Each executor task used to construct a fresh GraphRAGTool whose _initialize()
# re-loaded graph.pkl / entities.pkl / relationships.pkl and the FAISS
# vectorstore from disk (plus the embedding model).  Share them per process so
# the heavy I/O happens once; index reads are then cheap and thread-safe.
_SHARED_LOCK = threading.Lock()
_SHARED_RESOURCES: Dict[str, Dict[str, Any]] = {}


class GraphRAGTool(BaseTool):
    """GraphRAG retrieval tool (LightRAG style)

    - naive: pure vector search with rerank (recall 20, rerank 5)
    - local: entity + graph traversal with rerank (recall 40, rerank 5)
    - global: relationship search with rerank (recall 40, rerank 5)
    """
    def __init__(
        self,
        storage_path: str = "./storage/graphrag_index",
        recall_top_k: int = 20,
        rerank_top_k: int = 5,
        max_recall_chunks: int = 40,
        top_k_entities: int = 10,
        top_k_relationships: int = 5,
        reranker_model: Optional[str] = None,
        reranker_device: str = "cuda:0"
    ):
        self.storage_path = storage_path
        self.recall_top_k = recall_top_k
        self.rerank_top_k = rerank_top_k
        self.max_recall_chunks = max_recall_chunks
        self.top_k_entities = top_k_entities
        self.top_k_relationships = top_k_relationships
        self.reranker_model = reranker_model
        self.reranker_device = reranker_device
        self._graph = None
        self._entity_index = None
        self._entity_metadata = None
        self._relationship_index = None
        self._relationship_metadata = None
        self._vectorstore = None
        self._embedding = None
        self._reranker = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization (loads shared index resources once per process)."""
        if self._initialized:
            return

        with _SHARED_LOCK:
            res = _SHARED_RESOURCES.get(self.storage_path)
            if res is None:
                from moderag.models.embedding import get_embedding
                from langchain_community.vectorstores import FAISS

                graph_path = os.path.join(self.storage_path, 'graph.pkl')
                with open(graph_path, "rb") as f:
                    graph = pickle.load(f)

                entities_path = os.path.join(self.storage_path, 'entities.pkl')
                with open(entities_path, "rb") as f:
                    entities_data = pickle.load(f)

                relationships_path = os.path.join(self.storage_path, 'relationships.pkl')
                with open(relationships_path, "rb") as f:
                    relationships_data = pickle.load(f)

                embedding = get_embedding(model="BAAI/bge-m3", device="cpu")
                vectorstore = FAISS.load_local(
                    os.path.join(self.storage_path, 'vectorstore'),
                    embeddings=embedding.embed_model,
                    allow_dangerous_deserialization=True,
                )
                res = {
                    "graph": graph,
                    "entity_index": entities_data["index"],
                    "entity_metadata": entities_data["metadata"],
                    "relationship_index": relationships_data["index"],
                    "relationship_metadata": relationships_data["metadata"],
                    "vectorstore": vectorstore,
                    "embedding": embedding,
                }
                _SHARED_RESOURCES[self.storage_path] = res

            self._graph = res["graph"]
            self._entity_index = res["entity_index"]
            self._entity_metadata = res["entity_metadata"]
            self._relationship_index = res["relationship_index"]
            self._relationship_metadata = res["relationship_metadata"]
            self._vectorstore = res["vectorstore"]
            self._embedding = res["embedding"]

            if self.reranker_model:
                try:
                    from moderag.models.reranker import get_reranker
                    self._reranker = get_reranker(
                        model=self.reranker_model,
                        device=self.reranker_device,
                        top_k=self.rerank_top_k
                    )
                except Exception as e:
                    print(f"Warning: Failed to initialize reranker: {e}")
                    self._reranker = None

        self._initialized = True

    def get_name(self) -> str:
        return "graphrag"

    def naive_search(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """Naive vector search with reranker"""
        try:
            self._initialize()
            from moderag.graphrag.retriever import get_graphrag_retriever
            retriever = get_graphrag_retriever(
                graph=self._graph,
                entity_index=self._entity_index,
                entity_metadata=self._entity_metadata,
                embedding=self._embedding,
                vectorstore=self._vectorstore,
                recall_top_k=self.recall_top_k,
                rerank_top_k=self.rerank_top_k,
                reranker=self._reranker  # Use reranker by default
            )
            docs = retriever.naive_search(query=query, top_k=top_k)
            return self._docs_to_tool_result(docs, "naive")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def local_search(self, query: str, top_k_entities: Optional[int] = None,
                     top_k_relationships: Optional[int] = None,
                     max_chunks: Optional[int] = None,
                     top_k: Optional[int] = None,
                     **kwargs) -> ToolResult:
        """Local search: entity retrieval + single-hop traversal + rerank (recall 40, rerank 5)"""
        try:
            self._initialize()
            from moderag.graphrag.retriever import get_graphrag_retriever
            retriever = get_graphrag_retriever(
                graph=self._graph,
                entity_index=self._entity_index,
                entity_metadata=self._entity_metadata,
                embedding=self._embedding,
                vectorstore=self._vectorstore,
                recall_top_k=self.recall_top_k,
                rerank_top_k=self.rerank_top_k,
                reranker=self._reranker  # Use reranker for local search
            )
            docs = retriever.local_search(
                query=query,
                top_k_entities=top_k_entities or self.top_k_entities,
                top_k_relationships=top_k_relationships or self.top_k_relationships,
                max_chunks=max_chunks or self.max_recall_chunks,
                top_k=top_k or self.rerank_top_k
            )
            return self._docs_to_tool_result(docs, "local")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def global_search(self, query: str, top_k_relationships: Optional[int] = None,
                      max_chunks: Optional[int] = None, top_k: Optional[int] = None,
                      **kwargs) -> ToolResult:
        """Global search: relationship search + rerank (recall 40, rerank 5)"""
        try:
            self._initialize()
            from moderag.graphrag.retriever import get_graphrag_retriever
            retriever = get_graphrag_retriever(
                graph=self._graph,
                entity_index=self._entity_index,
                entity_metadata=self._entity_metadata,
                relationship_index=self._relationship_index,
                relationship_metadata=self._relationship_metadata,
                embedding=self._embedding,
                vectorstore=self._vectorstore,
                recall_top_k=self.recall_top_k,
                rerank_top_k=self.rerank_top_k,
                reranker=self._reranker
            )
            docs = retriever.global_search(
                query=query,
                top_k_relationships=top_k_relationships or self.top_k_relationships,
                max_chunks=max_chunks or self.max_recall_chunks,
                top_k=top_k or self.rerank_top_k
            )
            return self._docs_to_tool_result(docs, "global")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def search(self, query: str, mode: str = "global", **kwargs) -> ToolResult:
        """Execute GraphRAG retrieval"""
        try:
            if mode == "naive":
                return self.naive_search(query, **kwargs)
            elif mode == "local":
                return self.local_search(query, **kwargs)
            else:
                return self.global_search(query, **kwargs)
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def _docs_to_tool_result(self, docs: list, mode: str) -> ToolResult:
        evidence = []
        answer_parts = []
        entities = []
        edges = []

        for i, doc in enumerate(docs):
            # Extract graph context (entities and edges) from metadata
            retrieval_type = doc.metadata.get("retrieval_type", "")
            if retrieval_type in ["local_graph", "global_graph"]:
                # This is the graph context document
                entities = doc.metadata.get("entities", [])
                # Parse edges from page_content
                for line in doc.page_content.split("\n"):
                    if "--" in line and "-->" in line:
                        # Parse: "  A --TYPE--> B"
                        parts = line.strip().split(" --")
                        if len(parts) >= 2:
                            src = parts[0].strip()
                            tgt_rel = parts[1].split("-->")
                            rel_type = tgt_rel[0].strip() if tgt_rel else ""
                            tgt = tgt_rel[1].strip() if len(tgt_rel) > 1 else ""
                            edges.append({
                                "src_id": src,
                                "tgt_id": tgt,
                                "rel_type": rel_type
                            })
                # Also include as evidence
                evidence_item = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "retrieval_type": retrieval_type,
                    "score": doc.metadata.get("score", 0.0),
                    "metadata": doc.metadata
                }
                evidence.append(evidence_item)
            else:
                # Regular chunk
                evidence_item = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "retrieval_type": retrieval_type,
                    "score": doc.metadata.get("score", 0.0),
                    "metadata": doc.metadata
                }
                evidence.append(evidence_item)
                type_str = {"naive": "Vector", "local": "Local Graph", "global": "Global"}.get(mode, "Result")
                answer_parts.append(f"[{type_str} Result {i+1}]: {doc.page_content}")

        answer = "\n\n".join(answer_parts) if answer_parts else "No relevant information found"
        return ToolResult(success=True, answer=answer, evidence=evidence, entities=entities, edges=edges)

    def structured_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        mode = payload.get("mode", "global")
        result = self.search(query=query, mode=mode, **payload)
        return {
            "success": result.success,
            "answer": result.answer,
            "retrieval_results": result.evidence,
            "entities": result.entities,
            "edges": result.edges,
            "error": result.error
        }
