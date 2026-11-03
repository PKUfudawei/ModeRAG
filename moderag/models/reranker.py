from typing import List, Optional
import threading
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# Process-wide singleton cache for the reranker CrossEncoder (same reason as the
# embedding cache: constructing a CrossEncoder reloads the model from disk).
# key: (model, device, top_k) -> SimpleReranker
_RERANKER_CACHE: dict = {}
_RERANKER_CACHE_LOCK = threading.Lock()


class SimpleReranker:
    """Simple reranker based on CrossEncoder"""
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda:0", top_k: int = 5):
        self.model = CrossEncoder(model, device=device)
        self.top_k = top_k
        # Serialize predict() calls on the shared model instance.
        self._lock = threading.Lock()

    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """Rerank documents based on query relevance"""
        scored_docs = self.rerank_with_score(query, documents, top_k)
        result_docs = []
        for doc, score in scored_docs:
            new_doc = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
            new_doc.metadata["rerank_score"] = float(score)
            result_docs.append(new_doc)

        return result_docs

    def rerank_with_score(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """Rerank documents and return results with scores"""
        texts = [(query, doc.page_content) for doc in documents]
        with self._lock:
            scores = self.model.predict(texts)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        k = top_k if top_k is not None else self.top_k
        scored_docs = scored_docs[:k] if k else scored_docs

        return scored_docs


def get_reranker(model="BAAI/bge-reranker-v2-m3", device='cuda:0', top_k: int = 5):
    """Get a cached SimpleReranker instance (default top_k=5)"""
    key = (model, device, top_k)
    with _RERANKER_CACHE_LOCK:
        cached = _RERANKER_CACHE.get(key)
        if cached is None:
            cached = SimpleReranker(model=model, device=device, top_k=top_k)
            _RERANKER_CACHE[key] = cached
        return cached
