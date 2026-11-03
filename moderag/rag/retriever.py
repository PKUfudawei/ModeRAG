"""
Retriever module - supports vector search, keyword search, and reranking
"""

from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever as LangchainBM25Retriever
from moderag.models.reranker import get_reranker


class Retriever:
    """RAG retriever supporting vector search, keyword search, and reranking"""

    def __init__(
        self,
        vectorstore: FAISS,
        reranker: Optional[Any] = None,
        top_k: int = 10,
        recall_top_k: int = 20,
        rerank_top_k: int = 5,
        bm25_retriever: Optional[LangchainBM25Retriever] = None,
    ):
        """
        Initialize the retriever

        Args:
            vectorstore: FAISS vector store
            reranker: Optional reranker instance
            top_k: Number of documents to return (final)
            recall_top_k: Number of documents to recall in first stage (default 20)
            rerank_top_k: Number of documents after reranking (default 5)
            bm25_retriever: Optional BM25 retriever instance
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.recall_top_k = recall_top_k
        self.rerank_top_k = rerank_top_k
        self.reranker = reranker
        self.bm25_retriever = bm25_retriever

    def set_bm25_retriever(self, documents: List[Document]):
        """
        Set up BM25 retriever

        Args:
            documents: List of documents to build BM25 index
        """
        self.bm25_retriever = LangchainBM25Retriever.from_documents(documents)

    def vector_search(self, query: str) -> List[Document]:
        """Vector search with two-stage recall and rerank

        Args:
            query: Query text

        Returns:
            List of retrieved documents
        """
        # Stage 1: Recall top_k documents
        docs = self.vectorstore.similarity_search(query, k=self.recall_top_k)

        # Stage 2: Rerank and return top rerank_top_k
        if self.reranker:
            docs = self.reranker.rerank(query, docs, top_k=self.rerank_top_k)
        else:
            docs = docs[:self.rerank_top_k]
        return docs

    def bm25_search(self, query: str) -> List[Document]:
        """BM25 keyword search with two-stage recall and rerank

        Args:
            query: Query text

        Returns:
            List of retrieved documents
        """
        if self.bm25_retriever is None:
            raise ValueError("BM25 retriever not initialized. Call set_bm25_retriever() first.")
        # Stage 1: Recall
        docs = self.bm25_retriever.get_relevant_documents(query, k=self.recall_top_k)

        # Stage 2: Rerank
        if self.reranker:
            docs = self.reranker.rerank(query, docs, top_k=self.rerank_top_k)
        else:
            docs = docs[:self.rerank_top_k]
        return docs

    def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[Document]:
        """
        Hybrid search combining vector and BM25 with two-stage recall and rerank

        Args:
            query: Query text
            k: Number of documents to return (uses top_k if not specified)
            vector_weight: Weight for vector search (default 0.5)
            bm25_weight: Weight for BM25 search (default 0.5)

        Returns:
            List of hybrid retrieved documents
        """
        k = k or self.top_k
        results = []

        # Stage 1: Recall from both sources
        vector_results = self.vectorstore.similarity_search(query, k=self.recall_top_k)
        vector_scores = self._normalize_scores(len(vector_results))

        # Use doc_key mapping to avoid duplicates
        doc_map = {}
        for doc, score in zip(vector_results, vector_scores):
            doc_key = hash(doc.page_content) % (10**9)
            doc_map[doc_key] = {"doc": doc, "vector_score": score, "bm25_score": 0}

        # BM25 recall (if initialized)
        if self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.get_relevant_documents(query, k=self.recall_top_k)
            bm25_scores = self._normalize_scores(len(bm25_results))

            for i, doc in enumerate(bm25_results):
                doc_key = hash(doc.page_content) % (10**9)
                if doc_key in doc_map:
                    doc_map[doc_key]["bm25_score"] = bm25_scores[i]
                else:
                    doc_map[doc_key] = {"doc": doc, "vector_score": 0, "bm25_score": bm25_scores[i]}

        # Compute fused scores
        results = list(doc_map.values())
        for r in results:
            r["final_score"] = vector_weight * r["vector_score"] + bm25_weight * r["bm25_score"]

        # Sort by fused score
        results.sort(key=lambda x: x["final_score"], reverse=True)

        # Stage 2: Rerank top results
        if self.reranker is not None:
            top_docs = [r["doc"] for r in results[:self.recall_top_k]]
            reranked_docs = self.reranker.rerank(query, top_docs, top_k=self.rerank_top_k)
            return reranked_docs

        return [r["doc"] for r in results[:k]]

    def _normalize_scores(self, n_docs: int) -> List[float]:
        """
        Normalize scores based on ranking

        Args:
            n_docs: Total number of documents

        Returns:
            List of normalized scores
        """
        if n_docs == 0:
            return []
        # Use reciprocal normalization
        scores = [1 / (i + 1) for i in range(n_docs)]
        total = sum(scores)
        return [s / total for s in scores]


def get_retriever(
    vectorstore: FAISS,
    top_k: int = 5,
    recall_top_k: int = 20,
    rerank_top_k: int = 5,
    reranker_model: Optional[str] = None,  # Default: None (no reranker)
    reranker_device: str = "cuda:1",
) -> Retriever:
    """
    Get a Retriever instance

    Args:
        vectorstore: FAISS vector store
        top_k: Number of documents to return (final)
        recall_top_k: Number of documents to recall in first stage (default 20)
        rerank_top_k: Number of documents after reranking (default 5)
        reranker_model: Reranker model name (default: None, set to model name to enable reranker)
        reranker_device: Reranker model device

    Returns:
        Retriever instance
    """
    reranker = get_reranker(
        model=reranker_model,
        device=reranker_device,
        top_k=rerank_top_k,
    ) if reranker_model else None
    return Retriever(
        vectorstore=vectorstore,
        top_k=top_k,
        recall_top_k=recall_top_k,
        rerank_top_k=rerank_top_k,
        reranker=reranker,
    )


if __name__ == "__main__":
    from moderag.rag.indexer import get_indexer
    from moderag.models.chunker import get_chunker
    from moderag.models.embedding import get_embedding

    print("=" * 60)
    print("Retriever module test")
    print("=" * 60)

    # Create vector index using indexer
    chunker = get_chunker(model="cl100k_base", chunk_size=100, overlap=20)
    embedding = get_embedding(model="BAAI/bge-m3", device="cuda:0")
    indexer = get_indexer(chunker=chunker, embedding=embedding)

    texts = [
        "The capital of China is Beijing. Beijing is the capital city of the People's Republic of China.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects.",
        "Python is a programming language. Python is widely used for web development and data science.",
        "Machine learning is a subset of artificial intelligence. ML algorithms learn from data.",
        "JavaScript is a programming language for web development. It runs in browsers.",
    ]

    documents = [Document(page_content=t, metadata={"source": f"doc{i+1}"}) for i, t in enumerate(texts)]
    chunks = indexer.index_documents(documents)
    vectorstore = indexer.build_vectorstore(chunks)
    print(f"✓ Indexer created {len(chunks)} chunks")
    print()

    # Test 1: Retriever without reranker
    print("-" * 40)
    print("Test 1: Retriever without reranker")
    print("-" * 40)
    retriever_no_rerank = Retriever(
        vectorstore=vectorstore,
        top_k=3,
        recall_top_k=20,
        rerank_top_k=5,
        reranker=None,
    )
    results = retriever_no_rerank.vector_search("What is the capital of China?")
    print(f"✓ Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:50]}...")
    print()

    # Test 2: Create retriever with reranker using get_retriever
    print("-" * 40)
    print("Test 2: Create retriever with reranker using get_retriever")
    print("-" * 40)
    retriever = get_retriever(
        vectorstore,
        top_k=5,
        recall_top_k=20,
        rerank_top_k=3,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_device="cuda:1",
    )
    results = retriever.vector_search("What is the capital of China?")
    print(f"✓ Retrieved {len(results)} documents (rerank_top_k=3):")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:50]}...")
    print()

    # Test 3: hybrid_search method (with BM25) - requires rank_bm25 dependency
    print("-" * 40)
    print("Test 3: hybrid_search method (with BM25)")
    print("-" * 40)
    try:
        retriever.set_bm25_retriever(chunks)
        results = retriever.hybrid_search("programming language for web")
        print(f"✓ hybrid_search retrieved {len(results)} documents:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:50]}...")
    except ImportError as e:
        print(f"⚠ Skipped: {e}")
    print()

    # Test 4: Different query
    print("-" * 40)
    print("Test 4: Different query - 'machine learning'")
    print("-" * 40)
    results = retriever.vector_search("machine learning")
    print(f"✓ Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:50]}...")
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
