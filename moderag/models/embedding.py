from typing import Optional, List
import numpy as np
import asyncio
import threading
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


# Process-wide singleton cache for embedding models.  Loading BAAI/bge-m3 (or
# any transformer embedding model) from disk is expensive, and the executor
# used to create a fresh tool per task, which reloaded the model every time.
# key: (model, device, wrapped) -> EmbeddingWrapper | HuggingFaceEmbeddings
_EMBEDDING_CACHE: dict = {}
_EMBEDDING_CACHE_LOCK = threading.Lock()


class EmbeddingWrapper:
    """Wrapper to adapt LangChain embed_model to encode interface."""

    def __init__(self, embed_model):
        self.embed_model = embed_model
        # Concurrent encode calls on the same transformer model are not
        # guaranteed thread-safe; serialize access with a per-wrapper lock.
        self._lock = threading.Lock()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the wrapped embed_model."""
        with self._lock:
            embeddings = self.embed_model.embed_documents(texts)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text."""
        with self._lock:
            return self.embed_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents."""
        with self._lock:
            return self.embed_model.embed_documents(texts)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text (alias for embed_query)."""
        return self.embed_query(text)

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a query text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)


def get_embedding(model="BAAI/bge-m3", device="cpu", wrapped=True):
    """
    获取嵌入模型实例（进程级单例缓存）

    Args:
        model: 嵌入模型名称
        device: 运行设备 (默认 cpu)
        wrapped: 是否返回 EmbeddingWrapper，False 时返回原始 HuggingFaceEmbeddings (用于 ragas)

    Returns:
        缓存的 EmbeddingWrapper 或 HuggingFaceEmbeddings 实例
    """
    key = (model, device, wrapped)
    with _EMBEDDING_CACHE_LOCK:
        cached = _EMBEDDING_CACHE.get(key)
        if cached is not None:
            return cached

        # 只传递 device 参数，避免 device_map 导致的问题
        model_kwargs = {"device": device}

        hf_embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
            show_progress=False
        )

        result = EmbeddingWrapper(hf_embeddings) if wrapped else hf_embeddings
        _EMBEDDING_CACHE[key] = result
        return result


if __name__ == "__main__":
    # 获取嵌入模型实例
    embed_model = get_embedding()

    # The queries and documents to embed
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    # Use the encode method from EmbedderWrapper
    query_embeddings = embed_model.encode(queries)
    document_embeddings = embed_model.encode(documents)

    # Compute cosine similarity manually using numpy
    import numpy as np

    # Normalize embeddings
    print("Norm of query:", np.linalg.norm(query_embeddings, axis=-1))
    print("Norm of doc:", np.linalg.norm(document_embeddings, axis=-1))

    similarity = query_embeddings.dot(document_embeddings.T)
    print(similarity)
    # Similar to: tensor([[0.7646, 0.1414],
    #                     [0.1355, 0.6000]])
