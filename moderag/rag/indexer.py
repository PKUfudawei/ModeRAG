"""
Indexer 模块 - 整合 chunker 和 embedder 进行文档索引
"""

import os
from typing import List, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS

from moderag.models.chunker import get_chunker
from moderag.models.embedding import get_embedding


class Indexer:
    """文档索引器 - 整合分块和嵌入功能"""

    def __init__(
        self,
        chunker: Optional[TokenTextSplitter] = None,
        embedding: Optional[Embeddings] = None,
    ):
        """
        初始化索引器

        Args:
            chunker: 预创建的 chunker 实例
            embedding: 预创建的 Embeddings 实例
            chunk_size: 每个 chunk 的 token 数量
            overlap: 相邻 chunk 之间的重叠 token 数量
            truncations: 在哪些关键词处截断文档
            embed_model: 嵌入模型名称
            embed_device: 嵌入模型设备
            chunk_model: tiktoken encoding 名称
        """
        self.chunker = chunker or get_chunker()
        self.embedding = embedding or get_embedding()

    def index_documents(
        self,
        documents: List[Document],
        output_path: str = None
    ) -> tuple:
        """
        一站式索引文档：分块 + 构建向量索引 + 保存

        返回 (chunks, vectorstore) 元组。

        Args:
            documents: langchain Document 列表
            output_path: 向量索引保存路径（可选，不传则只返回 chunks 和 vectorstore 对象）

        Returns:
            (chunks, vectorstore) 元组
        """
        # Step 1: 分块
        print(f"[Indexer] Chunking {len(documents)} documents...")
        chunks = self.chunker.split_documents(documents)
        print(f"  Generated {len(chunks)} chunks")

        # Step 2: 构建向量索引（带进度条）
        print(f"[Indexer] Building vectorstore...")
        vectorstore = self.build_vectorstore(chunks)

        # Step 3: 保存（如果指定了路径）
        if output_path:
            self.save_vectorstore(vectorstore, output_path)

        return chunks, vectorstore

    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        构建 FAISS 向量索引

        Args:
            documents: 文档列表

        Returns:
            FAISS 向量存储实例
        """
        # 直接使用 FAISS.from_documents，它会自动调用 embedding.embed_documents
        embed_model = self.embedding.embed_model if hasattr(self.embedding, 'embed_model') else self.embedding
        return FAISS.from_documents(documents, embed_model)

    def save_vectorstore(self, vectorstore: FAISS, path: str):
        """保存向量存储"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        vectorstore.save_local(path)
        print(f"Vectorstore saved to {path}")

    def load_vectorstore(self, path: str) -> FAISS:
        """加载向量存储"""
        embed_model = self.embedding.embed_model if hasattr(self.embedding, 'embed_model') else self.embedding
        return FAISS.load_local(
            path,
            embed_model,
            allow_dangerous_deserialization=True
        )


def get_indexer(
    chunker: Optional[TokenTextSplitter] = None,
    embedding = None,
) -> Indexer:
    """
    获取索引器实例

    Args:
        chunker: 预创建的 chunker 实例
        chunk_size: 每个 chunk 的 token 数量
        overlap: 相邻 chunk 之间的重叠 token 数量
        truncations: 在哪些关键词处截断文档
        embed_model: 嵌入模型名称
        embed_device: 嵌入模型设备
        chunk_model: tiktoken encoding 名称

    Returns:
        Indexer 实例
    """
    chunker = chunker or get_chunker()
    embedding = embedding or get_embedding()
    return Indexer(
        chunker=chunker,
        embedding=embedding,
    )


if __name__ == "__main__":
    texts = ["""
    This is the main content of the document.
    It contains important information about the topic.
    More content here to make it longer.

    # Acknowledgements
    We would like to thank the following people for their contributions.
    This section should be truncated from the main content.
    """] * 3

    # 将 texts 转换为 Document 对象用于测试
    documents = [Document(page_content=t, metadata={"source": f"doc_{i}"}) for i, t in enumerate(texts)]

    # 测试：index_documents 返回 (chunks, vectorstore)
    print("=" * 60)
    print("Test: index_documents")
    chunker = get_chunker(chunk_size=100, overlap=20, truncations=['acknowledgement', 'acknowledgment'])
    indexer = get_indexer(chunker=chunker)
    chunks, vectorstore = indexer.index_documents(documents)
    print(f"Generated {len(chunks)} chunks")
    print(f"Vectorstore size: {vectorstore.index.ntotal}")
    for i, doc in enumerate(chunks[:3]):
        print(f"- Chunk {i}: {doc.page_content.strip()[:100]}...")
