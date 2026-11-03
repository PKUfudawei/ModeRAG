"""
Chunker 模块 - 文档分块功能
"""

import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from tqdm import tqdm


class Chunker(TokenTextSplitter):
    """自定义分块器，支持在指定关键词处截断文档"""

    def __init__(
        self,
        model: str = "cl100k_base",
        chunk_size: int = 512,
        overlap: int = 50,
        truncations: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        初始化分块器

        Args:
            model: tiktoken encoding 名称
            chunk_size: 每个 chunk 的 token 数量
            overlap: 相邻 chunk 之间的重叠 token 数量
            truncations: 在哪些关键词处截断文档（不区分大小写）
            **kwargs: 其他参数传递给 TokenTextSplitter
        """
        import tiktoken

        # 保存 truncations 到实例（None 表示不截断）
        self.truncations = truncations
        # 编译正则表达式
        if self.truncations:
            patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in self.truncations]
            self._patterns = patterns
        else:
            self._patterns = []

        # 调用父类初始化
        TokenTextSplitter.__init__(
            self,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            **kwargs,
        )
        # 设置 encoding
        self._encoding_name = model
        self._encoding = tiktoken.get_encoding(model)

    def truncate(self, text: str) -> str:
        """
        在 truncations 关键词处截断文本

        Args:
            text: 原始文本

        Returns:
            截断后的文本（只保留第一个截断点之前的内容）
        """
        if not self._patterns:
            return text

        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                return text[:match.start()]

        return text

    def split_text(self, text: str) -> List[str]:
        """
        先截断再分块

        Args:
            text: 原始文本

        Returns:
            分块后的文本列表
        """
        truncated = self.truncate(text)
        return super().split_text(truncated)

    def split_documents(
        self, documents: List[Document], **kwargs
    ) -> List[Document]:
        """
        先截断文档再分块，并为每个 chunk 添加 metadata:
        - chunk_id: "{file_hash}_{chunk_index}" 格式
        - file_hash: 基于 page_content 的 md5 hash（前 16 位）
        - source: 原文档的 source 字段
        - 保留原文档的所有 metadata（如 book_title, authors, context_id 等）

        Args:
            documents: 原始文档列表

        Returns:
            分块后的文档列表
        """
        import hashlib

        all_chunks = []

        for doc in tqdm(documents, desc='Chunking documents'):
            # 截断文本
            truncated_text = self.truncate(doc.page_content)

            # 计算 file_hash（基于 page_content，更可靠）
            file_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]

            # 分块（create_documents 会保留原始 metadata）
            doc_chunks = self.create_documents([truncated_text], metadatas=[doc.metadata])

            # 为每个 chunk 添加 chunk_id 和 file_hash
            for idx, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_id"] = f"{file_hash}_{idx}"
                chunk.metadata["file_hash"] = file_hash
                all_chunks.append(chunk)

        print(f"Generated {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


def get_chunker(
    model: str = "cl100k_base",
    chunk_size: int = 512,
    overlap: int = 50,
    truncations: Optional[List[str]] = None,
    **kwargs,
) -> Chunker:
    """
    获取分块器实例

    Args:
        model: tiktoken encoding 名称
        chunk_size: 每个 chunk 的 token 数量
        overlap: 相邻 chunk 之间的重叠 token 数量
        truncations: 在哪些关键词处截断文档（不区分大小写）
                     None (默认) 使用默认截断词 ['acknowledgement', 'acknowledgment']
                     [] 禁用截断
                     列表 使用自定义截断词
        **kwargs: 其他参数传递给 Chunker

    Returns:
        Chunker 实例
    """
    return Chunker(
        model=model,
        chunk_size=chunk_size,
        overlap=overlap,
        truncations=truncations,
        **kwargs,
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

    # 将 texts 转换为 Document 对象用于 split_documents 测试
    documents = [Document(page_content=t, metadata={"source": f"doc_{i}"}) for i, t in enumerate(texts)]

    # 测试 1: 默认 truncations=['acknowledgement', 'acknowledgment'] - split_documents
    print("=" * 60)
    print("Test 1:")
    chunker1 = get_chunker(chunk_size=100, overlap=20)
    result1 = chunker1.split_documents(documents)
    for i, doc in enumerate(result1):
        print(f"  Chunk {i}: {doc.page_content.strip()}")

    # 测试 2: truncations=[] (禁用) - split_documents
    print()
    print("=" * 60)
    print("Test 2:")
    chunker2 = get_chunker(chunk_size=100, overlap=20, truncations=[])
    result2 = chunker2.split_documents(documents)
    for i, doc in enumerate(result2):
        print(f"  Chunk {i}: {doc.page_content.strip()}")

    # 测试 3: 自定义 truncations=['topic'] - split_documents
    print()
    print("=" * 60)
    print("Test 3:")
    chunker3 = get_chunker(chunk_size=200, overlap=20, truncations=['topic'])
    result3 = chunker3.split_documents(documents)
    for i, doc in enumerate(result3):
        print(f"  Chunk {i}: {doc.page_content.strip()}")

    # 测试 4: 自定义 truncations=['#'] (在 markdown 标题处截断) - split_documents
    print()
    print("=" * 60)
    print("Test 4:")
    chunker4 = get_chunker(chunk_size=200, overlap=20, truncations=['acknowledgement'])
    result4 = chunker4.split_documents(documents)
    for i, doc in enumerate(result4):
        print(f"- Chunk {i}: {doc.page_content.strip()}")
