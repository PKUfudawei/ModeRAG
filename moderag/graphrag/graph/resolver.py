"""Entity resolution using FAISS."""
from collections import Counter
from typing import Dict, List

import faiss
import numpy as np
from tqdm import tqdm

from moderag.models.embedding import get_embedding


class Resolver:
    """Resolve entity aliases using embeddings and FAISS.

    Args:
        embed_model: Embedding model with encode method (EmbeddingWrapper).
        threshold: Similarity threshold. Default is 0.9.
    """

    def __init__(self, embed_model, threshold: float = 0.9):
        self.embed_model = embed_model
        self.threshold = threshold

    def find_aliases(self, entity_names: List[str]) -> Dict[str, str]:
        """Find entity aliases using embedding similarity.

        Args:
            entity_names: List of entity names.

        Returns:
            Alias mapping dictionary {alias: canonical_name}.
        """
        if not entity_names:
            return {}

        freq = Counter(entity_names)
        unique_names = list(freq.keys())

        print(f"[Resolver] Generating embeddings for {len(unique_names)} unique entities...")
        embeddings = self.embed_model.encode(list(tqdm(unique_names, desc="  Embedding")))
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        index.add(embeddings)

        print(f"[Resolver] Clustering similar entities...")
        alias_map = self._cluster_similar_entities(index, embeddings, unique_names, freq)
        return alias_map

    def _cluster_similar_entities(
        self,
        index: faiss.Index,
        embeddings: np.ndarray,
        names: List[str],
        freq: Counter,
    ) -> Dict[str, str]:
        """Cluster similar entities using Union-Find algorithm."""
        n = len(names)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        for i, name in enumerate(names):
            D, I = index.search(embeddings[i : i + 1], min(50, n))
            for score, idx in zip(D[0], I[0]):
                if idx >= 0 and score >= self.threshold and idx != i:
                    union(i, idx)

        clusters = {}
        for i, name in enumerate(names):
            root = find(i)
            clusters.setdefault(root, []).append(name)

        alias_map = {}
        for cluster in clusters.values():
            if not cluster:
                continue
            canonical = max(cluster, key=lambda x: (freq[x], x))
            alias_map.update({n: canonical for n in cluster})

        return alias_map


def get_resolver(embed_model=None, threshold: float = 0.9) -> Resolver:
    """Get an entity resolver instance.

    Args:
        threshold: Similarity threshold. Default is 0.9.
        embed_model: Optional EmbeddingWrapper instance. If None, uses default.

    Returns:
        Resolver instance.
    """
    embed_model = embed_model or get_embedding()

    return Resolver(embed_model=embed_model, threshold=threshold)


if __name__ == "__main__":
    resolver = get_resolver(threshold=0.85)

    names = ["北京", "北京市", "中国首都", "上海"]
    alias_map = resolver.find_aliases(names)
    print(f"Input: {names}")
    print(f"Alias map: {alias_map}")
