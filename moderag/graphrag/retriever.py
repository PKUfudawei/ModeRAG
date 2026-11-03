"""GraphRAG Retriever - vector search + graph traversal (LightRAG style)"""
from __future__ import annotations
from typing import List, Optional, Set, Tuple, Dict, Any
import numpy as np
import json
import logging

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_core.messages import HumanMessage, SystemMessage

from moderag.rag.retriever import Retriever

logger = logging.getLogger(__name__)


# Keywords extraction prompt (LightRAG style)
KEYWORDS_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are an expert keyword extractor for a GraphRAG retrieval system.

---Task---
Extract two types of keywords from the user query:
1. **high_level_keywords**: Overarching concepts, themes, and abstract ideas (for global/multi-hop graph search)
2. **low_level_keywords**: Specific entities, proper nouns, technical terms, and concrete details (for local/entity search)

---Instructions---
1. Output ONLY a valid JSON object with no additional text or markdown formatting
2. All keywords must be derived from the query
3. Keywords should be concise words or meaningful phrases
4. Both keyword categories must contain content for meaningful queries
5. For vague or nonsensical queries, return empty lists

---Output Format---
{{
  "high_level_keywords": ["concept1", "concept2"],
  "low_level_keywords": ["entity1", "entity2"]
}}

---Examples---
Query: "How does the EIC collider compare to EicC in terms of physics capabilities?"
Output: {{"high_level_keywords": ["collider comparison", "physics capabilities", "research potential"], "low_level_keywords": ["EIC", "EicC", "collider"]}}

Query: "What is the luminosity range of the EIC?"
Output: {{"high_level_keywords": ["luminosity", "performance metrics"], "low_level_keywords": ["EIC", "luminosity range"]}}
"""

KEYWORDS_EXTRACTION_HUMAN_PROMPT = """User Query: {query}

Output:"""


class GraphRAGRetriever(Retriever):
    """GraphRAG retriever with naive/local/global search modes (LightRAG style)

    Local Search (LightRAG style):
        1. Entities: embedding 粗排 (top_k_entities) → degree 细排
        2. Relations: BFS 扩展 from entities → (degree, weight) 双排序
        3. Chunks: collect from entities → reranker 细排

    Global Search (LightRAG style):
        1. Relations: embedding 粗排 (top_k_relationships) → keep cosine order
        2. Entities: extract from relations → no sorting/limit
        3. Chunks: collect from relations → reranker 细排
    """
    def __init__(
        self,
        graph,
        entity_index: Optional[faiss.Index] = None,
        entity_metadata: Optional[List[dict]] = None,
        relationship_index: Optional[faiss.Index] = None,
        relationship_metadata: Optional[List[dict]] = None,
        embedding=None,
        vectorstore=None,
        top_k: int = 5,
        recall_top_k: int = 20,
        rerank_top_k: int = 5,
        reranker=None,
        llm=None,
    ):
        super().__init__(
            vectorstore=vectorstore,
            top_k=top_k,
            recall_top_k=recall_top_k,
            rerank_top_k=rerank_top_k,
            reranker=reranker
        )

        self.graph = graph
        self.entity_index = entity_index
        self.entity_metadata = entity_metadata or []
        self.relationship_index = relationship_index
        self.relationship_metadata = relationship_metadata or []
        self.embedding = embedding
        self.llm = llm
        self._llm_initialized = False

    def _initialize_llm(self):
        """Initialize LLM if not already initialized."""
        if self._llm_initialized:
            return
        if self.llm is None:
            from moderag.models.llm import get_llm
            self.llm = get_llm(temperature=0)
        self._llm_initialized = True

    def _attach_chunk_scores(self, query: str, chunks: List[Document]) -> None:
        """Attach a per-chunk query--chunk cosine similarity to each chunk.

        Cross-mode evidence fusion has to rank chunks from different modes
        against each other.  Naive results previously carried no score at all,
        and graph modes assigned every collected chunk the same mode-level
        ``graph_score`` (the similarity of the recalled entity/relationship,
        not of the chunk).  With no per-chunk score the merge in
        ``Executor._merge_evidence`` degenerates to input order, which puts the
        vector-floor passages last and lets the evidence cap truncate them.

        The score is computed with the shared embedding function over a single
        batch (query first, then the chunks), so all modes report the same
        quantity: cosine similarity between the sub-query and the passage.
        """
        if not chunks or self.embedding is None:
            return
        try:
            vectors = self.embedding.encode([query] + [c.page_content for c in chunks])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("chunk scoring failed, leaving scores unset: %s", exc)
            return

        query_vec = np.asarray(vectors[0], dtype="float32")
        query_norm = float(np.linalg.norm(query_vec)) or 1.0
        for doc, vector in zip(chunks, vectors[1:]):
            vec = np.asarray(vector, dtype="float32")
            denom = query_norm * (float(np.linalg.norm(vec)) or 1.0)
            doc.metadata["score"] = float(np.dot(query_vec, vec) / denom)
            doc.metadata["score_type"] = "cosine_query_chunk"

    def _extract_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract high-level and low-level keywords from query using LLM (LightRAG style).

        Returns:
            Tuple of (high_level_keywords, low_level_keywords)
        """
        self._initialize_llm()

        try:
            messages = [
                SystemMessage(content=KEYWORDS_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=KEYWORDS_EXTRACTION_HUMAN_PROMPT.format(query=query))
            ]
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            keywords_data = json.loads(content)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])

            logger.debug(f"Extracted keywords from '{query}': HL={hl_keywords}, LL={ll_keywords}")
            return hl_keywords, ll_keywords

        except Exception as e:
            logger.warning(f"Keyword extraction failed for '{query}': {e}. Using query as fallback.")
            # Fallback: use the original query for both
            return [query], [query]

    def _search_entities(self, query: str, top_k: int) -> Tuple[List[dict], float]:
        """Search entities by embedding, return with degree for reranking (LightRAG style)

        Returns: List of {entity_name, cosine_score, degree} sorted by cosine (粗排结果)
        """
        if self.entity_index is None or self.entity_index.ntotal == 0:
            return [], 0.0

        query_embedding = self.embedding.encode([query])
        query_array = np.array(query_embedding, dtype=np.float32)
        scores, indices = self.entity_index.search(query_array, min(top_k, self.entity_index.ntotal))

        entities_with_degree = []
        max_score = 0.0
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.entity_metadata):
                metadata = self.entity_metadata[idx]
                entity_name = metadata.get("name", "")
                # Get degree from graph (number of connected edges)
                degree = self.graph.degree(entity_name) if entity_name in self.graph else 0
                entities_with_degree.append({
                    "entity_name": entity_name,
                    "cosine_score": float(score),
                    "degree": degree
                })
                if max_score == 0.0:
                    max_score = float(score)

        return entities_with_degree, max_score

    def _search_relationships(self, query: str, top_k: int) -> Tuple[List[dict], float]:
        """Search relationships by embedding, return with weight (LightRAG style)

        Returns: List of {src_id, tgt_id, rel_type, cosine_score, weight} sorted by cosine (粗排结果)
        Note: LightRAG keeps cosine order, does NOT rerank by weight
        """
        if self.relationship_index is None or self.relationship_index.ntotal == 0:
            return [], 0.0

        query_embedding = self.embedding.encode([query])
        query_array = np.array(query_embedding, dtype=np.float32)
        scores, indices = self.relationship_index.search(query_array, min(top_k, self.relationship_index.ntotal))

        relationships = []
        max_score = 0.0
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.relationship_metadata):
                metadata = self.relationship_metadata[idx]
                src_id = metadata["src_id"]
                tgt_id = metadata["tgt_id"]
                # Get weight from graph edge
                weight = 1.0
                if self.graph.has_edge(src_id, tgt_id):
                    weight = self.graph.edges[src_id, tgt_id].get("weight", 1.0)
                elif self.graph.has_edge(tgt_id, src_id):
                    weight = self.graph.edges[tgt_id, src_id].get("weight", 1.0)

                relationships.append({
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "rel_type": metadata.get("rel_type", "RELATED_TO"),
                    "cosine_score": float(score),
                    "weight": weight
                })
                if max_score == 0.0:
                    max_score = float(score)

        return relationships, max_score

    def _get_entities_from_relationships(self, relationships: List[dict]) -> Tuple[List[Node], List[Relationship], List[dict]]:
        """Extract entities from relationships (LightRAG global search style)

        Entities are extracted from relations WITHOUT sorting/limiting.
        Returns entities, relationships, and original relation data with weight.
        """
        graph = self.graph
        nodes: List[Node] = []
        node_map: dict = {}
        rels: List[Relationship] = []
        seen_entities: Set[str] = set()
        relation_data_list: List[dict] = []

        for rel in relationships:
            src_id = rel["src_id"]
            tgt_id = rel["tgt_id"]
            rel_type = rel.get("rel_type", "RELATED_TO")

            if src_id not in seen_entities and src_id in graph:
                node_data = graph.nodes[src_id]
                node = Node(id=src_id, type=node_data.get("type", "Entity"))
                nodes.append(node)
                node_map[src_id] = node
                seen_entities.add(src_id)

            if tgt_id not in seen_entities and tgt_id in graph:
                node_data = graph.nodes[tgt_id]
                node = Node(id=tgt_id, type=node_data.get("type", "Entity"))
                nodes.append(node)
                node_map[tgt_id] = node
                seen_entities.add(tgt_id)

            if src_id in node_map and tgt_id in node_map:
                rels.append(Relationship(source=node_map[src_id], target=node_map[tgt_id], type=rel_type))

            # Keep original relation data for chunk retrieval
            relation_data_list.append(rel)

        return nodes, rels, relation_data_list

    def _find_related_edges_from_entities(
        self, entities: List[dict], top_k: Optional[int] = None
    ) -> Tuple[List[Relationship], List[dict]]:
        """Find edges connected to entities and sort by (degree, weight) (LightRAG style)

        LightRAG local search:
        1. BFS expand edges from entities (no limit on expansion)
        2. Sort edges by (degree, weight) descending

        Args:
            entities: List of {entity_name, cosine_score, degree}
            top_k: Optional limit on returned edges

        Returns:
            Tuple of (Relationship objects, edge data list with degree/weight)
        """
        graph = self.graph
        node_map: dict = {}
        all_edges: List[dict] = []
        seen_edges: Set[tuple] = set()

        # Build node map
        for entity in entities:
            entity_name = entity["entity_name"]
            if entity_name in graph and entity_name not in node_map:
                node_data = graph.nodes[entity_name]
                node_map[entity_name] = Node(id=entity_name, type=node_data.get("type", "Entity"))

        # Collect all edges connected to entities (BFS style, no limit)
        for entity in entities:
            entity_name = entity["entity_name"]
            if entity_name not in graph:
                continue

            # Outgoing edges
            for target in graph.successors(entity_name):
                edge_key = (entity_name, target)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edge_data = graph.edges[entity_name, target]
                    # Calculate edge degree (sum of both endpoint degrees)
                    src_degree = graph.degree(entity_name)
                    tgt_degree = graph.degree(target)
                    edge_degree = src_degree + tgt_degree
                    weight = edge_data.get("weight", 1.0)
                    all_edges.append({
                        "src_id": entity_name,
                        "tgt_id": target,
                        "rel_type": edge_data.get("rel_type", "RELATED_TO"),
                        "degree": edge_degree,
                        "weight": weight,
                        "description": edge_data.get("description", ""),
                        "chunk_ids": edge_data.get("chunk_ids", [])
                    })

            # Incoming edges
            for source in graph.predecessors(entity_name):
                edge_key = (source, entity_name)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edge_data = graph.edges[source, entity_name]
                    src_degree = graph.degree(source)
                    tgt_degree = graph.degree(entity_name)
                    edge_degree = src_degree + tgt_degree
                    weight = edge_data.get("weight", 1.0)
                    all_edges.append({
                        "src_id": source,
                        "tgt_id": entity_name,
                        "rel_type": edge_data.get("rel_type", "RELATED_TO"),
                        "degree": edge_degree,
                        "weight": weight,
                        "description": edge_data.get("description", ""),
                        "chunk_ids": edge_data.get("chunk_ids", [])
                    })

        # Sort by (degree, weight) descending (LightRAG style)
        all_edges.sort(key=lambda x: (x["degree"], x["weight"]), reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            all_edges = all_edges[:top_k]

        # Build Relationship objects
        relationships: List[Relationship] = []
        for edge in all_edges:
            src_id, tgt_id = edge["src_id"], edge["tgt_id"]
            if src_id in node_map and tgt_id in node_map:
                relationships.append(
                    Relationship(
                        source=node_map[src_id],
                        target=node_map[tgt_id],
                        type=edge["rel_type"]
                    )
                )
            elif src_id in graph and tgt_id in graph:
                # Add missing nodes to node_map
                if src_id not in node_map:
                    node_map[src_id] = Node(id=src_id, type=graph.nodes[src_id].get("type", "Entity"))
                if tgt_id not in node_map:
                    node_map[tgt_id] = Node(id=tgt_id, type=graph.nodes[tgt_id].get("type", "Entity"))
                relationships.append(
                    Relationship(
                        source=node_map[src_id],
                        target=node_map[tgt_id],
                        type=edge["rel_type"]
                    )
                )

        return relationships, all_edges

    def _get_all_nodes_from_entities_and_edges(
        self, entities: List[dict], edges: List[dict]
    ) -> List[Node]:
        """Collect all unique nodes from entities and edges"""
        node_map: dict = {}

        # Add entities
        for entity in entities:
            entity_name = entity["entity_name"]
            if entity_name in self.graph and entity_name not in node_map:
                node_data = self.graph.nodes[entity_name]
                node_map[entity_name] = Node(id=entity_name, type=node_data.get("type", "Entity"))

        # Add nodes from edges
        for edge in edges:
            for node_id in [edge["src_id"], edge["tgt_id"]]:
                if node_id in self.graph and node_id not in node_map:
                    node_data = self.graph.nodes[node_id]
                    node_map[node_id] = Node(id=node_id, type=node_data.get("type", "Entity"))

        return list(node_map.values())

    def _get_chunks_by_entity_ids(self, entity_ids: List[str], max_chunks: int = 10) -> List[Document]:
        if self.vectorstore is None:
            return []

        chunk_id_count: dict[str, int] = {}
        chunk_id_to_entities: dict[str, list[str]] = {}

        for entity_id in entity_ids:
            if entity_id in self.graph:
                node_data = self.graph.nodes[entity_id]
                chunk_ids = node_data.get("chunk_ids", [])
                for chunk_id in chunk_ids:
                    chunk_id_count[chunk_id] = chunk_id_count.get(chunk_id, 0) + 1
                    if chunk_id not in chunk_id_to_entities:
                        chunk_id_to_entities[chunk_id] = []
                    chunk_id_to_entities[chunk_id].append(entity_id)

        sorted_chunk_ids = sorted(chunk_id_count.keys(), key=lambda x: chunk_id_count[x], reverse=True)

        chunk_id_to_key: dict[str, str] = {}
        for key, doc in self.vectorstore.docstore._dict.items():
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                chunk_id_to_key[chunk_id] = key

        chunks: List[Document] = []
        for chunk_id in sorted_chunk_ids[:max_chunks]:
            if chunk_id in chunk_id_to_key:
                doc = self.vectorstore.docstore.search(chunk_id_to_key[chunk_id])
                doc.metadata["related_entities"] = chunk_id_to_entities[chunk_id]
                doc.metadata["entity_count"] = chunk_id_count[chunk_id]
                chunks.append(doc)

        return chunks

    def _get_chunks_by_relationships(
        self, relationships: List[dict], max_chunks: int = 10
    ) -> List[Document]:
        """Get chunks from relationships (accepts dict format with chunk_ids)"""
        if self.vectorstore is None:
            return []

        chunk_id_count: dict[str, int] = {}
        chunk_id_to_relationships: dict[str, list[dict]] = {}

        for rel in relationships:
            # chunk_ids are already in the relation dict
            chunk_ids = rel.get("chunk_ids", [])
            if not chunk_ids:
                # Fallback: try to get from graph edge
                edge_data = self.graph.get_edge_data(rel["src_id"], rel["tgt_id"])
                if edge_data:
                    chunk_ids = edge_data.get("chunk_ids", [])

            for chunk_id in chunk_ids:
                if chunk_id:
                    chunk_id_count[chunk_id] = chunk_id_count.get(chunk_id, 0) + 1
                    if chunk_id not in chunk_id_to_relationships:
                        chunk_id_to_relationships[chunk_id] = []
                    chunk_id_to_relationships[chunk_id].append(rel)

        sorted_chunk_ids = sorted(chunk_id_count.keys(), key=lambda x: chunk_id_count[x], reverse=True)

        chunk_id_to_key: dict[str, str] = {}
        for key, doc in self.vectorstore.docstore._dict.items():
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                chunk_id_to_key[chunk_id] = key

        chunks: List[Document] = []
        for chunk_id in sorted_chunk_ids[:max_chunks]:
            if chunk_id in chunk_id_to_key:
                doc = self.vectorstore.docstore.search(chunk_id_to_key[chunk_id])
                doc.metadata["related_relationships"] = chunk_id_to_relationships[chunk_id]
                doc.metadata["relationship_count"] = chunk_id_count[chunk_id]
                chunks.append(doc)

        return chunks

    def naive_search(
        self, query: str, top_k: int = 5
    ) -> List[Document]:
        """Naive search: vector search with two-stage recall and rerank

        Args:
            query: Search query
            top_k: Number of chunks to return (default: 5)
        """
        if self.vectorstore is None:
            return []

        # Stage 1: Recall
        docs = self.vectorstore.similarity_search(query, k=self.recall_top_k)

        if not docs:
            return []

        # Stage 2: Rerank
        if self.reranker is not None:
            rerank_k = min(top_k, len(docs))
            docs = self.reranker.rerank(query, docs, top_k=rerank_k)
        else:
            docs = docs[:top_k]

        # Update metadata
        for i, doc in enumerate(docs):
            doc.metadata["retrieval_type"] = "naive"
            doc.metadata["rank"] = i
        self._attach_chunk_scores(query, docs)

        return docs

    def local_search(
        self,
        query: str,
        top_k_entities: int = 10,
        top_k_relationships: Optional[int] = None,
        max_chunks: int = 40,
        top_k: Optional[int] = 5,
        use_keyword_extraction: bool = True
    ) -> List[Document]:
        """Local search (LightRAG style):
        1. Extract low-level keywords from query
        2. Entities: embedding 粗排 (top_k_entities) using low-level keywords → degree 细排
        3. Relations: BFS expand from entities → (degree, weight) 双排序
        4. Chunks: collect from entities → reranker 细排

        Args:
            query: Search query
            top_k_entities: Number of entities for embedding 粗排 (before degree 细排)
            top_k_relationships: Max relationships to return after (degree, weight) sorting
            max_chunks: Max chunks to collect before reranking (default: 40)
            top_k: Final chunks to return after reranking (default: 5)
            use_keyword_extraction: Whether to use LLM-based keyword extraction (default: True)

        Returns:
            List of Document (reranked chunks + graph context)
        """
        final_top_k = top_k if top_k is not None else 5

        # Step 0: Extract low-level keywords for entity search (LightRAG style)
        search_query = query
        if use_keyword_extraction:
            hl_keywords, ll_keywords = self._extract_keywords(query)
            if ll_keywords:
                # Join low-level keywords for entity search
                search_query = ", ".join(ll_keywords)
                logger.debug(f"Local search using low-level keywords: '{search_query}'")

        # Step 1: Entity embedding 粗排 (using low-level keywords)
        entities_with_degree, graph_score = self._search_entities(search_query, top_k_entities)
        if not entities_with_degree:
            return []

        # Step 2: Entity degree 细排 (sort by degree descending)
        entities_with_degree.sort(key=lambda x: x["degree"], reverse=True)

        # Step 3: BFS expand edges from entities (no limit)
        relationships, edge_data_list = self._find_related_edges_from_entities(
            entities_with_degree, top_k=top_k_relationships
        )

        # Step 4: Collect all nodes
        nodes = self._get_all_nodes_from_entities_and_edges(entities_with_degree, edge_data_list)
        all_entity_ids = [n.id for n in nodes]

        # Step 5: Collect chunks from entities
        recall_chunks = self._get_chunks_by_entity_ids(all_entity_ids, max_chunks=max_chunks)

        # Step 6: Rerank chunks
        if self.reranker is not None and recall_chunks:
            rerank_k = min(final_top_k, len(recall_chunks))
            reranked_chunks = self.reranker.rerank(query, recall_chunks, top_k=rerank_k)
        else:
            reranked_chunks = recall_chunks[:final_top_k]

        # Build results
        results = []
        for i, chunk in enumerate(reranked_chunks):
            chunk.metadata["retrieval_type"] = "local_chunk"
            chunk.metadata["rank"] = i
            chunk.metadata["reranked"] = self.reranker is not None
            results.append(chunk)
        self._attach_chunk_scores(query, reranked_chunks)

        # Build graph context
        context_lines = [f"Entity: {n.id} (Type: {n.type})" for n in nodes]
        context_lines.extend(f"  {rel.source.id} --{rel.type}--> {rel.target.id}" for rel in relationships)

        context_doc = Document(
            page_content="\n".join(context_lines),
            metadata={
                "source": "local_graph",
                "score": graph_score,
                "retrieval_type": "local_graph",
                "entities": all_entity_ids,
                "relationships": len(relationships)
            }
        )
        results.append(context_doc)

        return results

    def global_search(
        self,
        query: str,
        top_k_relationships: int = 5,
        max_chunks: int = 40,
        top_k: Optional[int] = 5,
        use_keyword_extraction: bool = True
    ) -> List[Document]:
        """Global search (LightRAG style):
        1. Extract high-level keywords from query
        2. Relations: embedding 粗排 (top_k_relationships) using high-level keywords → keep cosine order
        3. Entities: extract from relations → no sorting/limit
        4. Chunks: collect from relations → reranker 细排

        Args:
            query: Search query
            top_k_relationships: Number of relationships for embedding 粗排
            max_chunks: Max chunks to collect before reranking (default: 40)
            top_k: Final chunks to return after reranking (default: 5)
            use_keyword_extraction: Whether to use LLM-based keyword extraction (default: True)

        Returns:
            List of Document (reranked chunks + graph context)
        """
        final_top_k = top_k if top_k is not None else 5

        # Step 0: Extract high-level keywords for relationship search (LightRAG style)
        search_query = query
        if use_keyword_extraction:
            hl_keywords, ll_keywords = self._extract_keywords(query)
            if hl_keywords:
                # Join high-level keywords for relationship search
                search_query = ", ".join(hl_keywords)
                logger.debug(f"Global search using high-level keywords: '{search_query}'")

        # Step 1: Relationship embedding 粗排 (using high-level keywords, keep cosine order)
        relationships_with_weight, graph_score = self._search_relationships(search_query, top_k_relationships)
        if not relationships_with_weight:
            return []

        # Step 2: Extract entities from relationships (no sorting/limit)
        nodes, relationships, relation_data_list = self._get_entities_from_relationships(relationships_with_weight)
        if not nodes:
            return []

        # Step 3: Collect chunks from relationships
        recall_chunks = self._get_chunks_by_relationships(relation_data_list, max_chunks=max_chunks)

        # Step 4: Rerank chunks
        if self.reranker is not None and recall_chunks:
            rerank_k = min(final_top_k, len(recall_chunks))
            reranked_chunks = self.reranker.rerank(query, recall_chunks, top_k=rerank_k)
        else:
            reranked_chunks = recall_chunks[:final_top_k]

        # Build results
        results = []
        for i, chunk in enumerate(reranked_chunks):
            chunk.metadata["retrieval_type"] = "global_chunk"
            chunk.metadata["rank"] = i
            chunk.metadata["reranked"] = self.reranker is not None
            results.append(chunk)
        self._attach_chunk_scores(query, reranked_chunks)

        # Build graph context
        context_lines = [f"Entity: {n.id} (Type: {n.type})" for n in nodes]
        context_lines.extend(f"  {rel.source.id} --{rel.type}--> {rel.target.id}" for rel in relationships)

        context_doc = Document(
            page_content="\n".join(context_lines),
            metadata={
                "source": "global_graph",
                "score": graph_score,
                "retrieval_type": "global_graph",
                "entities": [n.id for n in nodes],
                "relationships": len(relationships)
            }
        )
        results.append(context_doc)

        return results


def get_graphrag_retriever(
    graph,
    entity_index: Optional[faiss.Index] = None,
    entity_metadata: Optional[List[dict]] = None,
    relationship_index: Optional[faiss.Index] = None,
    relationship_metadata: Optional[List[dict]] = None,
    embedding=None,
    vectorstore=None,
    recall_top_k: int = 20,
    rerank_top_k: int = 5,
    reranker=None,
    llm=None,
) -> GraphRAGRetriever:
    """Get a GraphRAG retriever instance (recall 20, rerank 5, entities 10, relationships 5)

    Args:
        graph: NetworkX graph
        entity_index: FAISS index for entities
        entity_metadata: List of entity metadata
        relationship_index: FAISS index for relationships
        relationship_metadata: List of relationship metadata
        embedding: Embedding function
        vectorstore: Vector store for chunks
        recall_top_k: Number of chunks to recall before reranking
        rerank_top_k: Number of chunks to return after reranking
        reranker: Reranker instance
        llm: LLM for keyword extraction (optional, will be auto-initialized if not provided)
    """
    return GraphRAGRetriever(
        graph=graph,
        entity_index=entity_index,
        entity_metadata=entity_metadata,
        relationship_index=relationship_index,
        relationship_metadata=relationship_metadata,
        embedding=embedding,
        vectorstore=vectorstore,
        recall_top_k=recall_top_k,
        rerank_top_k=rerank_top_k,
        reranker=reranker,
        llm=llm,
    )
