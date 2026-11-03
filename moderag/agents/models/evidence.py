"""Evidence-related data models"""
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from enum import Enum
import uuid
from datetime import datetime


class EvidenceSource(Enum):
    """Evidence source"""
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"


@dataclass
class Evidence:
    """Single evidence item"""
    evidence_id: str
    source: EvidenceSource
    content: str
    score: float
    task_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "evidence_id": self.evidence_id,
            "source": self.source.value,
            "content": self.content,
            "score": self.score,
            "task_id": self.task_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        """Create from dictionary"""
        return cls(
            evidence_id=data["evidence_id"],
            source=EvidenceSource(data["source"]),
            content=data["content"],
            score=data["score"],
            task_id=data["task_id"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        )


@dataclass
class EvidenceChain:
    """Evidence chain - tracks all evidence during reasoning process"""
    chain_id: str
    query: str
    evidence_list: List[Evidence] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    graph_paths: List[List[str]] = field(default_factory=list)  # Graph paths
    entities: List[Dict[str, Any]] = field(default_factory=list)  # Retrieved entities
    edges: List[Dict[str, Any]] = field(default_factory=list)  # Retrieved edges

    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence"""
        self.evidence_list.append(evidence)

    def add_reasoning_step(self, step: str) -> None:
        """Add reasoning step"""
        self.reasoning_steps.append(step)

    def add_graph_path(self, path: List[str]) -> None:
        """Add graph path"""
        self.graph_paths.append(path)

    def add_entity(self, entity: Dict[str, Any]) -> None:
        """Add entity"""
        self.entities.append(entity)

    def add_edge(self, edge: Dict[str, Any]) -> None:
        """Add edge"""
        self.edges.append(edge)

    def merge(self, other: "EvidenceChain") -> None:
        """
        Merge another EvidenceChain content into current object
        Used for thread-safe result aggregation
        """
        self.evidence_list.extend(other.evidence_list)
        self.reasoning_steps.extend(other.reasoning_steps)
        self.graph_paths.extend(other.graph_paths)
        self.entities.extend(other.entities)
        self.edges.extend(other.edges)

    @classmethod
    def empty(cls, chain_id: str, query: str) -> "EvidenceChain":
        """Create an empty EvidenceChain"""
        return cls(chain_id=chain_id, query=query)

    def get_evidence_by_task(self, task_id: str) -> List[Evidence]:
        """Get all evidence for specified task"""
        return [e for e in self.evidence_list if e.task_id == task_id]

    def get_evidence_by_source(self, source: EvidenceSource) -> List[Evidence]:
        """Get all evidence from specified source"""
        return [e for e in self.evidence_list if e.source == source]

    def get_top_evidence(self, top_k: int = 5) -> List[Evidence]:
        """Get highest-scoring evidence"""
        sorted_evidence = sorted(self.evidence_list, key=lambda x: x.score, reverse=True)
        return sorted_evidence[:top_k]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "chain_id": self.chain_id,
            "query": self.query,
            "evidence_list": [e.to_dict() for e in self.evidence_list],
            "reasoning_steps": self.reasoning_steps,
            "graph_paths": self.graph_paths,
            "entities": self.entities,
            "edges": self.edges,
            "total_evidence": len(self.evidence_list)
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceChain":
        """Create from dictionary"""
        chain = cls(
            chain_id=data["chain_id"],
            query=data["query"],
            evidence_list=[Evidence.from_dict(e) for e in data.get("evidence_list", [])],
            reasoning_steps=data.get("reasoning_steps", []),
            graph_paths=data.get("graph_paths", []),
            entities=data.get("entities", []),
            edges=data.get("edges", [])
        )
        return chain


def generate_evidence_id() -> str:
    """Generate unique evidence ID"""
    return f"evidence_{uuid.uuid4().hex[:8]}"


def generate_chain_id() ->str:
    """Generate unique evidence chain ID"""
    return f"chain_{uuid.uuid4().hex[:8]}"
