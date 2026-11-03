"""Report-related data models"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from .evidence import Evidence, EvidenceChain


@dataclass
class ReasoningStep:
    """单个推理步骤 - 用于 reasoning 模式"""
    step_id: str
    claim: str
    supporting_evidence: List[str] = field(default_factory=list)
    task_id: Optional[str] = None
    confidence: float = 1.0
    uncertainty_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "claim": self.claim,
            "supporting_evidence": self.supporting_evidence,
            "task_id": self.task_id,
            "confidence": self.confidence,
            "uncertainty_note": self.uncertainty_note
        }


@dataclass
class EvidenceTrace:
    """证据追踪 - 记录证据如何支撑推理步骤"""
    evidence_id: str
    source: str
    contributes_to: List[str] = field(default_factory=list)
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "contributes_to": self.contributes_to,
            "relevance_score": self.relevance_score
        }


@dataclass
class ReportSection:
    """Report section"""
    section_id: str
    title: str
    content: str
    evidence_ids: List[str] = field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "evidence_ids": self.evidence_ids,
            "reasoning_steps": [s.to_dict() for s in self.reasoning_steps]
        }


@dataclass
class Report:
    """Final report"""
    report_id: str
    query: str
    final_answer: str
    sections: List[ReportSection] = field(default_factory=list)
    evidence_chain: Optional[EvidenceChain] = None
    plan_id: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.now)

    # Reasoning mode specific fields
    reasoning_chain: List[ReasoningStep] = field(default_factory=list)
    evidence_traces: List[EvidenceTrace] = field(default_factory=list)
    report_mode: str = "concise"

    # Structured evidence (for frontend display)
    structured_evidence: Dict[str, Any] = field(default_factory=dict)

    def add_section(self, section: ReportSection) -> None:
        """Add section"""
        self.sections.append(section)

    def add_reasoning_step(self, step: ReasoningStep) -> None:
        """Add reasoning step"""
        self.reasoning_chain.append(step)

    def add_evidence_trace(self, trace: EvidenceTrace) -> None:
        """Add evidence trace"""
        self.evidence_traces.append(trace)

    def build_structured_evidence(self) -> None:
        """Build structured evidence"""
        if self.evidence_chain:
            base_evidence = {
                "top_evidence": [e.to_dict() for e in self.evidence_chain.get_top_evidence(10)],
                "graph_paths": self.evidence_chain.graph_paths
            }

            if self.report_mode == "reasoning":
                base_evidence["reasoning_chain"] = [s.to_dict() for s in self.reasoning_chain]
                base_evidence["evidence_traces"] = [t.to_dict() for t in self.evidence_traces]
                base_evidence["executor_reasoning_steps"] = self.evidence_chain.reasoning_steps
            else:
                base_evidence["reasoning_steps"] = self.evidence_chain.reasoning_steps

            self.structured_evidence = base_evidence

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {
            "report_id": self.report_id,
            "query": self.query,
            "final_answer": self.final_answer,
            "sections": [s.to_dict() for s in self.sections],
            "plan_id": self.plan_id,
            "generated_at": self.generated_at.isoformat(),
            "report_mode": self.report_mode,
            "structured_evidence": self.structured_evidence
        }

        if self.report_mode == "reasoning":
            result["reasoning_chain"] = [s.to_dict() for s in self.reasoning_chain]
            result["evidence_traces"] = [t.to_dict() for t in self.evidence_traces]

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Report":
        """Create from dictionary"""
        return cls(
            report_id=data["report_id"],
            query=data["query"],
            final_answer=data["final_answer"],
            sections=[ReportSection(**s) for s in data.get("sections", [])],
            plan_id=data.get("plan_id"),
            generated_at=datetime.fromisoformat(data["generated_at"]) if data.get("generated_at") else datetime.now(),
            report_mode=data.get("report_mode", "concise"),
            structured_evidence=data.get("structured_evidence", {})
        )


def generate_report_id() -> str:
    """Generate unique report ID"""
    return f"report_{uuid.uuid4().hex[:8]}"


def generate_reasoning_step_id() -> str:
    """Generate unique reasoning step ID"""
    return f"step_{uuid.uuid4().hex[:8]}"
