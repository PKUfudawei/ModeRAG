"""Reporter Agent - Report Generation

Supports two modes:
1. Concise mode: Direct answer generation using retrieved context (like traditional RAG/GraphRAG)
2. Reasoning mode: Evidence chain tracking with step-by-step reasoning attribution
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from enum import Enum

from .models.plan import Plan
from .models.evidence import Evidence, EvidenceChain, EvidenceSource
from .models.report import (
    Report, ReportSection, ReasoningStep, EvidenceTrace,
    generate_report_id, generate_reasoning_step_id
)
from .executor import ExecutorResult, TaskExecutionResult

logger = logging.getLogger(__name__)


class ReportMode(str, Enum):
    """Report generation mode"""
    CONCISE = "concise"
    REASONING = "reasoning"


# System prompts
REPORTER_SYSTEM_PROMPT = """You are a report generation expert. Your task is to generate structured answers based on retrieved evidence."""

REASONING_SYSTEM_PROMPT = """You are a reasoning and evidence analysis expert. Your task is to:
1. Analyze retrieved evidence from multiple sources
2. Build a chain of reasoning with clear attribution to evidence
3. Identify which evidence supports which claims
4. Flag any conclusions that lack sufficient evidence support"""


# Concise mode prompt (LightRAG style) - direct answer generation
REPORTER_CONCISE_PROMPT = """---Role---

You are an expert AI assistant specializing in synthesizing information from retrieved scientific documents and knowledge graphs. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a concise, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph data and Document Chunks found in the **Context**.

---Instructions---

1. Step-by-Step Instruction:
   - Carefully determine the user's query intent to fully understand the information need.
   - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**.
   - Identify and extract all pieces of information that are directly relevant to answering the query.
   - Look for conclusions, summaries, and key findings - they often appear in section endings or "Conclusions" sections.
   - Weave the extracted facts into a coherent and logical response.
   - Your own knowledge must ONLY be used to formulate fluent sentences, NOT to introduce external information.

2. Content & Grounding:
   - Strictly adhere to the provided **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
   - **CRITICAL - Numeric Values**: When the query asks for specific numeric values (e.g., "lowest transverse momentum", "dimension", "range"), ONLY use values EXPLICITLY stated in the retrieved context.
   - **CRITICAL - Source Attribution**: For each numeric value, verify it comes from context relevant to the query entity. If a value appears in one context but the query is about a different entity, do NOT use that value.
   - **CRITICAL - Context Relevance**: Vector Search Results are typically more reliable for specific numeric queries. Graph-Based Results may contain information from multiple documents - verify each value is relevant.
   - Technical content, formulas, and equations ARE valid evidence - use them to answer.
   - **CRITICAL - Partial answers**: the context is often long and densely formatted; scan ALL of it before concluding anything is missing. If the query has multiple parts, answer EVERY part the context supports and explicitly mark only the unsupported parts as "not stated in the provided context". Never refuse the whole query when part of it is answerable. Only if, after a full review, nothing relevant exists in the context, state that the information is not reported in the provided context.

3. Formatting & Language:
   - The response MUST be in the same language as the user query.
   - The response MUST utilize Markdown formatting for clarity (e.g., bold text, bullet points).
   - Keep the answer concise but complete (3-6 sentences).
   - Do not include citations or references in the output.

---Context---

{evidence_context}

---Query---

{query}

---Answer---
"""


# Reasoning mode prompts
REASONING_ANALYSIS_PROMPT = """You will analyze evidence and build a reasoning chain.

Query: {query}

Retrieved Evidence:
{evidence_context}

Task Execution Results:
{task_results}

Your task is to:
1. Identify key claims that answer the query
2. For each claim, identify which evidence supports it
3. Assess confidence level for each claim
4. Flag claims with insufficient evidence

Output format (JSON):
{{
    "reasoning_steps": [
        {{
            "claim": "A specific claim/conclusion",
            "supporting_evidence_ids": ["evidence_id_1", "evidence_id_2"],
            "confidence": 0.8-1.0,
            "uncertainty_note": "only if confidence < 0.8"
        }}
    ],
    "final_answer": "Synthesized answer with inline citations like [E1], [E2]"
}}
"""


class ReporterResult:
    """Reporter execution result"""
    def __init__(
        self,
        report: Optional[Report] = None,
        error: Optional[str] = None,
        tokens_used: int = 0,
        reasoning_chain: Optional[List[ReasoningStep]] = None,
        evidence_traces: Optional[List[EvidenceTrace]] = None
    ):
        self.report = report
        self.error = error
        self.tokens_used = tokens_used
        self.reasoning_chain = reasoning_chain or []
        self.evidence_traces = evidence_traces or []
        self.success = report is not None and error is None

    def to_dict(self) -> dict:
        result = {
            "success": self.success,
            "report": self.report.to_dict() if self.report else None,
            "error": self.error,
            "tokens_used": self.tokens_used
        }

        if self.reasoning_chain:
            result["reasoning_chain"] = [s.to_dict() for s in self.reasoning_chain]
        if self.evidence_traces:
            result["evidence_traces"] = [t.to_dict() for t in self.evidence_traces]

        return result


class Reporter:
    """Report generator with dual modes: concise and reasoning"""

    def __init__(self, llm=None, mode: ReportMode = ReportMode.CONCISE):
        """
        Initialize Reporter

        Args:
            llm: LLM instance, uses default if None
            mode: Report generation mode (concise or reasoning)
        """
        self.llm = llm
        self.mode = mode
        self._llm_initialized = False

    def _initialize_llm(self):
        """Lazy initialize LLM"""
        if self._llm_initialized:
            return

        if self.llm is None:
            from moderag.models.llm import get_llm
            self.llm = get_llm(temperature=0)

        self._llm_initialized = True

    def generate(
        self,
        executor_result: ExecutorResult,
        plan: Optional[Plan] = None
    ) -> ReporterResult:
        """
        Generate report based on mode

        Args:
            executor_result: Executor execution result
            plan: Task plan (optional)

        Returns:
            ReporterResult object
        """
        try:
            self._initialize_llm()

            evidence_chain = executor_result.evidence_chain
            query = evidence_chain.query

            # Generate based on mode
            if self.mode == ReportMode.REASONING:
                return self._generate_reasoning_report(query, executor_result, plan)
            else:
                return self._generate_concise_report(query, evidence_chain, plan)

        except Exception as e:
            logger.error(f"Reporter execution failed: {e}")
            return ReporterResult(error=str(e))

    def _generate_concise_report(
        self,
        query: str,
        evidence_chain: EvidenceChain,
        plan: Optional[Plan]
    ) -> ReporterResult:
        """
        Generate concise report (traditional RAG style)

        Args:
            query: User query
            evidence_chain: Evidence chain from executor
            plan: Task plan (optional)

        Returns:
            ReporterResult with concise answer
        """
        # Build evidence context (use more context for better answers)
        evidence_context = self._build_evidence_context(evidence_chain, max_chars=2000)

        # Generate answer
        prompt = REPORTER_CONCISE_PROMPT.format(
            query=query,
            evidence_context=evidence_context
        )
        messages = [
            SystemMessage(content=REPORTER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        tokens_used = self._extract_token_usage(response)

        # Build report
        report = Report(
            report_id=generate_report_id(),
            query=query,
            final_answer=answer.strip(),
            evidence_chain=evidence_chain,
            plan_id=plan.plan_id if plan else None,
            report_mode="concise"
        )
        report.build_structured_evidence()

        return ReporterResult(report=report, tokens_used=tokens_used)

    def _generate_reasoning_report(
        self,
        query: str,
        executor_result: ExecutorResult,
        plan: Optional[Plan]
    ) -> ReporterResult:
        """
        Generate reasoning report with evidence chain tracking

        Args:
            query: User query
            executor_result: Full executor result with task results
            plan: Task plan (optional)

        Returns:
            ReporterResult with reasoning chain and evidence traces
        """
        evidence_chain = executor_result.evidence_chain
        task_results = executor_result.task_results

        # Build structured evidence context with IDs
        evidence_context = self._build_evidence_context_with_ids(evidence_chain)

        # Build task results summary
        task_results_summary = self._build_task_results_summary(task_results)

        # Generate reasoning chain using LLM
        prompt = REASONING_ANALYSIS_PROMPT.format(
            query=query,
            evidence_context=evidence_context,
            task_results=task_results_summary
        )
        messages = [
            SystemMessage(content=REASONING_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        response_content = response.content if hasattr(response, 'content') else str(response)
        tokens_used = self._extract_token_usage(response)

        # Parse reasoning output
        reasoning_data = self._parse_reasoning_response(response_content)

        # Build reasoning steps
        reasoning_steps: List[ReasoningStep] = []
        for step_data in reasoning_data.get("reasoning_steps", []):
            step = ReasoningStep(
                step_id=generate_reasoning_step_id(),
                claim=step_data.get("claim", ""),
                supporting_evidence=step_data.get("supporting_evidence_ids", []),
                confidence=step_data.get("confidence", 0.8),
                uncertainty_note=step_data.get("uncertainty_note")
            )
            reasoning_steps.append(step)

            # Link to task if possible
            self._link_step_to_task(step, task_results)

        # Build evidence traces
        evidence_traces = self._build_evidence_traces(evidence_chain, reasoning_steps)

        # Build final answer with citations
        final_answer = reasoning_data.get("final_answer", "")
        if not final_answer:
            final_answer = self._synthesize_answer_from_steps(reasoning_steps, evidence_chain)

        # Build report
        report = Report(
            report_id=generate_report_id(),
            query=query,
            final_answer=final_answer.strip(),
            evidence_chain=evidence_chain,
            plan_id=plan.plan_id if plan else None,
            report_mode="reasoning"
        )

        # Add reasoning chain and evidence traces
        for step in reasoning_steps:
            report.add_reasoning_step(step)
        for trace in evidence_traces:
            report.add_evidence_trace(trace)

        # Build sections organized by conclusion
        report.sections = self._build_sections_by_conclusion(reasoning_steps, evidence_chain)

        report.build_structured_evidence()

        return ReporterResult(
            report=report,
            tokens_used=tokens_used,
            reasoning_chain=reasoning_steps,
            evidence_traces=evidence_traces
        )

    def _link_step_to_task(self, step: ReasoningStep, task_results: List[TaskExecutionResult]) -> None:
        """Link reasoning step to task that produced supporting evidence"""
        if not step.supporting_evidence:
            return

        # Find task that produced the first supporting evidence
        for result in task_results:
            task_evidence_ids = {e.evidence_id for e in result.evidence}
            if task_evidence_ids & set(step.supporting_evidence):
                step.task_id = result.task.task_id
                break

    def _build_evidence_traces(
        self,
        evidence_chain: EvidenceChain,
        reasoning_steps: List[ReasoningStep]
    ) -> List[EvidenceTrace]:
        """Build evidence traces showing which evidence supports which steps"""
        traces = []

        for evidence in evidence_chain.evidence_list:
            # Find which reasoning steps this evidence supports
            contributes_to = [
                step.step_id for step in reasoning_steps
                if evidence.evidence_id in step.supporting_evidence
            ]

            trace = EvidenceTrace(
                evidence_id=evidence.evidence_id,
                source=evidence.source.value,
                contributes_to=contributes_to,
                relevance_score=evidence.score
            )
            traces.append(trace)

        return traces

    def _build_sections_by_conclusion(
        self,
        reasoning_steps: List[ReasoningStep],
        evidence_chain: EvidenceChain
    ) -> List[ReportSection]:
        """Build report sections organized by conclusion"""
        sections = []

        # Group steps by confidence level
        high_confidence = [s for s in reasoning_steps if s.confidence >= 0.8]
        low_confidence = [s for s in reasoning_steps if s.confidence < 0.8]

        # Main conclusions section
        if high_confidence:
            content = "\n\n".join([s.claim for s in high_confidence])
            evidence_ids = []
            for step in high_confidence:
                evidence_ids.extend(step.supporting_evidence)

            section = ReportSection(
                section_id="main_conclusions",
                title="Main Conclusions",
                content=content,
                evidence_ids=list(set(evidence_ids)),
                reasoning_steps=high_confidence
            )
            sections.append(section)

        # Uncertain conclusions section
        if low_confidence:
            content = ""
            for step in low_confidence:
                uncertainty = step.uncertainty_note or "Limited evidence available"
                content += f"- {step.claim}\n  Note: {uncertainty}\n"

            evidence_ids = []
            for step in low_confidence:
                evidence_ids.extend(step.supporting_evidence)

            section = ReportSection(
                section_id="uncertain_conclusions",
                title="Conclusions with Limited Evidence",
                content=content.strip(),
                evidence_ids=list(set(evidence_ids)),
                reasoning_steps=low_confidence
            )
            sections.append(section)

        return sections

    def _synthesize_answer_from_steps(
        self,
        reasoning_steps: List[ReasoningStep],
        evidence_chain: EvidenceChain
    ) -> str:
        """Synthesize final answer from reasoning steps when LLM fails to provide one"""
        parts = []

        for step in reasoning_steps:
            if step.confidence >= 0.8:
                parts.append(step.claim)
            else:
                note = step.uncertainty_note or "limited evidence"
                parts.append(f"{step.claim} (uncertainty: {note})")

        return "\n\n".join(parts) if parts else "Insufficient evidence to answer the query."

    def _build_evidence_context(self, evidence_chain: EvidenceChain, max_chars: int = 500) -> str:
        """
        Build evidence context text (for concise mode) - LightRAG style format

        Format follows LightRAG's kg_query_context:
        - Knowledge Graph Data (Entity) in JSON
        - Knowledge Graph Data (Relationship) in JSON
        - Document Chunks in JSON
        - Reference Document List

        Args:
            evidence_chain: Evidence chain
            max_chars: Maximum characters per evidence chunk

        Returns:
            Evidence context text in LightRAG style format
        """
        from .models.evidence import EvidenceSource
        import json

        lines = []

        # Group by source type
        naive_evidence = evidence_chain.get_evidence_by_source(EvidenceSource.NAIVE)
        local_evidence = evidence_chain.get_evidence_by_source(EvidenceSource.LOCAL)
        global_evidence = evidence_chain.get_evidence_by_source(EvidenceSource.GLOBAL)

        # === Knowledge Graph Data (Entity) ===
        if evidence_chain.entities:
            lines.append("Knowledge Graph Data (Entity):")
            lines.append("")
            lines.append("```json")
            # Filter out graph context docs and build entity list
            entity_list = []
            for entity in evidence_chain.entities:
                if isinstance(entity, dict):
                    entity_list.append({
                        "entity": entity.get("id", str(entity)),
                        "type": entity.get("type", "Entity")
                    })
                else:
                    entity_list.append({"entity": str(entity), "type": "Entity"})
            lines.append(json.dumps(entity_list, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

        # === Knowledge Graph Data (Relationship) ===
        if evidence_chain.edges:
            lines.append("Knowledge Graph Data (Relationship):")
            lines.append("")
            lines.append("```json")
            rel_list = []
            for edge in evidence_chain.edges:
                rel_list.append({
                    "entity1": edge.get("src_id", "unknown"),
                    "entity2": edge.get("tgt_id", "unknown"),
                    "relationship_type": edge.get("rel_type", "RELATED_TO")
                })
            lines.append(json.dumps(rel_list, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

        # === Document Chunks ===
        all_chunks = []
        reference_list = []
        ref_id = 1

        # Process all evidence (skip graph context docs)
        for ev in naive_evidence + local_evidence + global_evidence:
            if ev.metadata.get("retrieval_type") in ["local_graph", "global_graph"]:
                continue

            source_label = ev.source.value
            content_preview = ev.content[:max_chars]
            if len(ev.content) > max_chars:
                content_preview += "..."

            all_chunks.append({
                "reference_id": ref_id,
                "source": source_label,
                "score": round(ev.score, 3),
                "content": content_preview
            })

            if ev.metadata.get("source"):
                reference_list.append(f"[{ref_id}] {ev.metadata.get('source')}")

            ref_id += 1

        if all_chunks:
            lines.append("Document Chunks (Each entry has a reference_id referring to the Reference Document List):")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(all_chunks, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

        # === Reference Document List ===
        if reference_list:
            lines.append("Reference Document List:")
            lines.append("")
            lines.append("```")
            lines.append("\n".join(reference_list))
            lines.append("```")
            lines.append("")

        return "\n".join(lines) if lines else "No relevant evidence found"

    def _build_evidence_context_with_ids(self, evidence_chain: EvidenceChain, max_chars: int = 800) -> str:
        """
        Build evidence context with unique IDs for citation (for reasoning mode)

        Args:
            evidence_chain: Evidence chain
            max_chars: Maximum characters per evidence chunk

        Returns:
            Evidence context text with [E1], [E2] style IDs
        """
        lines = []
        evidence_index = 1

        # Process all evidence in score order
        all_evidence = sorted(evidence_chain.evidence_list, key=lambda x: x.score, reverse=True)

        for evidence in all_evidence:
            evidence_id = f"E{evidence_index}"
            source_label = evidence.source.value.upper()
            content_preview = evidence.content[:max_chars] + "..." if len(evidence.content) > max_chars else evidence.content

            lines.append(f"[{evidence_id}] ({source_label}, score: {evidence.score:.3f}, id: {evidence.evidence_id})")
            lines.append(f"{content_preview}")

            if evidence.metadata.get("source"):
                lines.append(f"Source: {evidence.metadata.get('source')}")
            lines.append("")

            evidence_index += 1

        return "\n".join(lines) if lines else "No relevant evidence found"

    def _build_task_results_summary(self, task_results: List[TaskExecutionResult]) -> str:
        """Build summary of task execution results"""
        lines = []
        for result in task_results:
            status = "SUCCESS" if result.success else "FAILED"
            lines.append(f"Task {result.task.task_id} ({result.task.task_type.value}): {status}")
            lines.append(f"  Query: {result.task.query}")
            if result.error:
                lines.append(f"  Error: {result.error}")
            lines.append("")
        return "\n".join(lines) if lines else "No task results"

    def _parse_reasoning_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM reasoning response"""
        import json

        content = content.strip()

        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Reasoning JSON parsing failed: {e}")
            return {"reasoning_steps": [], "final_answer": ""}

    def _extract_token_usage(self, response) -> int:
        """Extract token usage from LLM response"""
        try:
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                if usage:
                    return usage.get('total_tokens', 0)
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'total_tokens'):
                    return usage.total_tokens
        except Exception:
            pass
        return 0


def get_reporter(llm=None, mode: ReportMode = ReportMode.CONCISE) -> Reporter:
    """Get Reporter instance"""
    return Reporter(llm=llm, mode=mode)
