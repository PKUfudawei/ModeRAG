"""Executor Agent - Task Execution"""
import hashlib
import contextvars
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .models.task import Task, TaskStatus, TaskType
from .models.plan import Plan
from .models.evidence import Evidence, EvidenceSource, EvidenceChain, generate_evidence_id
from .tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# --- Evidence quality controls (env-tunable) --------------------------------
# Graph (local/global) retrieval often returns "Entity: X (Type: Y)" listings
# instead of quotable passages. Those listings are not usable evidence, so:
#   * FUSION_RETRIEVAL  -> always add a vector pass for a graph task (floor = naive)
#   * EVIDENCE_FALLBACK -> add a vector pass when graph evidence is too thin
# Both are disabled for pure-mode ablations (--tool-config local/global) so the
# ablation keeps its "single retriever only" semantics.
FUSION_RETRIEVAL = os.getenv("FUSION_RETRIEVAL", "1") == "1"
EVIDENCE_FALLBACK = os.getenv("EVIDENCE_FALLBACK", "1") == "1"
FALLBACK_MIN_TEXT_EV = int(os.getenv("FALLBACK_MIN_TEXT_EV", "3"))
MAX_MERGED_EVIDENCE = int(os.getenv("MAX_MERGED_EVIDENCE", "10"))


def _is_entity_dump(content: str) -> bool:
    """True for graph neighborhood listings ('Entity: X (Type: Y)...') which carry
    entity names but no quotable text."""
    return (content or "").lstrip().startswith("Entity:")


def _merge_evidence(primary: List[Dict[str, Any]], extra: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge two retrieval result lists: dedupe, prefer real passages over entity
    listings, then highest score first.

    Scores are per-chunk query--chunk cosine similarities assigned by the
    retriever (``GraphRAGRetriever._attach_chunk_scores``). Items whose mode did
    not provide a score fall back to 0.0 and therefore rank last within their
    tier, which keeps un-scored material from displacing scored passages.
    """
    seen = set()
    merged: List[Dict[str, Any]] = []
    for item in list(primary) + list(extra):
        key = hashlib.md5((item.get("content") or "").encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    merged.sort(key=lambda x: (_is_entity_dump(x.get("content", "")), -(x.get("score") or 0.0)))
    return merged[:MAX_MERGED_EVIDENCE]


@dataclass
class TaskExecutionResult:
    """Task execution result"""
    task: Task
    success: bool
    evidence: List[Evidence]
    answer: str
    error: Optional[str] = None
    retry_count: int = 0


class ExecutorResult:
    """Executor execution result"""
    def __init__(
        self,
        evidence_chain: EvidenceChain,
        task_results: List[TaskExecutionResult],
        num_subqueries: int = 0,
        max_parallelism: int = 0,
        num_batches: int = 0
    ):
        self.evidence_chain = evidence_chain
        self.task_results = task_results
        self.num_subqueries = num_subqueries
        self.max_parallelism = max_parallelism
        self.num_batches = num_batches
        self.success = all(r.success for r in task_results)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "evidence_chain": self.evidence_chain.to_dict(),
            "task_results": [
                {
                    "task_id": r.task.task_id,
                    "success": r.success,
                    "evidence_count": len(r.evidence),
                    "error": r.error
                }
                for r in self.task_results
            ],
            "statistics": {
                "num_subqueries": self.num_subqueries,
                "max_parallelism": self.max_parallelism,
                "num_batches": self.num_batches
            }
        }


class Executor:
    """Task executor"""

    def __init__(
        self,
        max_parallel_workers: int = 4,
        max_retries: int = 1
    ):
        """
        Initialize Executor

        Args:
            max_parallel_workers: Maximum number of parallel worker threads
            max_retries: Maximum retry attempts
        """
        self.max_parallel_workers = max_parallel_workers
        self.max_retries = max_retries
        self._merge_lock = threading.Lock()  # Protect evidence chain merge operations

    def _execute_single_task(self, task: Task, evidence_chain: EvidenceChain) -> TaskExecutionResult:
        """
        Execute a single task (with retries)

        Args:
            task: Task
            evidence_chain: Evidence chain

        Returns:
            TaskExecutionResult object
        """
        task.status = TaskStatus.RUNNING
        last_error: Optional[str] = None

        # Retry loop
        for retry in range(self.max_retries + 1):
            try:
                if retry > 0:
                    logger.info(f"Task {task.task_id} retry {retry}/{self.max_retries}")

                result = self._execute_task_with_tool(task, evidence_chain)
                if result.success:
                    result.retry_count = retry
                    return result
                last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Task {task.task_id} execution exception (retry {retry}): {e}")

        # All retries exhausted
        logger.error(f"Task {task.task_id} final failure: {last_error}")
        return TaskExecutionResult(
            task=task,
            success=False,
            evidence=[],
            answer="",
            error=last_error,
            retry_count=self.max_retries
        )

    def _execute_task_with_tool(self, task: Task, evidence_chain: EvidenceChain) -> TaskExecutionResult:
        """Execute task with specified tool"""
        # Get tool (using ToolRegistry.create_tool method)
        tool = ToolRegistry.create_tool(task.task_type.value)
        if not tool:
            raise ValueError(f"Tool not found: {task.task_type.value}")

        # Execute search
        payload = {"query": task.query, **task.parameters}
        result = tool.structured_search(payload)

        if not result.get("success"):
            raise ValueError(result.get("error", "Tool execution failed"))

        retrieval_results = list(result.get("retrieval_results", []))

        # Evidence quality control: graph modes (local/global) frequently return
        # entity listings rather than passages. Guarantee a vector-retrieval floor
        # so a graph miss degrades to naive behaviour instead of an empty/"insufficient
        # context" answer.
        mode = task.task_type.value
        if mode != "naive" and (FUSION_RETRIEVAL or EVIDENCE_FALLBACK):
            text_evidence = [x for x in retrieval_results if not _is_entity_dump(x.get("content", ""))]
            need_vector = FUSION_RETRIEVAL or len(text_evidence) < FALLBACK_MIN_TEXT_EV
            if need_vector:
                naive_tool = ToolRegistry.create_tool("naive")
                if naive_tool is not None:
                    try:
                        fb = naive_tool.structured_search({"query": task.query})
                        if fb.get("success"):
                            retrieval_results = _merge_evidence(
                                retrieval_results, fb.get("retrieval_results", [])
                            )
                            logger.info(
                                "task %s (%s): merged vector evidence -> %d items (%d real text)",
                                task.task_id, mode, len(retrieval_results),
                                sum(1 for x in retrieval_results if not _is_entity_dump(x.get("content", ""))),
                            )
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("vector fallback for task %s failed: %s", task.task_id, exc)

        # Convert to Evidence
        evidence_list = []
        for item in retrieval_results:
            evidence = Evidence(
                evidence_id=generate_evidence_id(),
                source=EvidenceSource(task.task_type.value),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                task_id=task.task_id,
                metadata=item.get("metadata", {})
            )
            evidence_list.append(evidence)
            evidence_chain.add_evidence(evidence)

        # Add entities and edges to evidence chain (if available)
        for entity_id in result.get("entities", []):
            evidence_chain.add_entity({"id": entity_id, "task_id": task.task_id})
        for edge in result.get("edges", []):
            evidence_chain.add_edge({**edge, "task_id": task.task_id})

        # Add reasoning step
        evidence_chain.add_reasoning_step(
            f"Execute task {task.task_id} ({task.task_type.value}): {task.query}"
        )

        return TaskExecutionResult(
            task=task,
            success=True,
            evidence=evidence_list,
            answer=result.get("answer", "")
        )

    def execute_parallel(self, plan: Plan) -> ExecutorResult:
        """
        Execute plan in parallel (dependency-aware parallelism)

        Thread-safe implementation:
        - Each task uses independent EvidenceChain to avoid race conditions
        - Use lock to protect main evidence chain merge operations

        Args:
            plan: Task plan

        Returns:
            ExecutorResult object
        """
        # Main evidence chain (for final merge)
        main_evidence_chain = EvidenceChain(
            chain_id=plan.plan_id.replace("plan_", "chain_"),
            query=plan.original_query
        )

        task_results: List[TaskExecutionResult] = []
        task_map = {t.task_id: t for t in plan.tasks}
        completed_tasks: Dict[str, TaskExecutionResult] = {}

        # Execute in batches (each batch executes all tasks with satisfied dependencies)
        pending_tasks = set(t.task_id for t in plan.tasks)
        num_batches = 0
        max_parallelism = 0

        while pending_tasks:
            num_batches += 1
            # Find tasks that can be executed
            ready_tasks = []
            for task_id in list(pending_tasks):
                task = task_map[task_id]
                deps_satisfied = all(
                    dep_id in completed_tasks and completed_tasks[dep_id].success
                    for dep_id in task.depends_on
                )
                if deps_satisfied:
                    ready_tasks.append(task)

            if not ready_tasks:
                # No executable tasks, possibly circular dependency or failed prerequisites
                logger.warning("No executable tasks, skipping remaining tasks")
                break

            # Track max parallelism
            max_parallelism = max(max_parallelism, len(ready_tasks))

            # Execute ready tasks in parallel (thread-safe version)
            with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
                # Each task uses independent EvidenceChain
                # Copy the calling context into each worker so that the LLM
                # usage scope (models/llm.usage_scope) is visible to the tool
                # calls made inside the batch. Without this, per-query
                # statistics omitted every keyword-extraction call.
                futures = {
                    executor.submit(
                        contextvars.copy_context().run,
                        self._execute_single_task,
                        task,
                        EvidenceChain.empty(
                            chain_id=f"chain_{task.task_id}",
                            query=task.query
                        )
                    ): task
                    for task in ready_tasks
                }

                for future in as_completed(futures):
                    result = future.result()
                    task_results.append(result)
                    completed_tasks[result.task.task_id] = result
                    pending_tasks.discard(result.task.task_id)

                    # Thread-safe evidence chain merge
                    with self._merge_lock:
                        # Extract evidence from result and merge to main evidence chain
                        for evidence in result.evidence:
                            main_evidence_chain.add_evidence(evidence)
                        main_evidence_chain.add_reasoning_step(
                            f"Execute task {result.task.task_id} ({result.task.task_type.value}): {result.task.query}"
                        )

                    # Update task status
                    task_map[result.task.task_id].status = (
                        TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                    )

        return ExecutorResult(
            evidence_chain=main_evidence_chain,
            task_results=task_results,
            num_subqueries=len(plan.tasks),
            max_parallelism=max_parallelism,
            num_batches=num_batches
        )


def get_executor(max_parallel_workers: int = 4) -> Executor:
    """Get Executor instance"""
    return Executor(max_parallel_workers=max_parallel_workers)
