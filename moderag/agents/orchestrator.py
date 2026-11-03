"""Orchestrator - Multi-Agent Orchestrator

Unified entry: All queries follow Planner → Executor → Reporter flow
Ensures complete evidence chain tracking and explainable reasoning for each query.
"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .planner import Planner, PlannerResult
from .tools.registry import ToolRegistry
from .executor import Executor, ExecutorResult
from .reporter import Reporter, ReporterResult, ReportMode
from .models.plan import Plan
from .models.report import Report

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Orchestration result"""
    success: bool
    query: str
    answer: str
    report: Optional[Report] = None
    plan: Optional[Plan] = None
    error: Optional[str] = None
    # Statistics
    planner_tokens: int = 0
    reporter_tokens: int = 0
    num_subqueries: int = 0
    max_parallelism: int = 0
    num_batches: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "query": self.query,
            "answer": self.answer,
            "report": self.report.to_dict() if self.report else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "error": self.error,
            "statistics": {
                "planner_tokens": self.planner_tokens,
                "reporter_tokens": self.reporter_tokens,
                "num_subqueries": self.num_subqueries,
                "max_parallelism": self.max_parallelism,
                "num_batches": self.num_batches
            }
        }


class Orchestrator:
    """Multi-agent orchestrator

    All queries follow Planner → Executor → Reporter flow,
    ensuring complete evidence chain tracking and explainable reasoning.
    """

    def __init__(
        self,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        reporter: Optional[Reporter] = None,
        report_mode: ReportMode = ReportMode.CONCISE,
        rag_storage_path: str = "./storage/rag_index",
        graphrag_storage_path: str = "./storage/graphrag_index"
    ):
        """
        Initialize orchestrator

        Args:
            planner: Task planner
            executor: Task executor
            reporter: Report generator
            report_mode: Report generation mode (concise or detailed)
            rag_storage_path: RAG index storage path
            graphrag_storage_path: GraphRAG index storage path
        """
        # Get available tools from registry
        available_tools = ToolRegistry.list_tools()
        self.planner = planner or Planner(available_tools=available_tools)
        self.executor = executor or Executor()
        self.reporter = reporter or Reporter(mode=report_mode)
        self.report_mode = report_mode
        self.rag_storage_path = rag_storage_path
        self.graphrag_storage_path = graphrag_storage_path

    def process_query(self, query: str) -> OrchestratorResult:
        """
        Process user query (follow Multi-Agent flow)

        Flow:
        1. Planner: Task decomposition (simple queries return single task, complex queries use LLM)
        2. Executor: Parallel task execution, calling RAG/GraphRAG tools
        3. Reporter: Integrate results, generate final answer with evidence chain

        Args:
            query: User query

        Returns:
            OrchestratorResult object
        """
        logger.info(f"Processing query: {query}")

        try:
            # Step 1: Planning
            logger.info("Step 1: Task planning...")
            planner_result = self.planner.plan(query)

            if not planner_result.success:
                raise ValueError(f"Planning failed: {planner_result.error}")

            plan = planner_result.plan
            planner_tokens = planner_result.tokens_used
            logger.info(f"Planning completed, {len(plan.tasks)} tasks")

            # Step 2: Execution
            logger.info("Step 2: Task execution...")
            executor_result = self.executor.execute_parallel(plan)
            logger.info(f"Execution completed, success: {executor_result.success}")

            # Step 3: Reporting
            logger.info("Step 3: Report generation...")
            reporter_result = self.reporter.generate(executor_result, plan)

            if not reporter_result.success:
                raise ValueError(f"Report generation failed: {reporter_result.error}")

            report = reporter_result.report

            return OrchestratorResult(
                success=True,
                query=query,
                answer=report.final_answer,
                report=report,
                plan=plan,
                planner_tokens=planner_tokens,
                reporter_tokens=reporter_result.tokens_used,
                num_subqueries=executor_result.num_subqueries,
                max_parallelism=executor_result.max_parallelism,
                num_batches=executor_result.num_batches
            )

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return OrchestratorResult(
                success=False,
                query=query,
                answer="",
                error=str(e)
            )

    def set_report_mode(self, mode: ReportMode) -> None:
        """
        Set report generation mode

        Args:
            mode: Report mode (concise or detailed)
        """
        self.report_mode = mode
        self.reporter.mode = mode


def get_orchestrator(
    rag_storage_path: str = "./storage/rag_index",
    graphrag_storage_path: str = "./storage/graphrag_index",
    report_mode: ReportMode = ReportMode.CONCISE
) -> Orchestrator:
    """Get orchestrator instance"""
    return Orchestrator(
        rag_storage_path=rag_storage_path,
        graphrag_storage_path=graphrag_storage_path,
        report_mode=report_mode
    )
