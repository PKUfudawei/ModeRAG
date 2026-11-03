"""
Multi-Agent Intelligent Dialogue System

Based on Plan-Execute-Report architecture, implements automatic task decomposition,
parallel retrieval, and explainable reasoning.
"""

from .planner import Planner, PlannerResult, get_planner
from .executor import Executor, ExecutorResult, TaskExecutionResult, get_executor
from .reporter import Reporter, ReporterResult, get_reporter, ReportMode
from .orchestrator import Orchestrator, OrchestratorResult, get_orchestrator
from .tools import ToolRegistry, BaseTool, ToolResult, GraphRAGTool

# Core data models
from .models import (
    Task, TaskType, TaskStatus, generate_task_id,
    Evidence, EvidenceSource, EvidenceChain, generate_evidence_id, generate_chain_id,
    Plan, PlanStatus, generate_plan_id,
    Report, ReportSection, generate_report_id
)

__version__ = "0.1.0"

__all__ = [
    # Planner
    "Planner", "PlannerResult", "get_planner",
    # Executor
    "Executor", "ExecutorResult", "TaskExecutionResult", "get_executor",
    # Reporter
    "Reporter", "ReporterResult", "get_reporter", "ReportMode",
    # Orchestrator
    "Orchestrator", "OrchestratorResult", "get_orchestrator",
    # Tools
    "ToolRegistry", "BaseTool", "ToolResult", "GraphRAGTool",
    # Models
    "Task", "TaskType", "TaskStatus", "generate_task_id",
    "Evidence", "EvidenceSource", "EvidenceChain", "generate_evidence_id", "generate_chain_id",
    "Plan", "PlanStatus", "generate_plan_id",
    "Report", "ReportSection", "generate_report_id",
]
