"""Plan-related data models"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import uuid
from datetime import datetime

from .task import Task, TaskStatus


class PlanStatus(Enum):
    """Plan status"""
    DRAFT = "draft"
    VALIDATING = "validating"
    VALID = "valid"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Plan:
    """Task plan"""
    plan_id: str
    original_query: str
    tasks: List[Task] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    assumptions: List[str] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)  # Topologically sorted execution order
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_task(self, task: Task) -> None:
        """Add task"""
        self.tasks.append(task)
        self.updated_at = datetime.now()

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status"""
        task = self.get_task_by_id(task_id)
        if task:
            task.status = status
            self.updated_at = datetime.now()

    def get_pending_tasks(self) -> List[Task]:
        """Get pending tasks"""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_completed_tasks(self) -> List[Task]:
        """Get completed tasks"""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def is_all_completed(self) -> bool:
        """Check if all tasks are completed"""
        return all(t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for t in self.tasks)

    def has_failed_tasks(self) -> bool:
        """Check if any tasks failed"""
        return any(t.status == TaskStatus.FAILED for t in self.tasks)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "plan_id": self.plan_id,
            "original_query": self.original_query,
            "tasks": [t.to_dict() for t in self.tasks],
            "status": self.status.value,
            "assumptions": self.assumptions,
            "execution_order": self.execution_order,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        """Create from dictionary"""
        from .task import Task as TaskModel
        plan = cls(
            plan_id=data["plan_id"],
            original_query=data["original_query"],
            tasks=[TaskModel.from_dict(t) for t in data.get("tasks", [])],
            status=PlanStatus(data.get("status", "draft")),
            assumptions=data.get("assumptions", []),
            execution_order=data.get("execution_order", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now()
        )
        return plan


def generate_plan_id() -> str:
    """Generate unique plan ID"""
    return f"plan_{uuid.uuid4().hex[:8]}"
