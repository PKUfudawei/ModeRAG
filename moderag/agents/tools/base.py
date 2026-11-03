"""Tool base class and result"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    answer: str
    evidence: list  # List[Dict]
    error: Optional[str] = None
    entities: List[str] = field(default_factory=list)  # Retrieved entity IDs
    edges: List[Dict[str, Any]] = field(default_factory=list)  # Retrieved edges


class BaseTool(ABC):
    """Base tool class"""

    @abstractmethod
    def search(self, query: str, **kwargs) -> ToolResult:
        """Execute search"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get tool name"""
        pass
