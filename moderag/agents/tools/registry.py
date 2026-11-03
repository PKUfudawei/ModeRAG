"""Tool registry"""
from typing import Dict, Type, Any, Optional, Callable

from .base import BaseTool, ToolResult
from .graphrag_tool import GraphRAGTool


class SearchModeTool(BaseTool):
    """GraphRAG retrieval mode tool wrapper"""

    def __init__(
        self,
        mode: str,
        storage_path: str = "./storage/graphrag_index",
        reranker_model: Optional[str] = None,
        reranker_device: str = "cuda:0",
        # Defaults mirror GraphRAGTool's library defaults.
        recall_top_k: int = 20,
        rerank_top_k: int = 5,
        top_k_entities: int = 10,
        top_k_relationships: int = 5,
        max_recall_chunks: int = 40,
    ):
        self.mode = mode
        self.graphrag_tool = GraphRAGTool(
            storage_path=storage_path,
            reranker_model=reranker_model,
            reranker_device=reranker_device,
            recall_top_k=recall_top_k,
            rerank_top_k=rerank_top_k,
            top_k_entities=top_k_entities,
            top_k_relationships=top_k_relationships,
            max_recall_chunks=max_recall_chunks,
        )

    def get_name(self) -> str:
        return self.mode

    def search(self, query: str, **kwargs) -> ToolResult:
        """Execute search"""
        return getattr(self.graphrag_tool, f"{self.mode}_search")(query, **kwargs)

    def structured_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Structured search interface"""
        query = payload.get("query", "")
        # Remove query from payload to avoid duplicate argument
        search_kwargs = {k: v for k, v in payload.items() if k != "query"}
        result = getattr(self.graphrag_tool, f"{self.mode}_search")(query, **search_kwargs)
        return {
            "success": result.success,
            "answer": result.answer,
            "retrieval_results": result.evidence,
            "error": result.error
        }


class ToolRegistry:
    """Tool registry"""

    _tools: Dict[str, Any] = {}  # Can be class or factory function

    @classmethod
    def register(cls, name: str, tool: Any) -> None:
        """Register tool (class or factory function)"""
        cls._tools[name] = tool

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister tool"""
        if name in cls._tools:
            del cls._tools[name]

    @classmethod
    def get_tool(cls, name: str) -> Optional[Any]:
        """Get tool class or factory function"""
        return cls._tools.get(name)

    @classmethod
    def create_tool(cls, name: str, **kwargs) -> Optional[BaseTool]:
        """Create tool instance"""
        tool = cls.get_tool(name)
        if tool:
            try:
                # If factory function
                if callable(tool) and not isinstance(tool, type):
                    return tool(**kwargs)
                # If class
                elif isinstance(tool, type):
                    return tool(**kwargs)
            except TypeError:
                pass
        return None

    @classmethod
    def list_tools(cls) -> list:
        """List all registered tools"""
        return list(cls._tools.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear registry"""
        cls._tools.clear()


# Register three retrieval modes as independent tools
def create_naive_tool(**kwargs) -> BaseTool:
    return SearchModeTool(mode="naive", **kwargs)

def create_local_tool(**kwargs) -> BaseTool:
    return SearchModeTool(mode="local", **kwargs)

def create_global_tool(**kwargs) -> BaseTool:
    return SearchModeTool(mode="global", **kwargs)

ToolRegistry.register("naive", create_naive_tool)
ToolRegistry.register("local", create_local_tool)
ToolRegistry.register("global", create_global_tool)

__all__ = ["ToolRegistry", "BaseTool", "ToolResult", "SearchModeTool"]
