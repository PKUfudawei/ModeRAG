"""Planner Agent - Task Decomposition"""
import json
import logging
from typing import List, Optional, Dict, Any, Set
from langchain_core.messages import HumanMessage, SystemMessage

from .models.plan import Plan, PlanStatus, generate_plan_id
from .models.task import Task, TaskType, TaskStatus, generate_task_id

logger = logging.getLogger(__name__)

# System prompt template for planner (LightRAG style)
PLANNER_SYSTEM_PROMPT_TEMPLATE = """---Role---

You are a task planning expert for a GraphRAG retrieval system. Your task is to convert user queries into executable retrieval tasks by selecting the appropriate retrieval strategy.

---Available Tools---

**CRITICAL: You can ONLY use the tools listed below. Never use tools not in this list.**

{available_tools}

---Tool Descriptions---

1. **naive** - Naive Vector Retrieval
   - Pure vector similarity search with two-stage recall (top-20) + rerank (top-5)
   - Returns: top 5 most relevant text chunks by semantic similarity
   - BEST FOR:
     * Simple factual questions: "What is X", "Where is X located"
     * Numeric value queries: beam energies, luminosity, dimensions, counts, specifications
     * General concept explanations and definitions
     * Queries about specific document content or text passages
     * When you don't need entity relationships or cross-document synthesis

2. **local** - Local Graph Retrieval (LightRAG style)
   - Workflow: Extract low-level keywords (entities, proper nouns, technical terms) → Entity embedding search (top-10) → Degree-based rerank → BFS expand connected edges → (degree+weight) dual-sort → Collect chunks → Rerank
   - Returns: top 10 entities (by degree), top 5 edges (by degree+weight), reranked chunks, graph context
   - BEST FOR:
     * Entity-centric queries: "What detectors does EIC have", "Which experiments use EicC"
     * Entity relationship queries: "What is the relationship between X and Y"
     * Named entity lookups: specific facilities, collaborations, experiments, institutions
     * Queries where you need to explore connections from specific entities
     * When the query focuses on concrete entities and their immediate neighbors

3. **global** - Global Graph Retrieval (LightRAG style)
   - Workflow: Extract high-level keywords (concepts, themes, abstract ideas) → Relationship embedding search (top-5, cosine order preserved) → Extract connected entities → Collect chunks → Rerank
   - Returns: top 5 relationships (by cosine similarity), all connected entities, reranked chunks, graph context
   - BEST FOR:
     * Cross-document synthesis: "Synthesize findings from multiple papers on X"
     * Comprehensive topic overview: "What are the key aspects of collider physics"
     * Comparative analysis: "Compare physics capabilities of EIC and EicC"
     * Multi-hop reasoning: "How does X affect Y through Z"
     * Theme-based exploration: "What research directions are emerging in TMD physics"
     * When you need to understand broad concepts and their interconnections

---Task Planning Rules---

{planning_rules}

---STRICT CONSTRAINT---

- You MUST ONLY use tools from the "Available Tools" list above
- If your planned task_type is not in the available tools list, your plan will be REJECTED
- Double-check: every task's task_type must be one of: {tool_names}"""


# Tool descriptions (for backward compatibility)
TOOL_DESCRIPTIONS = {
    "naive": {
        "name": "naive",
        "title": "Naive Vector Retrieval",
        "description": "- Pure vector similarity search with two-stage recall (top-20) + rerank (top-5)",
        "best_for": [
            "* Simple factual questions: 'What is X', 'Where is X located'",
            "* Numeric value queries: beam energies, luminosity, dimensions, counts",
            "* General concept explanations and definitions",
            "* Queries about specific document content or text passages"
        ]
    },
    "local": {
        "name": "local",
        "title": "Local Graph Retrieval (LightRAG style)",
        "description": "- Low-level keywords (entities, proper nouns) → Entity search (top-10) → Degree rerank → BFS expand edges → (degree+weight) dual-sort",
        "best_for": [
            "* Entity-centric queries: 'What detectors does EIC have'",
            "* Entity relationship queries: 'What is the relationship between X and Y'",
            "* Named entity lookups: specific facilities, collaborations, experiments",
            "* Queries where you need to explore connections from specific entities"
        ]
    },
    "global": {
        "name": "global",
        "title": "Global Graph Retrieval (LightRAG style)",
        "description": "- High-level keywords (concepts, themes) → Relationship search (top-5, cosine order) → Extract connected entities",
        "best_for": [
            "* Cross-document synthesis: 'Synthesize findings from multiple papers'",
            "* Comprehensive topic overview: 'What are the key aspects of X'",
            "* Comparative analysis: 'Compare physics capabilities of EIC and EicC'",
            "* Multi-hop reasoning: 'How does X affect Y through Z'",
            "* Theme-based exploration: 'What research directions are emerging in X'"
        ]
    },
}


def build_system_prompt(available_tools: List[str]) -> str:
    """Build system prompt with only available tools and dynamic rules."""
    if not available_tools:
        available_tools = ["naive"]  # Default fallback

    # Build available tools list (simple numbered format)
    tools_text = "\n".join([f"{i}. **{tool}**" for i, tool in enumerate(available_tools, 1)])

    # Build dynamic planning rules based on available tools
    rules_lines = []

    rules_lines.append("1. **Tool Selection Strategy** (classify from the query text, then cover the required modes):")
    if len(available_tools) == 1:
        tool = available_tools[0]
        rules_lines.append(f"   - Only `{tool}` tool is available, use it for all queries")
    else:
        rules_lines.append("   - Classify the query into ONE evidence genotype, then emit subtasks that cover it:")
        rules_lines.append("     - **G1 single-source**: asks for a number, threshold, definition or property of ONE object/paper")
        rules_lines.append("       → `naive` only. Do NOT add `local`/`global`.")
        rules_lines.append("     - **G2 entity-linking**: an entity/alias from one paper must be connected through another paper")
        if "local" in available_tools:
            rules_lines.append("       → at least one `local` subtask, plus one `naive` subtask for the concrete value/detail")
        else:
            rules_lines.append("       → at least one `naive` subtask (no `local` available)")
        rules_lines.append("     - **G3 cross-document aggregation/comparison**: the same measured quantity must be collected from >=2 papers and compared/aggregated")
        if "global" in available_tools:
            rules_lines.append("       → at least one `global` subtask, plus one `naive` subtask for the per-paper values")
        else:
            rules_lines.append("       → at least one `naive` subtask (no `global` available)")
        rules_lines.append("     - **G4 heterogeneous multi-hop**: different hops need DIFFERENT kinds of evidence")
        rules_lines.append("       (a concrete value/detail AND an entity relation AND a cross-document aggregate)")
        rules_lines.append("       → at least one `naive` AND one `local` AND one `global` subtask.")
        rules_lines.append("   - **Minimum coverage is mandatory**: for G2/G3/G4 the plan MUST contain the required modes.")
        rules_lines.append("     A plan that collapses a G2/G3/G4 query onto `naive` only is invalid.")
        rules_lines.append("   - **Do not over-use graph modes**: a G1 query must stay `naive`-only; routing single-source")
        rules_lines.append("     numeric/detail questions to `local`/`global` actively hurts answer quality.")

    rules_lines.append("")
    rules_lines.append("2. **Task Decomposition Rules**:")
    rules_lines.append("   - Decompose complex queries into subtasks; prefer independent subtasks and use")
    rules_lines.append("     `depends_on` only when a later sub-query genuinely needs an earlier task to run first")
    rules_lines.append("   - **The dependency graph MUST be a DAG**: a task must not depend on itself, and the")
    rules_lines.append("     `depends_on` relation must not contain cycles")
    rules_lines.append("   - `execution_order` MUST list every task exactly once in an order consistent with")
    rules_lines.append("     `depends_on`, i.e. a topological order of that DAG")
    rules_lines.append("   - For multiple distinct entities/facilities, create separate tasks")
    rules_lines.append("   - Example: 'Compare EicC and EIC' → 2 tasks (one per facility, different task_type where the aspects differ)")
    rules_lines.append("   - Example: 'beam energies and luminosity of EIC' → 1 task (same entity, multiple attributes)")
    rules_lines.append("   - Keep tasks focused: each task should have a clear, single purpose")

    planning_rules = "\n".join(rules_lines)
    tool_names = ", ".join([f"'{t}'" for t in available_tools])

    return PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        available_tools=tools_text,
        planning_rules=planning_rules,
        tool_names=tool_names
    )


# Human prompt for planner
PLANNER_HUMAN_PROMPT = """## Output Format

Return a JSON object with the following format:

```json
{{
    "plan_id": "plan_xxx",
    "original_query": "User's original query",
    "assumptions": ["assumption 1", "assumption 2"],
    "tasks": [
        {{
            "task_id": "task_xxx",
            "task_type": "naive" or "local" or "global",
            "query": "Specific query for retrieval",
            "description": "Task description",
            "depends_on": []
        }}
    ],
    "execution_order": ["task_xxx", "task_yyy"]
}}
```

The dependency relation induced by `depends_on` must be acyclic, and `execution_order`
must be a topological order of it: every task appears exactly once, after all of its
prerequisites.

## Task Type Selection Examples

### Numeric/Factual Query → naive
User query: "What is the luminosity range of the EIC?"
Reasoning: Query asks for a specific numeric value (luminosity range)
```json
{{"plan_id": "plan_001", "original_query": "What is the luminosity range of the EIC?", "assumptions": [], "tasks": [{{"task_id": "task_001", "task_type": "naive", "query": "EIC luminosity range", "description": "Retrieve EIC luminosity specifications", "depends_on": []}}], "execution_order": ["task_001"]}}
```

### Entity-Centric Query → local
User query: "What detectors and experiments are associated with EicC?"
Reasoning: Query focuses on a named entity (EicC) and its connected entities (detectors, experiments)
```json
{{"plan_id": "plan_001", "original_query": "What detectors and experiments are associated with EicC?", "assumptions": [], "tasks": [{{"task_id": "task_001", "task_type": "local", "query": "EicC detectors experiments", "description": "Find entities connected to EicC in the knowledge graph", "depends_on": []}}], "execution_order": ["task_001"]}}
```

### Comparative/Synthesis Query → global
User query: "Compare the physics capabilities of EIC and EicC"
Reasoning: Query requires cross-entity synthesis and comparison of broad concepts (physics capabilities)
```json
{{"plan_id": "plan_001", "original_query": "Compare the physics capabilities of EIC and EicC", "assumptions": [], "tasks": [{{"task_id": "task_001", "task_type": "global", "query": "EIC EicC physics capabilities comparison", "description": "Synthesize physics capabilities from both facilities", "depends_on": []}}], "execution_order": ["task_001"]}}
```

### Multi-Entity Decomposition → multiple tasks
User query: "What are the beam energies of EIC, EicC, and LHC?"
Reasoning: Three distinct entities, can be parallelized
```json
{{"plan_id": "plan_001", "original_query": "What are the beam energies of EIC, EicC, and LHC?", "assumptions": [], "tasks": [{{"task_id": "task_001", "task_type": "naive", "query": "EIC beam energy", "description": "Retrieve EIC beam energy", "depends_on": []}}, {{"task_id": "task_002", "task_type": "naive", "query": "EicC beam energy", "description": "Retrieve EicC beam energy", "depends_on": []}}, {{"task_id": "task_003", "task_type": "naive", "query": "LHC beam energy", "description": "Retrieve LHC beam energy", "depends_on": []}}], "execution_order": ["task_001", "task_002", "task_003"]}}
```

### Theme Exploration → global
User query: "What are the key research directions in TMD physics?"
Reasoning: Query explores a broad theme/concept across multiple sources
```json
{{"plan_id": "plan_001", "original_query": "What are the key research directions in TMD physics?", "assumptions": [], "tasks": [{{"task_id": "task_001", "task_type": "global", "query": "TMD physics research directions", "description": "Explore TMD physics themes and research directions", "depends_on": []}}], "execution_order": ["task_001"]}}
```

## Current Query
User query: {query}

Return only the JSON object without any additional text.
"""


class PlannerResult:
    """Planner execution result"""
    def __init__(
        self,
        plan: Optional[Plan] = None,
        error: Optional[str] = None,
        tokens_used: int = 0
    ):
        self.plan = plan
        self.error = error
        self.tokens_used = tokens_used
        self.success = plan is not None and error is None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "plan": self.plan.to_dict() if self.plan else None,
            "error": self.error,
            "tokens_used": self.tokens_used
        }


class Planner:
    """Task planner"""
    def __init__(self, llm=None, available_tools: Optional[List[str]] = None):
        self.llm = llm
        self._llm_initialized = False
        self.available_tools = available_tools or ["naive", "local", "global"]

    def _initialize_llm(self):
        if self._llm_initialized:
            return
        if self.llm is None:
            from moderag.models.llm import get_llm
            self.llm = get_llm(temperature=0)
        self._llm_initialized = True

    def plan(self, query: str, available_tools: Optional[List[str]] = None) -> PlannerResult:
        """Plan a query into tasks.

        Args:
            query: User query
            available_tools: List of available tool names (overrides init value if provided)
        """
        try:
            return self._llm_plan(query, available_tools)
        except Exception as e:
            logger.error(f"Planner execution failed: {e}")
            return PlannerResult(error=str(e))

    def _llm_plan(self, query: str, available_tools: Optional[List[str]] = None) -> PlannerResult:
        self._initialize_llm()

        # Use provided tools or fall back to instance default
        tools = available_tools if available_tools else self.available_tools
        available_tools_set: Set[str] = set(tools)

        messages = [
            SystemMessage(content=build_system_prompt(tools)),
            HumanMessage(content=PLANNER_HUMAN_PROMPT.format(query=query))
        ]

        # LLM occasionally returns malformed JSON; retry a couple of times before giving up.
        last_error = None
        for attempt in range(3):
            try:
                response = self.llm.invoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                tokens_used = self._extract_token_usage(response)
                plan_data = self._parse_json_response(content)

                # Validate and fix task types to only use available tools
                plan_data = self._validate_and_fix_task_types(plan_data, available_tools_set)

                plan = self._build_plan(plan_data, available_tools_set)

                return PlannerResult(plan=plan, tokens_used=tokens_used)
            except Exception as e:
                last_error = e
                if attempt < 2:
                    logger.warning(f"Planner LLM attempt {attempt + 1} failed, retrying: {e}")
        raise last_error

    def _extract_token_usage(self, response) -> int:
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

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        content = content.strip()
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
            logger.error(f"JSON parsing failed: {e}, content: {content[:200]}...")
            raise ValueError(f"Unable to parse JSON from LLM response: {e}")

    def _validate_and_fix_task_types(self, plan_data: Dict[str, Any], available_tools: Set[str]) -> Dict[str, Any]:
        """Validate task types and fix any that are not in available tools.

        Fix strategy (priority order):
        1. If naive is not available but was chosen → use local (if available) or global
        2. If local is not available but was chosen → use global (if available) or naive
        3. If global is not available but was chosen → use local (if available) or naive
        """
        fixed_plan = {**plan_data}
        tasks = fixed_plan.get("tasks", [])

        for task in tasks:
            original_type = task.get("task_type", "naive")
            if original_type not in available_tools:
                # Need to fix this task type
                fixed_type = self._get_fallback_tool(original_type, available_tools)
                logger.warning(
                    f"Task '{task.get('task_id')}' used unavailable tool '{original_type}', "
                    f"fixed to '{fixed_type}' (available: {available_tools})"
                )
                task["task_type"] = fixed_type

        return fixed_plan

    def _get_fallback_tool(self, original_type: str, available_tools: Set[str]) -> str:
        """Get the best fallback tool when original is not available.

        Fallback priority based on semantic similarity:
        - naive → local → global (for simple queries, prefer more specific)
        - local → naive → global (for relationships, naive can still find related text)
        - global → local → naive (for analysis, local is more structured)
        """
        # Define fallback priority for each tool type
        fallback_priorities = {
            "naive": ["local", "global"],
            "local": ["naive", "global"],
            "global": ["local", "naive"]
        }

        priorities = fallback_priorities.get(original_type, ["naive", "local", "global"])

        # Find first available fallback
        for fallback in priorities:
            if fallback in available_tools:
                return fallback

        # If no fallback found, use first available tool
        if available_tools:
            return next(iter(available_tools))

        return "naive"  # Ultimate fallback

    def _build_plan(self, data: Dict[str, Any], available_tools: Optional[Set[str]] = None) -> Plan:
        if available_tools is None:
            available_tools = set(self.available_tools)

        # Get default tool for empty tasks
        default_tool = next(iter(available_tools), "naive")

        tasks = []
        for task_data in data.get("tasks", []):
            task_type_str = task_data.get("task_type", default_tool)

            # Double-check task type is valid
            if task_type_str not in available_tools:
                task_type_str = self._get_fallback_tool(task_type_str, available_tools)

            task = Task(
                task_id=task_data.get("task_id", generate_task_id()),
                task_type=TaskType(task_type_str),
                query=task_data["query"],
                description=task_data.get("description"),
                depends_on=task_data.get("depends_on", []),
                parameters=task_data.get("parameters", {}),
                status=TaskStatus.PENDING
            )
            tasks.append(task)

        if not tasks:
            task = Task(
                task_id=generate_task_id(),
                task_type=TaskType(default_tool),
                query=data.get("original_query", ""),
                description="Default retrieval task",
                status=TaskStatus.PENDING
            )
            tasks.append(task)

        plan = Plan(
            plan_id=data.get("plan_id", generate_plan_id()),
            original_query=data.get("original_query", ""),
            tasks=tasks,
            status=PlanStatus.VALID,
            assumptions=data.get("assumptions", []),
            execution_order=data.get("execution_order", [t.task_id for t in tasks])
        )

        return plan


def get_planner(llm=None) -> Planner:
    return Planner(llm=llm)
