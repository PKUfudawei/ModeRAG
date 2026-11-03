# Multi-Agent 智能对话系统 (moderag.agents)

基于 Plan-Execute-Report 架构，实现复杂任务自动分解、并行检索与证据链追踪。

## 核心特性

- ✅ **统一入口** - 所有查询统一走 Planner → Executor → Reporter 流程
- ✅ **自动任务分解** - Planner 调用 LLM 将查询分解为子任务，并为每个子任务选择检索模式
- ✅ **三种检索模式** - `naive`（向量检索）、`local`（实体为中心的图谱检索）、`global`（关系/跨文档合成）
- ✅ **并行执行** - 依赖感知的批次并行执行（ThreadPoolExecutor），线程安全证据合并
- ✅ **容错重试** - 任务失败自动重试（最多 `max_retries` 次），失败任务不影响其它任务
- ✅ **证据链追踪** - 完整记录每条证据的来源、得分与任务归属
- ✅ **可解释推理** - Reporter 支持 CONCISE（默认）与 REASONING 双模式，后者输出推理步骤与证据引用

## 架构设计

```
用户查询
   ↓
Planner — LLM 任务分解，为每个子任务选择检索模式（naive / local / global），产出 tasks + execution_order；
          请求的模式不在可用工具集内时，用确定性 fallback 映射改写成可用模式
   ↓
Executor — 依赖感知的批次并行执行（线程池），失败自动重试（max_retries），证据链在锁下合并
   ↓
Reporter — 汇总证据链生成最终答案（默认 CONCISE；REASONING 模式额外产出推理步骤与证据引用）
   ↓
最终答案 + 证据链
```

## 目录结构

```
moderag/agents/
├── __init__.py           # 模块导出
├── models/               # 核心数据模型
│   ├── task.py           # Task / TaskType (naive|local|global) / TaskStatus
│   ├── evidence.py       # Evidence / EvidenceSource / EvidenceChain
│   ├── plan.py           # Plan / PlanStatus
│   └── report.py         # Report / ReportSection / ReasoningStep / EvidenceTrace
├── tools/                # 工具层
│   ├── base.py           # BaseTool 抽象基类 / ToolResult
│   ├── registry.py       # ToolRegistry（注册 naive / local / global）+ SearchModeTool 包装
│   └── graphrag_tool.py  # GraphRAG 工具（三模式搜索入口）
├── planner.py            # Planner Agent (LLM 任务分解 + 模式选择)
├── executor.py           # Executor Agent (依赖感知并行执行 + 重试)
├── reporter.py           # Reporter Agent (报告生成, CONCISE/REASONING)
└── orchestrator.py       # Orchestrator (统一入口)
```

## 核心组件

### 1. Orchestrator (编排器)

统一入口，所有查询都走 Multi-Agent 流程：

```python
from moderag.agents import get_orchestrator

orchestrator = get_orchestrator()

result = orchestrator.process_query("What are the physics capabilities of the EIC?")
print(result.answer)           # 最终答案
print(result.plan)             # 任务计划（tasks + execution_order）
print(result.report)           # 完整报告（含证据链）
```

### 2. Planner (任务规划)

调用 LLM 将查询分解为子任务；每个子任务带一个检索模式 `task_type ∈ {naive, local, global}`。
当 Planner 请求的模式不在可用工具集中（消融实验场景），会用确定性 fallback 改写成可用模式，
保证每个 plan 都可执行：

```python
from moderag.agents.planner import Planner

planner = Planner()                       # 默认可用工具: naive, local, global
planner = Planner(available_tools=["local", "global"])  # 限制工具集

result = planner.plan("What is the luminosity range of the EIC?")
print(result.success, len(result.plan.tasks), result.plan.execution_order)
```

### 3. Executor (任务执行)

依赖感知的并行任务执行，支持失败重试：

```python
from moderag.agents.executor import Executor

# 配置：最大重试 1 次
executor = Executor(
    max_parallel_workers=16,
    max_retries=1            # 失败后重试次数
)
result = executor.execute_parallel(plan)

# 查看任务执行详情
for r in result.task_results:
    print(f"{r.task.task_id}: 成功={r.success} 重试{r.retry_count}次")
```

#### 容错策略

| 场景 | 处理 |
|------|------|
| 任务失败 | 自动重试（最多 `max_retries` 次） |
| 所有尝试失败 | 跳过任务，记录错误，继续其他任务 |

```
任务失败 → 重试 → 跳过
```

> 说明：运行时**没有**把失败任务改写为其它模式的降级机制；"模式回退"只发生在 Planner 阶段
> （不可用模式 → 确定性 fallback 改写）。

### 4. Reporter (报告生成)

整合执行结果，生成带证据链的最终答案：

```python
from moderag.agents.reporter import Reporter, ReportMode

reporter = Reporter()                      # 默认 CONCISE 模式
result = reporter.generate(executor_result, plan)

if result.success:
    report = result.report
    print(report.final_answer)
    print(report.structured_evidence)      # 结构化证据（含 top_evidence）
```

REASONING 模式额外产出推理步骤（claims + 置信度 + 不确定性标注）与证据引用，可用于展示证据链追踪：

```python
reporter = Reporter(mode=ReportMode.REASONING)
```

## 数据模型

### Task（任务）

```python
from moderag.agents.models import Task, TaskType

Task(
    task_id="task_001",
    task_type=TaskType.LOCAL,   # NAIVE | LOCAL | GLOBAL
    query="EIC detectors",
    description="检索 EIC 的探测器实体",
    depends_on=["task_000"]     # 依赖的任务 ID
)
```

### Evidence（证据）

```python
from moderag.agents.models import Evidence, EvidenceSource

Evidence(
    evidence_id="evidence_xxx",
    source=EvidenceSource.LOCAL,   # NAIVE | LOCAL | GLOBAL
    content="EIC 的探测器包括 ...",
    score=0.92,
    task_id="task_001"
)
```

### EvidenceChain（证据链）

```python
EvidenceChain(
    chain_id="chain_xxx",
    query="用户查询",
    evidence_list=[...],      # 所有证据
    reasoning_steps=[...],    # 推理步骤
    graph_paths=[...]         # 图谱路径
)
```

## 使用示例

```python
from moderag.agents import get_orchestrator

orchestrator = get_orchestrator()

result = orchestrator.process_query("Compare the physics capabilities of EIC and EicC")

if result.success:
    print("答案:", result.answer)
    print("任务数:", len(result.plan.tasks))
    print("执行顺序:", result.plan.execution_order)
    print("证据链:", result.report.structured_evidence)
```

## 扩展

### 添加新的检索模式工具

Executor 通过 `ToolRegistry.create_tool(task_type)` 取工具并调用 `structured_search(payload)`。
新工具实现该接口后即可注册成新的任务类型：

```python
from moderag.agents.tools.registry import ToolRegistry
from moderag.agents.tools.base import BaseTool, ToolResult

class MyTool(BaseTool):
    def get_name(self) -> str:
        return "my_tool"

    def search(self, query: str, **kwargs) -> ToolResult:
        ...  # 检索逻辑

    def structured_search(self, payload) -> dict:
        result = self.search(payload.get("query", ""), **payload)
        return {
            "success": result.success,
            "answer": result.answer,
            "retrieval_results": result.evidence,
            "entities": result.entities,
            "edges": result.edges,
            "error": result.error,
        }

ToolRegistry.register("my_tool", MyTool)
```

### 限制 Planner 的可用工具（消融）

```python
from moderag.agents.planner import Planner

planner = Planner(available_tools=["naive", "local"])
```

## 设计决策

### 为什么统一走 Multi-Agent？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Router 分流** | 简单查询更快 | 维护两套分支；检索能力单一 |
| **统一 Multi-Agent** | 单一路径、可组合多模式检索、证据链统一 | 每次查询多两次 LLM 调用 |

选择统一 Multi-Agent 的原因：
1. **检索组合** - 一个查询可同时使用 naive + local + global，多模式证据互补
2. **证据链统一** - 所有证据来源、得分、任务归属都被完整记录
3. **可维护性** - 单一路径，无额外分支

代价：每次查询额外 2 次 LLM 调用（1 次 Planner 分解 + 1 次 Reporter 汇总）；实际评测中
大部分查询只分解为 1 个子任务，开销约 7.4k–8.6k tokens/query（见论文实验部分）。
