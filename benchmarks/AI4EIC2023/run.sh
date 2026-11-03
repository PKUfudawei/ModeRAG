#!/usr/bin/env bash
# ============================================================================
# run.sh — AI4EIC2023 全流程 / 消融 driver
#
# 合并了旧的 run_experiments.sh（含建索引）和 run_ablation.sh（含参数解析）：
# 两者本来都只是驱动同一套配置，区别仅在于建不建索引、要不要 RAG 基线。
#
# 分工:
#   ../common/build_index.py       重建 GraphRAG 索引（--rebuild-index 时）
#   ../common/run_benchmark.py     naive / local / global / local+global / all
#   ../common/run_benchmark_rag.py RAG 基线 (recall 20 -> top 5, 不带 reranker)
#
# 用法:
#   ./run.sh                          # 默认跑全部 5 个配置 + RAG 基线
#   ./run.sh all naive                # 只跑指定配置
#   ./run.sh --rebuild-index          # 先重建索引
#   ./run.sh --no-rag                 # 跳过 RAG 基线
#   ./run.sh --workers 8 --force
#   MAX_QUESTIONS=10 ./run.sh
#
# LLM 后端通过环境变量配置 (moderag/models/llm.py):
#   LLM_MODEL / LLM_BASE_URL / LLM_API_KEY
# 默认指向腾讯内网 tt-switch 代理 + deepseek-v4-flash-ioa。
# embedding 固定走本地 bge-m3（腾讯端点没有 /embeddings）。
# 结果写入 results/（BENCH_OUTPUT_DIR 可覆盖）。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMMON_DIR="$(cd "$SCRIPT_DIR/../common" && pwd)"
VENV_PY="$PROJECT_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=python3

# The shared harness in ../common derives every path from BENCH_ROOT.
export BENCH_ROOT="$SCRIPT_DIR"

WORKERS="${WORKERS:-16}"
MAX_QUESTIONS="${MAX_QUESTIONS:-50}"
REBUILD_INDEX=false
RUN_RAG=true
FORCE=""
CONFIGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --rebuild-index) REBUILD_INDEX=true ;;
        --no-rag)        RUN_RAG=false ;;
        --workers=*)     WORKERS="${1#*=}" ;;
        --workers)       WORKERS="$2"; shift ;;
        --force|--force-regenerate) FORCE="--force-regenerate" ;;
        naive|local|global|naive+local|naive+global|local+global|all) CONFIGS+=("$1") ;;
        *) echo "WARNING: ignoring unknown argument: $1" >&2 ;;
    esac
    shift
done

if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(naive local global local+global all)
fi

# --- LLM 后端: 腾讯内网 tt-switch (仅内网可用; 可用环境变量覆盖) ---
export LLM_MODEL="${LLM_MODEL:-deepseek-v4-flash-ioa}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:15721/tencent/v1}"
if [ -z "${LLM_API_KEY:-}" ]; then
    # 从 tt-switch 本地数据库读 token (本机路径, 内网专用)
    LLM_API_KEY="$(grep -ao 'ttsw-[A-Za-z0-9_.-]\{10,\}' /root/.tt-switch-cli/tt-switch-cli.db 2>/dev/null | head -1 || true)"
    [ -n "$LLM_API_KEY" ] && export LLM_API_KEY
fi
if [ -z "${LLM_API_KEY:-}" ]; then
    echo "WARNING: LLM_API_KEY 未设置 (tt-switch token 未找到)" >&2
fi

step() {
  echo
  echo "============================================================"
  echo "== $1"
  echo "============================================================"
}

echo "BENCH_ROOT   = $BENCH_ROOT"
echo "PYTHON       = $VENV_PY"
echo "LLM_MODEL    = $LLM_MODEL"
echo "LLM_BASE_URL = $LLM_BASE_URL"
echo "API_KEY      = ${LLM_API_KEY:+set}"
echo "CONFIGS      = ${CONFIGS[*]}"
echo "MAX_QUESTIONS= $MAX_QUESTIONS   WORKERS=$WORKERS"
echo "REBUILD_INDEX= $REBUILD_INDEX    RUN_RAG=$RUN_RAG"

if [ "$REBUILD_INDEX" = true ]; then
    step "[1] 重建索引 build_index.py"
    "$VENV_PY" "$COMMON_DIR/build_index.py"
fi

step "[2] 检查索引产物"
for f in vectorstore/index.faiss vectorstore/index.pkl graph.pkl entities.pkl relationships.pkl; do
    p="$SCRIPT_DIR/indexes/$f"
    if [ -f "$p" ]; then
        echo "  OK   $f ($(du -h "$p" | cut -f1))"
    else
        echo "  MISS $f"
    fi
done

mkdir -p "$SCRIPT_DIR/results"

i=0
for config in "${CONFIGS[@]}"; do
    i=$((i + 1))
    step "[3.${i}] 配置: $config (workers=$WORKERS)"
    "$VENV_PY" "$COMMON_DIR/run_benchmark.py" \
        --max-questions "$MAX_QUESTIONS" \
        --tool-config "$config" \
        --workers "$WORKERS" \
        --with-ragas \
        $FORCE
done

if [ "$RUN_RAG" = true ]; then
    step "[4] RAG baseline (no reranker, recall 20 -> top 5)"
    "$VENV_PY" "$COMMON_DIR/run_benchmark_rag.py" \
        --max-questions "$MAX_QUESTIONS" \
        --top-k 5 --recall-top-k 20 \
        --with-ragas \
        $FORCE
fi

step "[5] 汇总"
"$VENV_PY" - "$SCRIPT_DIR/results/results.jsonl" <<'PYEOF'
import json, os, sys

path = sys.argv[1]
if not os.path.exists(path):
    print(f"  (no results file at {path})")
else:
    header = json.loads(open(path).readline())
    for cfg in header["configs"]:
        row = header["summary"][cfg]
        print(f"  {cfg:14s} n={row['n']:3d}  " + "  ".join(
            f"{k}={row[k]:.4f}" for k in ("faithfulness", "answer_relevancy", "answer_correctness")
            if row.get(k) is not None))
PYEOF

step "[6] 完成"
echo "结果目录: $SCRIPT_DIR/results/"
echo "配对 bootstrap 置信区间: python $SCRIPT_DIR/../analysis/bootstrap_ci.py"
