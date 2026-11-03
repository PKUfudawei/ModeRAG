#!/usr/bin/env bash
# ============================================================================
# run.sh — LHCpubs_run2: 在发布题集上跑分，结果就地写进 results/
#
# 发布题集 benchmark.jsonl 是成品（100 道，25/型），不会被脚本改写。共享
# harness 在 ../common/，每个配置把答案与指标 upsert 进 results/results.jsonl，
# 所以重跑任意配置都是在更新同一个文件，不会多出别的文件。
#
#   1) 可选重建索引（--rebuild-index）
#   2) 跑 5 个 agent 配置 + RAG 基线（naive/local/global/local+global/all/rag）
#   3) 打印结果文件的表头（= 论文主表的各配置均值行）
#
# 用法:
#   ./run.sh                      # 5 个配置 + RAG 基线
#   ./run.sh all naive            # 只跑指定配置
#   ./run.sh --no-rag --rebuild-index
#   MAX_QUESTIONS=10 WORKERS=8 ./run.sh
#
# LLM 后端用环境变量覆盖: LLM_MODEL / LLM_BASE_URL / LLM_API_KEY。
# 结果: results/results.jsonl
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT/../.." && pwd)"
COMMON_DIR="$PROJECT_ROOT/benchmarks/common"
PY="$PROJECT_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

export BENCH_ROOT="$ROOT"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RAGAS_TIMEOUT_SEC=600 RAGAS_MAX_RETRIES=8 RAGAS_MAX_WORKERS=6
export LLM_MODEL="${LLM_MODEL:-deepseek-v4-flash-ioa}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:15721/tencent/v1}"
if [ -z "${LLM_API_KEY:-}" ]; then
    LLM_API_KEY="$(grep -ao 'ttsw-[A-Za-z0-9_.-]\{10,\}' /root/.tt-switch-cli/tt-switch-cli.db 2>/dev/null | head -1 || true)"
    [ -n "$LLM_API_KEY" ] && export LLM_API_KEY
fi

WORKERS="${WORKERS:-16}"
MAX_QUESTIONS="${MAX_QUESTIONS:-100}"
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

echo "BENCH_ROOT   = $BENCH_ROOT"
echo "QUESTION SET = benchmark.jsonl ($(wc -l < "$ROOT/benchmark.jsonl" | tr -d ' ') 道)"
echo "CONFIGS      = ${CONFIGS[*]}"
echo "MAX_QUESTIONS= $MAX_QUESTIONS   WORKERS=$WORKERS   RUN_RAG=$RUN_RAG"

if [ "$REBUILD_INDEX" = true ]; then
    echo; echo "== 重建索引 =="
    "$PY" "$COMMON_DIR/build_index.py"
fi

for config in "${CONFIGS[@]}"; do
    echo; echo "== 配置: $config =="
    "$PY" "$COMMON_DIR/run_benchmark.py" \
        --max-questions "$MAX_QUESTIONS" \
        --tool-config "$config" \
        --questions-file benchmark.jsonl \
        --workers "$WORKERS" \
        --with-ragas \
        $FORCE
done

if [ "$RUN_RAG" = true ]; then
    echo; echo "== RAG 基线 (recall 20 -> top 5, 不带 reranker) =="
    "$PY" "$COMMON_DIR/run_benchmark_rag.py" \
        --max-questions "$MAX_QUESTIONS" \
        --questions-file benchmark.jsonl \
        --top-k 5 --recall-top-k 20 \
        --with-ragas \
        $FORCE
fi

echo; echo "== 结果 =="
"$PY" - "$ROOT/results/results.jsonl" <<'PYEOF'
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

echo
echo "结果: $ROOT/results/results.jsonl"
echo "配对 bootstrap 置信区间: python $PROJECT_ROOT/benchmarks/analysis/bootstrap_ci.py"
