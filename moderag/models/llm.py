import contextvars
import os
import re
import time
from contextlib import contextmanager
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# LLM 后端选择 (provider profile)
#
# 本库同时支持两种 LLM 接入模式, 通过环境变量 LLM_MODE 切换 (默认 tencent):
#
#   tencent (默认): 走腾讯内网.
#       base_url 默认 tt-switch 代理  http://127.0.0.1:15721/tencent/v1
#       model    默认 deepseek-v4-flash-ioa
#       api_key  未设置时自动从 tt-switch 本地 CLI 库读取 (ttsw-... token);
#                需要其它内网端点 (如 tcodex 直连 21.221.123.50:10003) 时覆盖 LLM_BASE_URL。
#
#   openai: OpenAI 官方或任意 OpenAI 兼容端点.
#       base_url 默认 https://api.openai.com/v1
#       model    默认 gpt-4o-mini
#       api_key  读 OPENAI_API_KEY (或 LLM_API_KEY), 缺失时抛错。
#
# 任何模式下 LLM_MODEL / LLM_BASE_URL / LLM_API_KEY 都优先于 profile 默认值;
# get_llm() / get_json_llm() 的显式参数优先级最高 (逐调用覆盖)。
# 说明: chat_template_kwargs (enable_thinking) 是腾讯/vLLM 侧专有字段,
# 仅 tencent 模式会发送, openai 模式不发送以免 OpenAI 拒绝请求。
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODE = "tencent"

# 各模式的默认 model / base_url (均可被环境变量或显式参数覆盖)
_MODE_DEFAULTS = {
    "tencent": {
        "model": "deepseek-v4-flash-ioa",
        "base_url": "http://127.0.0.1:15721/tencent/v1",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
    },
}

# 兼容旧引用 (模块级默认值指向 tencent profile)
DEFAULT_MODEL = _MODE_DEFAULTS["tencent"]["model"]
DEFAULT_BASE_URL = _MODE_DEFAULTS["tencent"]["base_url"]
DEFAULT_API_KEY = "EMPTY"


def get_llm_mode() -> str:
    """返回当前 LLM 模式: 'tencent' (默认) 或 'openai'。"""
    mode = os.getenv("LLM_MODE", DEFAULT_LLM_MODE).strip().lower()
    if mode not in _MODE_DEFAULTS:
        raise ValueError(
            f"未知 LLM_MODE={mode!r}, 可选值: {', '.join(_MODE_DEFAULTS)}"
        )
    return mode


def _tt_switch_api_key() -> str:
    """Best-effort: 从 tt-switch 本地 CLI 数据库读取 ttsw- token (仅内网本机)。

    路径可用 TT_SWITCH_DB 覆盖; 读不到时返回空串 (调用方回退到 'EMPTY')。
    """
    db = os.getenv("TT_SWITCH_DB", "/root/.tt-switch-cli/tt-switch-cli.db")
    try:
        with open(db, "rb") as f:
            data = f.read()
        match = re.search(rb"ttsw-[A-Za-z0-9_.-]{10,}", data)
        if match:
            return match.group(0).decode()
    except OSError:
        pass
    return ""


def _resolve_llm_config(model: Optional[str] = None,
                        base_url: Optional[str] = None,
                        api_key: Optional[str] = None) -> Tuple[str, str, str]:
    """解析本次调用的 (model, base_url, api_key)。

    优先级: 显式参数 > LLM_MODEL/LLM_BASE_URL/LLM_API_KEY > LLM_MODE profile 默认值。
    """
    mode = get_llm_mode()
    defaults = _MODE_DEFAULTS[mode]

    resolved_model = model or os.getenv("LLM_MODEL") or defaults["model"]
    resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or defaults["base_url"]

    resolved_key = api_key or os.getenv("LLM_API_KEY")
    if not resolved_key:
        if mode == "tencent":
            resolved_key = _tt_switch_api_key() or "EMPTY"
        else:
            resolved_key = os.getenv("OPENAI_API_KEY") or ""

    if mode == "openai" and resolved_key in ("", "EMPTY"):
        raise ValueError(
            "LLM_MODE=openai 需要设置 OPENAI_API_KEY (或 LLM_API_KEY); "
            "走腾讯内网请设置 LLM_MODE=tencent (默认)"
        )
    return resolved_model, resolved_base_url, resolved_key


# --- LLM usage tracking (token + time) --------------------------------------
# 评测跑实验时可用 usage_scope() 统计一次查询内所有 LLM 调用的 token 与耗时
# (planner 重试等多次调用会累加)。没有活跃 scope 时包装是无操作的，不影响其它调用方。
_usage_cvar: "contextvars.ContextVar" = contextvars.ContextVar(
    "moderag_llm_usage", default=None
)


class LLMUsage:
    """Accumulator for LLM token usage & wall-clock time inside a usage_scope()."""

    def __init__(self):
        self.calls = 0            # 成功拿到响应的 LLM 调用次数
        self.attempts = 0         # 总调用尝试次数 (含失败/重试)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.seconds = 0.0        # LLM 调用墙钟耗时累计
        self.estimated = False    # 后端未返回 usage 时是否启用了估算

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.calls,
            "llm_attempts": self.attempts,
            "llm_prompt_tokens": self.prompt_tokens,
            "llm_completion_tokens": self.completion_tokens,
            "llm_total_tokens": self.total_tokens,
            "llm_seconds": round(self.seconds, 3),
            "llm_usage_estimated": self.estimated,
        }


@contextmanager
def usage_scope():
    """Context manager that collects token usage & timing of every LLM call made
    inside the block (agents + retries included).

    Example:
        with usage_scope() as u:
            result = orchestrator.process_query(q)
        stats = u.to_dict()   # -> {"llm_calls": ..., "llm_total_tokens": ..., ...}
    """
    usage = LLMUsage()
    token = _usage_cvar.set(usage)
    try:
        yield usage
    finally:
        _usage_cvar.reset(token)


def _extract_usage_tokens(response):
    """从响应提取 (prompt_tokens, completion_tokens, total_tokens)，无则返回 None。"""
    try:
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            p = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
            c = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
            t = usage_metadata.get("total_tokens") or 0
            return p, c, t
    except Exception:
        pass
    return None


def _estimate_tokens(chars: int) -> int:
    # 后端未返回 usage 时的粗估 (英文约 4 chars/token)
    return max(1, chars // 4)


def _messages_text_len(messages) -> int:
    try:
        total = 0
        for m in messages:
            content = getattr(m, "content", None)
            total += len(content) if isinstance(content, str) else len(str(content or ""))
        return total
    except Exception:
        return 0


def _response_text_len(response) -> int:
    try:
        content = getattr(response, "content", None)
        return len(content) if isinstance(content, str) else len(str(content or ""))
    except Exception:
        return 0


class _UsageTrackingChatOpenAI(ChatOpenAI):
    """在启用 usage_scope() 时记录每次 invoke 的 token 与耗时。

    评测期 (planner/reporter/RAG baseline) 均通过 get_llm/get_json_llm 创建模型并
    以同步 invoke 调用，因此覆写 invoke 即可覆盖全流程 (含 planner 重试)。
    没有活跃 usage_scope 时与原行为完全一致。
    """

    def invoke(self, messages, *args, **kwargs):
        usage = _usage_cvar.get()
        if usage is None:
            return super().invoke(messages, *args, **kwargs)
        usage.attempts += 1
        t0 = time.perf_counter()
        try:
            response = super().invoke(messages, *args, **kwargs)
        except Exception:
            usage.seconds += time.perf_counter() - t0
            raise
        usage.seconds += time.perf_counter() - t0
        usage.calls += 1

        reported = _extract_usage_tokens(response)
        if reported is not None and any(reported):
            p, c, t = reported
            if not t:
                t = p + c
        else:
            p = _estimate_tokens(_messages_text_len(messages))
            c = _estimate_tokens(_response_text_len(response))
            t = p + c
            usage.estimated = True
        usage.prompt_tokens += p
        usage.completion_tokens += c
        usage.total_tokens += t
        return response


# 创建 ChatOpenAI 模型实例 (支持 OpenAI 及兼容 API 的服务如 vLLM, Ollama 等)
def get_llm(model=None, base_url=None, api_key=None, enable_thinking=False, temperature=0.1, **kwargs):
    """
    创建 ChatOpenAI 客户端。model/base_url/api_key 未传时按 LLM_MODE 解析默认值
    (见模块头部说明); 显式传入的参数优先。

    Args:
        model: 模型名称
        base_url: API 地址
        api_key: API 密钥
        enable_thinking: 是否启用思考模式 (仅 tencent 模式生效), 默认关闭 (False)
        temperature: 采样温度，默认 0.1 以获得更稳定的输出
        **kwargs: 其他参数传递给 ChatOpenAI
    """
    mode = get_llm_mode()
    resolved_model, resolved_base_url, resolved_key = _resolve_llm_config(model, base_url, api_key)
    extra_body = {}
    if mode == "tencent":
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return _UsageTrackingChatOpenAI(
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_key,
        temperature=temperature,
        extra_body=extra_body or None,
        **kwargs
    )


def get_json_llm(model=None, base_url=None, api_key=None, enable_thinking=False, **kwargs):
    """同 get_llm, 但强制 response_format=json_object (用于结构化输出)。"""
    mode = get_llm_mode()
    resolved_model, resolved_base_url, resolved_key = _resolve_llm_config(model, base_url, api_key)
    extra_body = {"response_format": {"type": "json_object"}}
    if mode == "tencent":
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return _UsageTrackingChatOpenAI(
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_key,
        extra_body=extra_body,
        **kwargs
    )


if __name__ == "__main__":
    llm = get_llm()
    # 基本用法：使用 invoke() 方法
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of China?")
    ])
    print(f"Response: {response.content}")

    # 无 system prompt 的用法
    response = llm.invoke([
        HumanMessage(content="Explain gravity in one sentence.")
    ])
    print(f"Response: {response.content}")

    # 使用 JSON 响应格式
    json_llm = get_json_llm()
    response = json_llm.invoke([
        SystemMessage(content="You are a JSON API. Always respond with valid JSON."),
        HumanMessage(content="Return the capital of China as JSON with key 'capital'")
    ])
    print(f"JSON Response: {response.content}")
