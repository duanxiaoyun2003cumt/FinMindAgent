"""Runtime construction and compatibility helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from finmindagent.agents.utils.memory import TradingMemoryLog
from finmindagent.dataflows.config import set_config
from finmindagent.default_config import DEFAULT_CONFIG
from finmindagent.llm_clients import create_llm_client
from finmindagent.runtime.context_manager import ContextManager
from finmindagent.runtime.engine import FinMindAgentLoop
from finmindagent.runtime.memory_manager import MemoryManager
from finmindagent.runtime.permission_manager import PermissionManager
from finmindagent.runtime.permissions.audit import PermissionAuditLog
from finmindagent.runtime.tool_registry import ToolRegistry


def create_runtime(config: dict[str, Any] | None = None, callbacks: list[Any] | None = None) -> FinMindAgentLoop:
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    set_config(cfg)
    Path(cfg["data_cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    llm_kwargs = _provider_kwargs(cfg)
    if callbacks:
        llm_kwargs["callbacks"] = callbacks

    quick_llm = _create_llm(cfg, cfg["quick_think_llm"], llm_kwargs)
    deep_llm = _create_llm(cfg, cfg["deep_think_llm"], llm_kwargs)

    # Phase A: structured-output fallback model. When the primary model
    # (deep_llm for PM/RM, quick_llm for analysts) doesn't support
    # structured output, this model is used instead for structured calls.
    # If not configured, structured-required agents fall back to free text.
    structured_fallback_model = cfg.get("structured_output_fallback_model")
    if structured_fallback_model:
        structured_llm = _create_llm(cfg, structured_fallback_model, llm_kwargs)
    elif cfg.get("structured_output", {}).get("fallback_model"):
        structured_llm = _create_llm(
            cfg, cfg["structured_output"]["fallback_model"], llm_kwargs
        )
    else:
        # Default: when deep model is unsupported for structured output,
        # use the quick model as structured fallback (it supports it).
        # Only auto-wire if models differ (avoid redundant client).
        if cfg["deep_think_llm"] != cfg["quick_think_llm"]:
            structured_llm = quick_llm
        else:
            structured_llm = None

    tool_registry = ToolRegistry()
    runtime_dir = Path(cfg["results_dir"]) / "runtime"
    memory_dir = cfg.get("memory_dir") or str(Path(cfg.get("memory_log_path", runtime_dir / "memory.md")).parent / "structured")
    audit_log = PermissionAuditLog(runtime_dir / "permission_audit.jsonl")
    permission_manager = PermissionManager(
        mode=cfg.get("permission_mode", "safe"),
        audit_log=audit_log,
    )
    memory_manager = MemoryManager(memory_dir=memory_dir, selector_llm=quick_llm)
    context_manager = ContextManager(
        tool_registry=tool_registry,
        quick_llm=quick_llm,
        max_context_tokens=int(cfg.get("runtime_context_tokens", 80_000)),
        artifact_dir=runtime_dir / "artifacts",
    )
    return FinMindAgentLoop(
        config=cfg,
        quick_llm=quick_llm,
        deep_llm=deep_llm,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        permission_manager=permission_manager,
        context_manager=context_manager,
        structured_llm=structured_llm,
    )


def run_analysis(
    ticker: str,
    trade_date: str,
    config: dict[str, Any] | None = None,
    callbacks: list[Any] | None = None,
):
    runtime = create_runtime(config=config, callbacks=callbacks)
    decision, legacy_state = runtime.propagate(ticker, trade_date)
    _store_legacy_memory(config or DEFAULT_CONFIG, ticker, trade_date, decision)
    return legacy_state, decision


def _create_llm(cfg: dict[str, Any], model: str, llm_kwargs: dict[str, Any]):
    if cfg.get("runtime_disable_llm"):
        return None
    client = create_llm_client(
        provider=cfg["llm_provider"],
        model=model,
        base_url=cfg.get("backend_url"),
        **llm_kwargs,
    )
    return client.get_llm()


def _provider_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = {}
    provider = cfg.get("llm_provider", "").lower()
    if provider == "google" and cfg.get("google_thinking_level"):
        kwargs["thinking_level"] = cfg["google_thinking_level"]
    elif provider == "openai" and cfg.get("openai_reasoning_effort"):
        kwargs["reasoning_effort"] = cfg["openai_reasoning_effort"]
    elif provider == "anthropic" and cfg.get("anthropic_effort"):
        kwargs["effort"] = cfg["anthropic_effort"]
    return kwargs


def _store_legacy_memory(config: dict[str, Any], ticker: str, trade_date: str, decision: str) -> None:
    try:
        TradingMemoryLog(config).store_decision(ticker, trade_date, decision)
    except Exception:
        pass
