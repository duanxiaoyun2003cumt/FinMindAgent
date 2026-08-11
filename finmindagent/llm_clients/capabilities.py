"""Model/provider structured-output capability detection and routing.

Phase A — Structured Reliability Gate (Final Semantic Repair):
- Static capability knowledge per provider/model
- Safe binding probe for UNKNOWN models
- StructuredCandidate: capability snapshot embedded per candidate (no re-probe)
- resolve_structured_candidates: ordered descriptors, engine never calls get_model_capabilities
- Secret sanitization centralized at persistence boundary

This is the SINGLE source of truth for structured-output capability.
Engine MUST NOT re-implement model capability detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel


class StructuredOutputMode(str, Enum):
    """How a model supports (or doesn't) structured output."""
    NATIVE = "native"
    TOOL_CALL = "tool_call"
    JSON_SCHEMA = "json_schema"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelCapabilities:
    """Immutable capability snapshot for one provider/model pair."""
    provider: str
    model: str
    supports_structured_output: bool = False
    supports_tool_calling: bool = False
    structured_output_mode: StructuredOutputMode = StructuredOutputMode.UNKNOWN


# ---------------------------------------------------------------------------
# Static capability knowledge (project-verified facts only)
# ---------------------------------------------------------------------------

# Retired DeepSeek API aliases retained for backwards-compatible config
# handling. They are not current production defaults.
_DEEPSEEK_UNSUPPORTED_MODELS = frozenset({"deepseek-reasoner"})

# Current DeepSeek V4 production models. Structured output runs through
# LangChain function_calling on a structured-only client with thinking
# disabled (provider contract: thinking mode rejects tool_choice). The
# production mechanism is therefore TOOL_CALL, not NATIVE.
_DEEPSEEK_V4_TOOL_CALL_MODELS = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})


def detect_capabilities(provider: str, model: str) -> ModelCapabilities:
    """Static capability snapshot from provider + model name."""
    provider_lower = provider.lower()
    model_lower = model.lower()
    if provider_lower == "deepseek":
        if model_lower in _DEEPSEEK_UNSUPPORTED_MODELS:
            return ModelCapabilities(
                provider=provider, model=model,
                supports_structured_output=False, supports_tool_calling=False,
                structured_output_mode=StructuredOutputMode.UNSUPPORTED,
            )
        if model_lower in _DEEPSEEK_V4_TOOL_CALL_MODELS:
            return ModelCapabilities(
                provider=provider, model=model,
                supports_structured_output=True, supports_tool_calling=True,
                structured_output_mode=StructuredOutputMode.TOOL_CALL,
            )
        # Retired DeepSeek API aliases (deepseek-chat, deepseek-v3) retained
        # for backwards-compatible config handling. They are not current
        # production defaults; their structured contract on today's API is
        # unverified → UNKNOWN (safe binding probe required).
        return ModelCapabilities(
            provider=provider, model=model,
            supports_structured_output=False, supports_tool_calling=False,
            structured_output_mode=StructuredOutputMode.UNKNOWN,
        )
    return ModelCapabilities(
        provider=provider, model=model,
        supports_structured_output=False, supports_tool_calling=False,
        structured_output_mode=StructuredOutputMode.UNKNOWN,
    )


class _StructuredCapabilityProbe(BaseModel):
    ok: bool = True


def probe_capability(llm: Any) -> bool:
    """Safe binding probe — does NOT invoke the API."""
    try:
        llm.with_structured_output(_StructuredCapabilityProbe)
        return True
    except Exception:
        return False


def get_model_capabilities(llm: Any, provider: str) -> ModelCapabilities:
    """Resolve capabilities for a live LLM instance (static + probe).

    Called ONCE per candidate by ``resolve_structured_candidates``.
    Engine MUST NOT call this directly.
    """
    model = getattr(llm, "model_name", "unknown")
    caps = detect_capabilities(provider, model)
    if caps.supports_structured_output or caps.structured_output_mode == StructuredOutputMode.UNSUPPORTED:
        return caps
    if probe_capability(llm):
        return ModelCapabilities(
            provider=provider, model=model,
            supports_structured_output=True, supports_tool_calling=True,
            structured_output_mode=StructuredOutputMode.TOOL_CALL,
        )
    return caps


# ---------------------------------------------------------------------------
# A2: StructuredCandidate — capability snapshot embedded per candidate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructuredCandidate:
    """One ordered candidate in the structured-output fallback chain.

    ``capabilities`` is computed ONCE by the capability layer.
    Engine reads it directly — no re-probe, no re-call to
    ``get_model_capabilities``.
    """

    llm: Any
    model: str
    provider: str
    source: str  # "primary", "fallback:1", "fallback:2"
    capabilities: ModelCapabilities


def resolve_structured_candidates(
    *,
    primary_llm: Any,
    provider: str,
    fallback_llms: list[Any] | None = None,
) -> list[StructuredCandidate]:
    """Resolve ordered structured-output candidates with embedded capabilities.

    Returns de-duplicated ``StructuredCandidate`` list.  Each candidate's
    ``capabilities`` field is computed exactly once — the engine iterates
    the list directly and never re-probes.

    Order: primary → explicit fallback → other tier.
    Duplicates (same model name) are skipped.
    """
    candidates: list[StructuredCandidate] = []
    seen_models: set[str] = set()

    for idx, llm in enumerate([primary_llm] + (fallback_llms or [])):
        if llm is None:
            continue
        model = getattr(llm, "model_name", "unknown")
        if model in seen_models:
            continue
        seen_models.add(model)
        source = "primary" if idx == 0 else f"fallback:{idx}"
        caps = get_model_capabilities(llm, provider)
        candidates.append(StructuredCandidate(
            llm=llm, model=model, provider=provider,
            source=source, capabilities=caps,
        ))

    return candidates


# ---------------------------------------------------------------------------
# CandidateAttempt for per-attempt diagnostics
# ---------------------------------------------------------------------------

@dataclass
class CandidateAttempt:
    """Record of one candidate model's structured-output attempt.

    Not frozen — engine writes failure info during the attempt loop.
    Failure messages are NOT sanitized here; sanitization happens at the
    diagnostics persistence boundary (``sanitize_diagnostic_message``).
    """

    model: str
    provider: str
    source: str
    capability_mode: str = "unknown"
    capability_supported: bool = False
    attempted: bool = False
    success: bool = False
    stage: str | None = None       # capability | binding | invoke | validation
    failure_type: str | None = None
    failure_message: str | None = None


# ---------------------------------------------------------------------------
# Legacy StructuredRoute (kept for backwards compat; prefer candidate list)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructuredRoute:
    """Legacy route result — prefer iterating ``StructuredCandidate`` list."""
    llm: Any | None
    source: str
    primary_supported: bool
    fallback_used: bool
    reason: str | None = None


# ---------------------------------------------------------------------------
# Secret sanitization — centralized at persistence boundary
# ---------------------------------------------------------------------------

_SECRET_REDACTION_PATTERNS: list[tuple[str, str]] = [
    (r'Authorization:\s*Bearer\s+\S+', 'Authorization: [REDACTED]'),
    (r'Authorization:\s*Basic\s+\S+', 'Authorization: [REDACTED]'),
    (r'api_key[=:\s]+\S+', 'api_key=[REDACTED]'),
    (r'api-key[=:\s]+\S+', 'api-key=[REDACTED]'),
    (r'x-api-key[=:\s]+\S+', 'x-api-key=[REDACTED]'),
    (r'secret[=:\s]+\S+', 'secret=[REDACTED]'),
    (r'access_token[=:\s]+\S+', 'access_token=[REDACTED]'),
    (r'Bearer\s+\S+', 'Bearer [REDACTED]'),
    (r'sk-[a-zA-Z0-9_-]{10,}', 'sk-[REDACTED]'),
]


def sanitize_diagnostic_message(message: str, *, max_chars: int = 500) -> str:
    """Deterministically redact secrets, then truncate."""
    if not message:
        return ""
    for pattern, replacement in _SECRET_REDACTION_PATTERNS:
        message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
    if len(message) > max_chars:
        message = message[:max_chars]
    return message


def sanitize_attempt_dict(attempt: dict[str, Any]) -> dict[str, Any]:
    """Sanitize failure_message in an attempt dict at persistence time."""
    if attempt.get("failure_message"):
        attempt["failure_message"] = sanitize_diagnostic_message(attempt["failure_message"])
    return attempt


def sanitize_structured_diagnostics(diag: dict[str, Any]) -> dict[str, Any]:
    """Centralized sanitization for a full diagnostics entry.

    Call this at the persistence boundary — the engine does not need to
    remember to sanitize individual fields.
    """
    if diag.get("failure_message"):
        diag["failure_message"] = sanitize_diagnostic_message(diag["failure_message"])
    for attempt in diag.get("attempts", []):
        sanitize_attempt_dict(attempt)
    return diag


# ---------------------------------------------------------------------------
# Completeness semantics helpers (single source of truth for consumers)
# ---------------------------------------------------------------------------

# Status values returned by completeness computation.
COMPLETENESS_OK: frozenset[str] = frozenset({"complete"})
SECTION_OK_WITH_EMPTY: frozenset[str] = frozenset({"complete", "empty_valid"})
SECTION_OK_WITH_NOT_APPLICABLE: frozenset[str] = frozenset({"complete", "not_applicable"})
SECTION_OK_ALL: frozenset[str] = frozenset({"complete", "empty_valid", "not_applicable"})


def completeness_is_ok(status: str) -> bool:
    """Section is fully valid."""
    return status in COMPLETENESS_OK


def completeness_allows_empty(status: str) -> bool:
    """Section is complete or has valid empty business collection."""
    return status in SECTION_OK_WITH_EMPTY


def completeness_allows_not_applicable(status: str) -> bool:
    """Section is complete, or legitimately not applicable (user didn't select)."""
    return status in SECTION_OK_WITH_NOT_APPLICABLE


def completeness_allows_all(status: str) -> bool:
    """Section is not a failure: complete, empty-valid, or not-applicable."""
    return status in SECTION_OK_ALL
