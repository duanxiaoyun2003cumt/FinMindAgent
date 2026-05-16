"""Output-language helpers for user-facing runtime reports."""

from __future__ import annotations

import re


_CHINESE_ALIASES = {"chinese", "zh", "zh-cn", "cn", "mandarin", "simplified chinese"}


def normalize_output_language(language: str | None) -> str:
    value = (language or "English").strip()
    if not value:
        return "English"
    if value.lower() in _CHINESE_ALIASES:
        return "Simplified Chinese"
    return value


def is_english(language: str | None) -> bool:
    return normalize_output_language(language).lower() == "english"


def language_instruction(language: str | None) -> str:
    normalized = normalize_output_language(language)
    if normalized.lower() == "english":
        return ""
    return (
        "MANDATORY OUTPUT LANGUAGE: Write every user-facing report, summary, "
        f"and final decision in {normalized}. Keep ticker symbols, tool names, "
        "numeric values, dates, and the exact rating labels Buy, Overweight, "
        "Hold, Underweight, Sell unchanged."
    )


def appears_localized(text: str, language: str | None) -> bool:
    normalized = normalize_output_language(language).lower()
    if normalized == "english":
        return True
    if normalized == "simplified chinese":
        return re.search(r"[\u4e00-\u9fff]", text or "") is not None
    return True
