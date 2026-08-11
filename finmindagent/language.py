"""Language source of truth for FinMindAgent (current-plan v9).

Config input ("Chinese" / "English") is normalized to ONE canonical value at
the boundary; every consumer (agent prompts, HTML locale, runtime state)
reads the same canonical language. The HTML renderer's ``DEFAULT_LOCALE``
(zh-CN) remains the legacy fallback for historical states.

Canonical values:
- Chinese → ``zh-CN``  (aligns with ``localization.SUPPORTED_LOCALES``)
- English → ``en``

Only natural-language *values* are language-constrained. Machine-readable
contracts (schema keys, enum values, tickers, indicators, evidence IDs,
source identifiers) are never translated — the instruction below says so
explicitly so the LLM respects the boundary.
"""

from __future__ import annotations

from typing import Any

LANGUAGE_CHINESE = "zh-CN"
LANGUAGE_ENGLISH = "en"

_LEGACY_DEFAULT = "Chinese"

_ALIASES: dict[str, str] = {
    # Chinese
    "chinese": LANGUAGE_CHINESE,
    "zh-cn": LANGUAGE_CHINESE,
    "zh_cn": LANGUAGE_CHINESE,
    "zh": LANGUAGE_CHINESE,
    "中文": LANGUAGE_CHINESE,
    "简体中文": LANGUAGE_CHINESE,
    # English
    "english": LANGUAGE_ENGLISH,
    "en": LANGUAGE_ENGLISH,
    "en-us": LANGUAGE_ENGLISH,
    "en_us": LANGUAGE_ENGLISH,
}


def normalize_language(value: Any) -> str:
    """Normalize a configured language to the canonical value.

    ``"Chinese"`` (or any documented alias) → ``"zh-CN"``,
    ``"English"`` (or alias) → ``"en"``.
    Unknown values raise ``ValueError`` — never silently fall back to a
    language the user did not configure.
    """
    if value is None or value == "":
        value = _LEGACY_DEFAULT
    key = str(value).strip().lower()
    canonical = _ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"Unsupported output_language {value!r}; expected one of "
            f"Chinese/zh-CN/zh or English/en."
        )
    return canonical


def language_instruction(language: str) -> str:
    """Return the shared language contract injected into agent prompts.

    Short (tens of tokens), placed at the top of the prompt as part of the
    trusted system layer — never buried at the bottom of long contexts.
    """
    if normalize_language(language) == LANGUAGE_ENGLISH:
        return (
            "COMMON SYSTEM RULES\n"
            "Output language: English (en).\n"
            "Write all natural-language report content in clear, professional "
            "English.\n"
        )
    return (
        "COMMON SYSTEM RULES\n"
        "Output language: Simplified Chinese (zh-CN).\n"
        "Write all natural-language report content in clear, professional "
        "Simplified Chinese.\n"
        "This requirement applies to: summaries, analysis, arguments, "
        "rationales, evidence descriptions, risks, plans, conditions, "
        "monitoring events, recommendations and explanations.\n"
        "Preserve: ticker symbols, company names where appropriate, financial "
        "abbreviations, technical indicators (e.g. RSI, MACD, SMA), source "
        "IDs, evidence IDs, schema keys, enum values required by the schema, "
        "and proper nouns when translation would reduce precision.\n"
        "Do not translate JSON/Pydantic field names.\n"
        "Do not switch to English merely because source data is in English; "
        "interpret the evidence and write the analytical explanation in "
        "Simplified Chinese.\n"
    )
