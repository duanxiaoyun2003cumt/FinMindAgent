"""Safe, minimal Markdown rendering for agent prose.

Only a fixed allowlist of tags is ever emitted (p/h2/h3/h4/ul/ol/li/strong/
em/code/blockquote/table/thead/tbody/tr/th/td). Raw HTML is always escaped —
there is no HTML passthrough, no javascript: URLs, no arbitrary attributes,
no style attributes.
"""

from __future__ import annotations

import re
from html import escape


def render_safe_markdown(text: str) -> str:
    """Render markdown-shaped agent prose into safe HTML.

    The input is treated as untrusted text: any HTML-like syntax is escaped
    at the token level before structural tags are emitted.
    """
    if not text:
        return ""
    lines = text.splitlines()
    blocks = _split_blocks(lines)
    return "\n".join(_render_block(block) for block in blocks)


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------

def _split_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{2,4})\s+(.*)$")
_UL_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_OL_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}\s*(\|\s*:?-{2,}\s*)*\|?\s*$")


def _render_block(block: list[str]) -> str:
    first = block[0]

    # Headings
    match = _HEADING_RE.match(first)
    if match:
        level = len(match.group(1))
        content = _render_inline(match.group(2))
        return f"<h{level}>{content}</h{level}>"

    # Tables: header row | separator row | data rows
    if _is_table(block):
        return _render_table(block)

    # Lists
    if all(_UL_RE.match(line) for line in block if line.strip()):
        items = "".join(
            f"<li>{_render_inline(_UL_RE.match(line).group(1))}</li>"
            for line in block
            if line.strip()
        )
        return f"<ul>{items}</ul>"
    if all(_OL_RE.match(line) for line in block if line.strip()):
        items = "".join(
            f"<li>{_render_inline(_OL_RE.match(line).group(1))}</li>"
            for line in block
            if line.strip()
        )
        return f"<ol>{items}</ol>"

    # Blockquote
    if all(_QUOTE_RE.match(line) for line in block):
        inner = "<br>".join(
            _render_inline(_QUOTE_RE.match(line).group(1)) for line in block
        )
        return f"<blockquote>{inner}</blockquote>"

    # Paragraph (multi-line joined with <br>)
    return f"<p>{_render_inline('<br>'.join(line.strip() for line in block))}</p>"


def _is_table(block: list[str]) -> bool:
    if len(block) < 2:
        return False
    if not (block[0].strip().startswith("|") and block[0].strip().endswith("|")):
        return False
    return bool(_TABLE_SEPARATOR_RE.match(block[1]))


def _render_table(block: list[str]) -> str:
    def _cells(row: str) -> list[str]:
        stripped = row.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    header = _cells(block[0])
    rows = [_cells(line) for line in block[2:] if line.strip()]
    thead = "".join(f"<th>{_render_inline(cell)}</th>" for cell in header)
    tbody_rows = []
    for row in rows:
        cells = "".join(f"<td>{_render_inline(cell)}</td>" for cell in row)
        tbody_rows.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(tbody_rows)}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Inline rendering (token level, always escaped first)
# ---------------------------------------------------------------------------

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+?)`")


def _render_inline(fragment: str) -> str:
    """Escape first, then apply a fixed inline markup pass on the escaped text.

    Because escaping happens before tokenizing, any ``**``/``*``/`` ` `` that
    came from raw HTML cannot form a tag; they can only form markup spans over
    already-escaped content, which is safe.
    """
    escaped = escape(fragment, quote=False)
    # Inline code first so code spans cannot be re-interpreted.
    escaped = _CODE_RE.sub(
        lambda match: f"<code>{match.group(1)}</code>", escaped
    )
    # Bold (strong) spans.
    escaped = _BOLD_RE.sub(
        lambda match: f"<strong>{match.group(1)}</strong>", escaped
    )
    # Italic spans (single asterisks, not part of a strong span).
    escaped = _ITALIC_RE.sub(
        lambda match: f"<em>{match.group(1)}</em>", escaped
    )
    return escaped
