"""Deterministic investment report construction and rendering."""

from finmindagent.reporting.builder import build_report_schema
from finmindagent.reporting.html import (
    generate_report_artifact,
    render_report_html,
    write_report_html,
)
from finmindagent.reporting.schemas import ReportSchema

__all__ = [
    "ReportSchema",
    "build_report_schema",
    "generate_report_artifact",
    "render_report_html",
    "write_report_html",
]
