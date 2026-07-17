"""Regression tests for UI HTML injection hardening.

The single-page UI builds several fragments with ``innerHTML``.  Any value
that comes from an agent, memory, backend response, or exception must be HTML
escaped before it is interpolated into those fragments.
"""

from pathlib import Path


UI_HTML = (
    Path(__file__).resolve().parents[1]
    / "memanto"
    / "app"
    / "ui"
    / "static"
    / "index.html"
).read_text(encoding="utf-8")


def test_dashboard_agent_profile_fields_are_html_escaped():
    """Stored agent metadata must not be inserted raw into dashboard HTML."""
    assert "${escHtml(agent.agent_id)}" in UI_HTML
    assert "${escHtml(agent.description ||" in UI_HTML
    assert "${escHtml(agent.namespace ||" in UI_HTML
    assert "${agent.description ||" not in UI_HTML


def test_answer_source_titles_are_html_escaped():
    """RAG source titles are memory-derived and rendered through innerHTML."""
    assert "${escHtml(trunc(s.title || s.id || 'memory', 30))}" in UI_HTML
    assert "${trunc(s.title || s.id || 'memory', 30)}" not in UI_HTML


def test_memory_table_metadata_is_escaped_before_innerhtml():
    """Memory metadata displayed in expandable rows must be encoded."""
    assert "${escHtml(m.provenance)}" in UI_HTML
    assert "${escHtml(m.type ||" in UI_HTML
    assert "ID: ${escHtml(memId ||" in UI_HTML
    assert "Source: ${escHtml(m.source ||" in UI_HTML
    assert "forgetMemory('${" not in UI_HTML
    assert 'data-memory-id="${attrEsc(memId)}"' in UI_HTML


def test_innerhtml_error_messages_are_html_escaped():
    """Exception messages may contain server-provided text; escape in HTML."""
    assert "Could not load agent: ${escHtml(e.message)}" in UI_HTML
    assert "Session may be expired: ${escHtml(e.message)}" in UI_HTML
    assert "Failed to load config: ${escHtml(e.message)}" in UI_HTML
    assert "Failed to load analytics: ${escHtml(e.message)}" in UI_HTML
    assert "Could not load agent: ${e.message}" not in UI_HTML
    assert "Session may be expired: ${e.message}" not in UI_HTML
    assert "Failed to load config: ${e.message}" not in UI_HTML
    assert "Failed to load analytics: ${e.message}" not in UI_HTML
