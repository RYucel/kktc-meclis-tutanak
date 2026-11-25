"""Smoke & structural checks for Dual-Parliament wings, historical milestones, llms.txt, and WebMCP tools."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "space_static" / "index.html"
LLMS_TXT = ROOT / "space_static" / "llms.txt"
DEPLOY_PY = ROOT / "src" / "deploy_space.py"

REQUIRED_TOOLS = (
    "search_speeches",
    "compare_parliaments",
    "get_speaker_profile",
    "get_speech_context",
    "list_archive_catalog",
)

REQUIRED_MILESTONES = (
    "kurulus_1960",
    "kriz_1963",
    "doruk_1977",
    "annan_2002",
    "crans_2015",
    "mulkiyet_su",
)


def test_llms_txt_documents_webmcp_and_endpoints() -> None:
    text = LLMS_TXT.read_text(encoding="utf-8")
    for tool in REQUIRED_TOOLS:
        assert tool in text
    assert "document.modelContext" in text
    assert "speeches.parquet" in text
    assert "KC.parquet" in text


def test_deploy_space_includes_llms_txt() -> None:
    src = DEPLOY_PY.read_text(encoding="utf-8")
    assert '"llms.txt"' in src


def test_index_html_has_wings_milestones_and_webmcp() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    for tool in REQUIRED_TOOLS:
        assert f"name: '{tool}'" in html
    for ms in REQUIRED_MILESTONES:
        assert f"id: '{ms}'" in html
    assert "document.modelContext" in html
    assert "navigator.modelContext" in html
    assert "window.meclisMCP" in html
    assert 'id="wing-north"' in html
    assert 'id="wing-cy"' in html
    assert 'id="wing-compare"' in html
    assert "CY_PARLIAMENT_PERIODS" in html
    assert "compare-results" in html
