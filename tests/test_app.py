"""Smoke tests: every Streamlit page renders without exceptions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

SRC = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
PAGES = ["overview", "telemetry", "diagnosis", "graph", "performance", "history", "system"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_string(SRC.replace("nav.run()", f"page_{page}()"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
