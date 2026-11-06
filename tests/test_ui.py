"""The demo renders in replay mode with no API key. Skipped where Streamlit isn't installed (CI)."""

import json

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.answer import Answer, Citation  # noqa: E402


def test_replay_renders_an_answer(tmp_path, monkeypatch):
    a = Answer("t", question="Why is Q3 opex over budget?", numbers={"opex_variance": 304948.87},
               scope="Q3 2026", table={"columns": ["opex_variance"], "labels": {"opex_variance": "Operating expenses variance"},
                                       "rows": [[304948.87]]},
               drivers=[{"cost_centre": "Engineering", "account": "Contractors", "type": "opex",
                         "metric": "opex_variance", "value": 264000.0}],
               explanation="Engineering contractors.", citations=[Citation("notes/x.md", "Cost impact", "q")],
               semantic_query={"metrics": ["opex_variance"]}, sql="select 1")
    cache = tmp_path / "answers.json"
    cache.write_text(json.dumps({"meta": {"model": "m", "recorded_at": "now"}, "answers": [a.to_dict()]}))
    monkeypatch.setenv("FINANCE_TICKMARK_DEMO_CACHE", str(cache))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    at = AppTest.from_file("../ui/app.py", default_timeout=60).run()
    at.session_state["example"] = "Why is Q3 opex over budget?"
    at.run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["£305k adverse"]
    assert "Sources (1)" in [e.label for e in at.expander]
