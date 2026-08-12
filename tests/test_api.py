"""HTTP-шар: приймання, черга, рішення людини."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.agent import Agent
from src.app import create_app
from src.config import AgentConfig
from tests.conftest import ScriptedLLM, call, say


@pytest.fixture
def api(store, kb):
    llm = ScriptedLLM(
        [call("reply_with_template", template="acknowledge"), say("Відповідь надіслано.")]
    )
    agent = Agent(llm=llm, store=store, kb=kb, cfg=AgentConfig())
    return TestClient(create_app(agent=agent, store=store))


def test_health_exposes_budgets(api):
    body = api.get("/health").json()
    assert body["agent_enabled"] is True
    assert body["budgets"]["max_steps"] > 0
    assert body["require_confirmation"] is True


def test_submit_runs_the_agent_and_pauses(api):
    body = api.post("/requests", json={"text": "Скільки днів відпустки?"}).json()
    assert body["status"] == "awaiting_confirmation"
    assert body["pending_tool"]["name"] == "reply_with_template"


def test_empty_request_is_rejected(api):
    assert api.post("/requests", json={"text": "   "}).status_code == 400


def test_decision_endpoint_finishes_the_session(api):
    session_id = api.post("/requests", json={"text": "Питання"}).json()["session_id"]

    body = api.post(f"/sessions/{session_id}/decision", json={"decision": "approve"}).json()

    assert body["status"] == "done"
    detail = api.get(f"/sessions/{session_id}").json()
    assert detail["side_effects"]["replies"], "після підтвердження лист має бути записаний"


def test_decision_on_a_finished_session_is_409(api):
    session_id = api.post("/requests", json={"text": "Питання"}).json()["session_id"]
    api.post(f"/sessions/{session_id}/decision", json={"decision": "approve"})

    again = api.post(f"/sessions/{session_id}/decision", json={"decision": "approve"})
    assert again.status_code == 409


def test_unknown_decision_and_session(api):
    assert api.post("/sessions/none/decision", json={"decision": "approve"}).status_code == 404
    session_id = api.post("/requests", json={"text": "Питання"}).json()["session_id"]
    bad = api.post(f"/sessions/{session_id}/decision", json={"decision": "хтозна"})
    assert bad.status_code == 400


def test_sessions_list_shows_tools_and_status(api):
    api.post("/requests", json={"text": "Питання про відпустку"})
    body = api.get("/sessions").json()
    assert body["count"] == 1 and body["awaiting"] == 1
    assert body["items"][0]["pending_tool"] == "reply_with_template"


def test_service_without_key_answers_503(tmp_path, monkeypatch):
    """Без ключа сервіс піднімається, але не вдає, що вміє працювати."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = create_app(db_path=tmp_path / "n.sqlite3")
    with TestClient(app) as client:
        assert client.get("/health").json()["agent_enabled"] is False
        assert client.post("/requests", json={"text": "Привіт"}).status_code == 503
