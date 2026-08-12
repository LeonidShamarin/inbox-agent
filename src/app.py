"""
HTTP-шар: приймання запитів, черга підтверджень, перегляд трас.

Логіка тут не живе — тільки транспорт. Причина та сама, що й у сусідніх
проектах: те саме приймання має однаково працювати і з CLI, і з вебхука, і з
веб-форми. Щойно маршрутизація опиниться в обробнику запиту, два з трьох
входів почнуть поводитись інакше.

Сервіс піднімається БЕЗ ключа: черга, траси й рішення людини працюють,
`POST /requests` віддає 503. Порожня форма замість роботи агента була б демо,
яке виглядає працюючим і не є ним.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.agent import Agent
from src.config import config_from_env
from src.llm import DailyQuotaExceeded, GeminiAgentLLM
from src.retrieval import KnowledgeBase
from src.schema import InboundRequest
from src.store import AgentStore, utc_now

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class RequestIn(BaseModel):
    text: str
    sender: str = "Користувач"
    channel: str = "web"


class DecisionIn(BaseModel):
    decision: str
    decided_by: str = "web-ui"


def create_app(
    db_path: Path | str = "state/agent.sqlite3",
    kb_dir: Path | str = "data/kb",
    agent: Optional[Agent] = None,
    store: Optional[AgentStore] = None,
) -> FastAPI:
    """
    `agent` і `store` передаються ззовні в тестах — з фейковим LLM замість
    Gemini. Передавати сховище окремо доводиться тому, що обробники тримають
    його в замиканні: підміна `app.state.store` виглядала б робочою і мовчки
    не діяла б.
    """
    app = FastAPI(title="inbox-agent", version="1.0")
    cfg = config_from_env()
    store = store if store is not None else AgentStore(db_path)

    if agent is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if api_key:
            llm = GeminiAgentLLM(api_key=api_key, model_name=cfg.model, rpm=cfg.rpm,
                                 temperature=cfg.temperature)
            agent = Agent(llm=llm, store=store, kb=KnowledgeBase.from_dir(Path(kb_dir)), cfg=cfg)
        else:
            logger.warning("GEMINI_API_KEY не заданий — приймання запитів вимкнено")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "agent_enabled": agent is not None,
            "model": cfg.model,
            "budgets": {
                "max_steps": cfg.max_steps,
                "max_tokens": cfg.max_tokens,
                "timeout_s": cfg.timeout_s,
            },
            "require_confirmation": cfg.require_confirmation,
            "sessions": store.counts_by_status(),
        }

    @app.post("/requests")
    async def submit(body: RequestIn) -> dict:
        if agent is None:
            raise HTTPException(status_code=503, detail="агент вимкнено: немає GEMINI_API_KEY")
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="порожній запит")

        request = InboundRequest(
            request_id=f"web-{uuid.uuid4().hex[:8]}",
            channel=body.channel,  # type: ignore[arg-type]
            sender=body.sender,
            text=text,
            received_at=utc_now(),
        )
        try:
            session = await agent.handle(request)
        except DailyQuotaExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return session.model_dump(mode="json")

    @app.get("/sessions")
    def sessions(limit: int = 30) -> dict:
        items = store.recent(limit=limit)
        return {
            "count": len(items),
            "awaiting": sum(1 for s in items if s.status == "awaiting_confirmation"),
            "items": [
                {
                    "session_id": s.session_id,
                    "status": s.status,
                    "sender": s.request.sender,
                    "channel": s.request.channel,
                    "text": s.request.text,
                    "created_at": s.created_at,
                    "tools": s.tool_names(),
                    "steps": s.budget.steps,
                    "tokens": s.budget.tokens,
                    "pending_tool": s.pending_tool.name if s.pending_tool else None,
                    "final_answer": s.final_answer,
                    "abort_reason": s.abort_reason,
                }
                for s in items
            ],
        }

    @app.get("/sessions/{session_id}")
    def session_detail(session_id: str) -> dict:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="сесію не знайдено")
        return {
            "session": session.model_dump(mode="json"),
            "side_effects": store.side_effects(session_id),
        }

    @app.post("/sessions/{session_id}/decision")
    async def decide(session_id: str, body: DecisionIn) -> dict:
        if agent is None:
            raise HTTPException(status_code=503, detail="агент вимкнено: немає GEMINI_API_KEY")
        if body.decision not in ("approve", "reject"):
            raise HTTPException(status_code=400, detail="decision має бути approve або reject")
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="сесію не знайдено")
        try:
            session = await agent.resume(session, body.decision, decided_by=body.decided_by)  # type: ignore[arg-type]
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return session.model_dump(mode="json")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app
