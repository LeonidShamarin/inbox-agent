"""
Спільні фікстури.

`ScriptedLLM` реалізує той самий протокол `AgentLLM`, що й Gemini, і віддає
наперед заданий сценарій ходів. Завдяки цьому весь набір перевіряє саме цикл
агента — запобіжники, паузу, відновлення — і не потребує ні ключа, ні мережі,
ні недетермінованої моделі.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent import Agent
from src.config import AgentConfig
from src.llm import ModelDecision, Turn
from src.retrieval import KnowledgeBase
from src.schema import InboundRequest, ToolCall
from src.store import AgentStore, utc_now
from src.tools import ToolSpec, build_registry

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"


def call(name: str, **args) -> ModelDecision:
    return ModelDecision(tool=ToolCall(name=name, args=args), tokens_in=100, tokens_out=40)


def say(text: str) -> ModelDecision:
    return ModelDecision(text=text, tokens_in=100, tokens_out=40)


class ScriptedLLM:
    """
    Віддає ходи зі сценарію по черзі. Коли сценарій вичерпано — завершує
    відповіддю, а не мовчить: інакше тест на бюджет неможливо відрізнити від
    тесту на порожній сценарій.
    """

    def __init__(self, script: list[ModelDecision] | None = None, fallback: str = "готово"):
        self.script = list(script or [])
        self.fallback = fallback
        self.calls: list[list[Turn]] = []

    async def decide(self, history: list[Turn], tools: list[ToolSpec]) -> ModelDecision:
        self.calls.append(list(history))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return say(self.fallback)


class LoopingLLM:
    """Завжди просить те саме — модель, яка зациклилась."""

    def __init__(self, name: str = "search_knowledge_base", **args):
        self.name = name
        self.args = args or {"query": "відпустка"}
        self.count = 0

    async def decide(self, history: list[Turn], tools: list[ToolSpec]) -> ModelDecision:
        self.count += 1
        return call(self.name, **self.args)


@pytest.fixture
def store(tmp_path) -> AgentStore:
    st = AgentStore(tmp_path / "state.sqlite3")
    yield st
    st.close()


@pytest.fixture
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_dir(KB_DIR)


@pytest.fixture
def cfg() -> AgentConfig:
    return AgentConfig()


def make_request(text: str = "Скільки днів відпустки на рік?", **kw) -> InboundRequest:
    data = {
        "request_id": kw.pop("request_id", "req-1"),
        "channel": kw.pop("channel", "email"),
        "sender": kw.pop("sender", "Олена"),
        "text": text,
        "received_at": utc_now(),
    }
    data.update(kw)
    return InboundRequest(**data)


def make_agent(llm, store, kb, cfg=None, **kwargs) -> Agent:
    return Agent(llm=llm, store=store, kb=kb, cfg=cfg or AgentConfig(),
                 registry=build_registry(), **kwargs)
