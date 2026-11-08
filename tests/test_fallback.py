"""Фолбек між Gemini-ключами: порядок ланцюжка, перемикання при збої.

Без мережі й без ключа: ланцюжок перевіряється на стабах, білдер — на env-словниках.
"""

from __future__ import annotations

import asyncio

import pytest

from src.llm import (
    DailyQuotaExceeded,
    FallbackAgentLLM,
    GeminiAgentLLM,
    LLMError,
    ModelDecision,
    build_agent_llm,
)


class _Stub:
    def __init__(self, name: str, *, exc: Exception | None = None) -> None:
        self.name, self.exc, self.calls = name, exc, 0

    async def decide(self, history, tools) -> ModelDecision:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return ModelDecision(text=self.name)


def test_first_working_key_short_circuits() -> None:
    a, b = _Stub("a"), _Stub("b")
    out = asyncio.run(FallbackAgentLLM([("a", a), ("b", b)]).decide([], []))
    assert out.text == "a" and (a.calls, b.calls) == (1, 0)


def test_quota_on_first_key_falls_through_to_next() -> None:
    a = _Stub("gemini:GEMINI_API_KEY", exc=DailyQuotaExceeded("per-day"))
    b = _Stub("gemini:GEMINI_API_KEY_2")
    out = asyncio.run(FallbackAgentLLM([("gemini:GEMINI_API_KEY", a),
                                        ("gemini:GEMINI_API_KEY_2", b)]).decide([], []))
    assert out.text == "gemini:GEMINI_API_KEY_2" and (a.calls, b.calls) == (1, 1)


def test_all_keys_failing_raises_naming_each() -> None:
    a = _Stub("gemini:GEMINI_API_KEY", exc=LLMError("down"))
    b = _Stub("gemini:GEMINI_API_KEY_2", exc=DailyQuotaExceeded("per-day"))
    with pytest.raises(LLMError) as e:
        asyncio.run(FallbackAgentLLM([("gemini:GEMINI_API_KEY", a),
                                      ("gemini:GEMINI_API_KEY_2", b)]).decide([], []))
    assert "GEMINI_API_KEY" in str(e.value) and "GEMINI_API_KEY_2" in str(e.value)


def test_build_agent_llm_orders_keys_and_dedups() -> None:
    members = build_agent_llm(env={"GEMINI_API_KEY": "k1", "GEMINI_API_KEY_2": "k2",
                                    "GEMINI_API_KEY_3": "k3"}).members
    assert [n for n, _ in members] == ["gemini:GEMINI_API_KEY", "gemini:GEMINI_API_KEY_2",
                                       "gemini:GEMINI_API_KEY_3"]
    # a duplicated key value is listed once
    one = build_agent_llm(env={"GEMINI_API_KEY": "same", "GEMINI_API_KEY_2": "same"})
    assert isinstance(one, GeminiAgentLLM)


def test_build_agent_llm_single_key_is_bare_client_and_empty_raises() -> None:
    assert isinstance(build_agent_llm(env={"GEMINI_API_KEY": "k"}), GeminiAgentLLM)
    with pytest.raises(LLMError):
        build_agent_llm(env={})
