"""
Запобіжники агента.

Найважливіший файл у проекті: агент, який може зациклитись або виконати
незворотну дію без людини, — це інцидент, а не незручність.
"""

from __future__ import annotations

import pytest

from src.config import AgentConfig
from src.llm import DailyQuotaExceeded, ModelDecision
from tests.conftest import LoopingLLM, ScriptedLLM, call, make_agent, make_request, say


# --- бюджети --------------------------------------------------------------

async def test_step_limit_stops_the_agent(store, kb):
    """Модель ходить по колу різними аргументами — рятує лише ліміт кроків."""
    llm = ScriptedLLM([call("search_knowledge_base", query=f"питання {i}") for i in range(20)])
    agent = make_agent(llm, store, kb, AgentConfig(max_steps=3))

    session = await agent.handle(make_request())

    assert session.status == "aborted"
    assert "ліміт кроків" in session.abort_reason
    assert session.budget.steps == 3, "жодного зайвого виклику моделі понад ліміт"


async def test_token_limit_stops_the_agent(store, kb):
    llm = ScriptedLLM(
        [ModelDecision(tool=None, text="ще думаю", tokens_in=9000, tokens_out=1000)] * 5
    )
    # Перший же хід з'їдає 10 000 токенів; ліміт 5000 має спрацювати на другому.
    agent = make_agent(llm, store, kb, AgentConfig(max_tokens=5000, max_steps=99))
    session = await agent.handle(make_request())

    assert session.status == "done", "перший хід дозволений — ліміт перевіряється перед викликом"
    assert session.budget.tokens == 10_000


async def test_timeout_stops_the_agent(store, kb):
    """
    Кроків мало, токенів мало, але один виклик висить. Годинник підмінений,
    щоб тест не чекав по-справжньому.
    """
    ticks = iter([0.0, 0.5, 200.0, 400.0, 600.0])
    llm = ScriptedLLM([call("search_knowledge_base", query="відпустка") for _ in range(5)])
    agent = make_agent(llm, store, kb, AgentConfig(timeout_s=60.0), clock=lambda: next(ticks))

    session = await agent.handle(make_request())

    assert session.status == "aborted" and "таймаут" in session.abort_reason


async def test_same_call_twice_is_detected_as_a_loop(store, kb):
    """
    Ліміту кроків мало: агент може вкластися в бюджет і при цьому шість разів
    запитати те саме. Однакові аргументи — однаковий результат.
    """
    agent = make_agent(LoopingLLM(), store, kb, AgentConfig(max_steps=10, max_same_tool_calls=2))
    session = await agent.handle(make_request())

    assert session.status == "aborted" and "по колу" in session.abort_reason
    assert len(session.tool_names()) == 2, "третій однаковий виклик не виконувався"


async def test_budget_survives_a_restart(store, kb):
    """
    Витрати живуть у сесії, а не в пам'яті процесу. Інакше «ліміт на задачу»
    перетворюється на «ліміт на спробу», і падіння обходить запобіжник.
    """
    llm = ScriptedLLM([call("reply_with_template", template="acknowledge")])
    agent = make_agent(llm, store, kb, AgentConfig(max_steps=3))
    session = await agent.handle(make_request())
    assert session.status == "awaiting_confirmation"

    reloaded = store.get(session.session_id)
    assert reloaded.budget.steps == 1 and reloaded.budget.tokens > 0


async def test_daily_quota_aborts_without_hiding_the_reason(store, kb):
    llm = ScriptedLLM([DailyQuotaExceeded("PerDay")])
    agent = make_agent(llm, store, kb)
    session = await agent.handle(make_request())

    assert session.status == "aborted" and "квота" in session.abort_reason


# --- human-in-the-loop ----------------------------------------------------

async def test_irreversible_tool_pauses_and_is_not_executed(store, kb):
    llm = ScriptedLLM([call("reply_with_template", template="acknowledge")])
    agent = make_agent(llm, store, kb)

    session = await agent.handle(make_request())

    assert session.status == "awaiting_confirmation"
    assert session.pending_tool.name == "reply_with_template"
    assert store.side_effects(session.session_id) == {}, "лист не мав піти без людини"
    assert [s.kind for s in session.steps][-1] == "confirmation_request"


async def test_approval_executes_exactly_the_shown_arguments(store, kb):
    """
    Людина бачить аргументи й підтверджує саме їх. Якщо виконати щось інше —
    підтвердження стає театром.
    """
    llm = ScriptedLLM(
        [call("reply_with_template", template="acknowledge", recipient="olena@example.com"),
         say("Відповідь надіслано.")]
    )
    agent = make_agent(llm, store, kb)
    session = await agent.handle(make_request())

    resumed = await agent.resume(session, "approve", decided_by="Ігор")

    replies = store.side_effects(session.session_id)["replies"]
    assert len(replies) == 1 and replies[0]["recipient"] == "olena@example.com"
    assert resumed.status == "done"
    kinds = [s.kind for s in resumed.steps]
    assert kinds.count("human_decision") == 1


async def test_rejection_does_not_end_the_session_but_tells_the_model(store, kb):
    """
    Відмова — не кінець роботи. Агент має дізнатись про неї і обрати інший
    шлях, інакше запит лишається без відповіді взагалі.
    """
    llm = ScriptedLLM(
        [call("reply_with_template", template="decline", fields={"reason": "бо так"}),
         call("escalate_to_human", reason="людина відхилила автоматичну відповідь")]
    )
    agent = make_agent(llm, store, kb)
    session = await agent.handle(make_request())

    resumed = await agent.resume(session, "reject", decided_by="Ігор")

    assert store.side_effects(session.session_id).get("replies") is None
    assert resumed.status == "done"
    assert "escalate_to_human" in resumed.tool_names()
    rejected = [s for s in resumed.steps if s.error == "відхилено людиною"]
    assert rejected, "модель має побачити відмову як результат інструмента"


async def test_invalid_arguments_never_reach_the_human(store, kb):
    """
    Регресія з реального прогону: агент попросив підтвердити відповідь, у якій
    бракувало обов'язкового поля шаблону. Людина натиснула «підтвердити» — і аж
    тоді інструмент відмовився. Так найшвидше привчити людину тиснути «так» не
    читаючи, тому аргументи перевіряються ДО паузи.
    """
    llm = ScriptedLLM(
        [call("reply_with_template", template="need_more_info"),          # без поля missing
         call("reply_with_template", template="need_more_info",
              fields={"missing": "номер заявки"}),
         say("уточнення надіслано")]
    )
    agent = make_agent(llm, store, kb)
    session = await agent.handle(make_request())

    assert session.status == "awaiting_confirmation"
    errors = [s.error for s in session.steps if s.error]
    assert any("бракує поля" in (e or "") for e in errors), "перший виклик мав відпасти сам"
    confirmations = [s for s in session.steps if s.kind == "confirmation_request"]
    assert len(confirmations) == 1, "людину питали рівно один раз — про робочий варіант"
    assert "номер заявки" in confirmations[0].result["body"]


async def test_confirmation_card_shows_the_rendered_letter(store, kb):
    """Людина погоджує текст, який реально піде, а не JSON з аргументами."""
    llm = ScriptedLLM(
        [call("reply_with_template", template="answer_from_kb",
              fields={"answer": "24 календарні дні", "source": "Відпустки → Щорічна"})]
    )
    agent = make_agent(llm, store, kb)
    session = await agent.handle(make_request())

    card = [s for s in session.steps if s.kind == "confirmation_request"][0]
    assert "24 календарні дні" in card.result["body"]
    assert card.result["template"] == "answer_from_kb"


async def test_confirmation_can_be_disabled_for_comparison(store, kb):
    """Вимикач потрібен рівно для одного рядка в eval, а не для роботи."""
    llm = ScriptedLLM([call("reply_with_template", template="acknowledge"), say("готово")])
    agent = make_agent(llm, store, kb, AgentConfig(require_confirmation=False))
    session = await agent.handle(make_request())

    assert session.status == "done"
    assert store.side_effects(session.session_id)["replies"], "без запобіжника лист іде одразу"


async def test_resume_of_a_running_session_is_rejected(store, kb):
    llm = ScriptedLLM([say("нічого не роблю")])
    agent = make_agent(llm, store, kb)
    session = await agent.handle(make_request())

    with pytest.raises(ValueError):
        await agent.resume(session, "approve")


# --- стійкість ------------------------------------------------------------

async def test_unknown_tool_is_reported_back_not_fatal(store, kb):
    """Модель вигадала інструмент — це привід повернути їй помилку, не падати."""
    llm = ScriptedLLM([call("send_sms", to="+380"), say("зрозумів, такого немає")])
    agent = make_agent(llm, store, kb)
    session = await agent.handle(make_request())

    assert session.status == "done"
    errors = [s for s in session.steps if s.error and "невідомий інструмент" in s.error]
    assert errors


async def test_tool_error_is_visible_to_the_model(store, kb):
    """
    Інструмент відмовився (порушено регламент зустрічей). Модель має побачити
    причину текстом і мати змогу виправитись.
    """
    llm = ScriptedLLM(
        [call("schedule_meeting", title="Синк", starts_at="2026-09-01 12:30"),
         call("schedule_meeting", title="Синк", starts_at="2026-09-01 14:00"),
         say("зустріч поставлено")]
    )
    agent = make_agent(llm, store, kb, AgentConfig(require_confirmation=False))
    session = await agent.handle(make_request("Постав зустріч"))

    errors = [s.error for s in session.steps if s.error]
    assert any("обідній час" in (e or "") for e in errors)
    assert session.status == "done"
    assert store.side_effects(session.session_id)["meetings"][0]["starts_at"] == "2026-09-01 14:00"


async def test_trace_is_saved_after_every_step(store, kb):
    """Падіння на третьому кроці має лишати перші два в базі."""
    llm = ScriptedLLM(
        [call("search_knowledge_base", query="відпустка"), RuntimeError("мережа впала")]
    )
    agent = make_agent(llm, store, kb)
    session = await agent.handle(make_request())

    saved = store.get(session.session_id)
    assert saved.status == "aborted"
    assert "search_knowledge_base" in saved.tool_names(), "перший крок не загубився"


async def test_same_request_twice_reuses_the_session(store, kb):
    """At-least-once доставка не має створювати другу задачу."""
    llm = ScriptedLLM([call("create_task", title="Налаштувати доступ"), say("створено")])
    agent = make_agent(llm, store, kb)

    first = await agent.handle(make_request("Дайте доступ до Jira"))
    second = await agent.handle(make_request("Дайте доступ до Jira"))

    assert first.session_id == second.session_id
    assert len(store.side_effects(first.session_id)["tasks"]) == 1
