"""Інструменти, пошук по базі знань і сховище."""

from __future__ import annotations

from datetime import date

import pytest

from src.retrieval import BM25Index, KnowledgeBase, expand_query, normalize, split_markdown, stem
from src.store import AgentStore, request_hash
from src.tools import TEMPLATES, ToolContext, ToolError, build_registry
from tests.conftest import make_request

TODAY = date(2026, 8, 12)


@pytest.fixture
def ctx(store, kb) -> ToolContext:
    return ToolContext(store=store, kb=kb, session_id="s-test", sender="Олена", today=TODAY)


@pytest.fixture
def registry():
    return build_registry()


# --- межа підтвердження ---------------------------------------------------

def test_only_outward_actions_require_confirmation(registry):
    """
    Межа проведена по зворотності й видимості, а не по «читає/пише». Якщо
    підтвердження просити на всьому, людина тиснутиме «так» не читаючи.
    """
    needs = {name for name, spec in registry.items() if spec.requires_confirmation}
    assert needs == {"reply_with_template", "schedule_meeting"}


def test_escalation_is_terminal(registry):
    assert registry["escalate_to_human"].terminal
    assert not registry["create_task"].terminal


def test_every_tool_declares_a_schema(registry):
    for name, spec in registry.items():
        assert spec.parameters["type"] == "object", name
        assert spec.parameters["properties"], name
        assert spec.description.strip(), name


# --- пошук ----------------------------------------------------------------

def test_stemmer_collapses_word_forms():
    """Словоформи одного слова мають злитися — без цього BM25 їх не зв'яже."""
    assert stem("відпустки") == stem("відпустку") == stem("відпустка") == "відпустк"
    assert stem("погодження") != stem("погодити"), "стемер не має злипати різні слова"


def test_short_roots_are_left_alone_deliberately():
    """
    Корені, коротші за MIN_STEM, не чіпаються: «днів» і «дні» так і лишаються
    різними токенами. Це свідома межа проти злипання коротких слів, і вона
    заміряна — на eval-наборі стемер усе одно дає hit@3 = 100% проти 92% без
    нього (`main.py eval --retrieval`).
    """
    assert stem("днів") == "днів" and stem("дні") == "дні"
    assert stem("рік") == "рік"
    assert normalize("2026 рік") == ["2026", "рік"]


def test_bm25_ranks_the_relevant_chunk_first():
    index = BM25Index(["відпустка 24 календарні дні", "ліміти погодження закупівель", "IT-підтримка"])
    hits = index.search("скільки днів відпустки", top_k=1)
    assert hits and hits[0][0] == 0


def test_structural_chunking_keeps_sections_whole(tmp_path):
    path = tmp_path / "reglament.md"
    path.write_text(
        "# Регламент\n\n## Розділ А\nтекст А\n\n## Розділ Б\nтекст Б\n", encoding="utf-8"
    )
    chunks = split_markdown(path.read_text(encoding="utf-8"), path)
    sections = [c.section for c in chunks]
    assert "Розділ А" in sections and "Розділ Б" in sections
    a = next(c for c in chunks if c.section == "Розділ А")
    assert a.text.strip() == "текст А" and a.doc_title == "Регламент"


def test_synonyms_close_the_vocabulary_gap(kb):
    """
    «Зламали акаунт» проти «компрометація облікового запису» — нуль спільних
    токенів. Це межа лексичного пошуку, латана словником синонімів.
    """
    assert "обліковий запис" in expand_query("зламали мій акаунт")
    hits = kb.search("зламали мій акаунт", top_k=1)
    assert hits and "пароль" in hits[0]["citation"].lower()


def test_search_returns_citation_and_score(kb):
    hits = kb.search("хто погоджує закупівлю на 1500 EUR", top_k=2)
    assert hits[0]["citation"] == "Закупівлі та витрати → Ліміти погодження"
    assert hits[0]["score"] > 0 and hits[0]["text"]


def test_relevance_threshold_rejects_a_formal_match(kb):
    """
    BM25 майже завжди повертає «щось»: слово «хто» є в половині регламентів,
    і «хто президент Марса» формально влучає. Поріг відсікає такі збіги —
    інакше агент побудує лист на випадковому розділі.
    """
    assert kb.search("хто президент Марса") == []
    assert kb.search("як приготувати борщ") == []
    assert kb.search("скільки днів відпустки на рік"), "справжнє питання поріг переживає"


def test_empty_result_pushes_the_agent_to_escalate(ctx, registry):
    result = registry["search_knowledge_base"].handler(ctx, {"query": "хто президент Марса"})
    assert result["found"] is False
    assert "escalate_to_human" in result["hint"], "порожній пошук має вести до людини"


# --- create_task ----------------------------------------------------------

def test_create_task_writes_a_row(ctx, registry, store):
    result = registry["create_task"].handler(
        ctx, {"title": "Видати доступ до Jira", "priority": "high", "assignee": "IT"}
    )
    assert result["created"] and result["task_id"]
    assert store.side_effects("s-test")["tasks"][0]["title"] == "Видати доступ до Jira"


def test_create_task_rejects_a_deadline_in_the_past(ctx, registry):
    with pytest.raises(ToolError, match="минулому"):
        registry["create_task"].handler(ctx, {"title": "Задача", "due_date": "2026-01-01"})


def test_create_task_rejects_unknown_priority(ctx, registry):
    with pytest.raises(ToolError, match="пріоритет"):
        registry["create_task"].handler(ctx, {"title": "Задача", "priority": "терміново!!!"})


# --- reply_with_template --------------------------------------------------

def test_reply_renders_the_template(ctx, registry):
    result = registry["reply_with_template"].handler(
        ctx, {"template": "answer_from_kb",
              "fields": {"answer": "24 календарні дні", "source": "Відпустки → Щорічна"}}
    )
    assert "24 календарні дні" in result["body"]
    assert "Відпустки → Щорічна" in result["body"], "джерело обов'язкове в тілі листа"
    assert result["body"].startswith("Доброго дня, Олена!")


def test_reply_reports_a_missing_field_instead_of_printing_a_placeholder(ctx, registry):
    with pytest.raises(ToolError, match="бракує поля"):
        registry["reply_with_template"].handler(ctx, {"template": "answer_from_kb", "fields": {}})


def test_reply_rejects_an_unknown_template(ctx, registry):
    with pytest.raises(ToolError, match="шаблон"):
        registry["reply_with_template"].handler(ctx, {"template": "friendly_hello"})


def test_all_templates_render_with_their_fields(ctx, registry):
    fields = {
        "acknowledge": {},
        "answer_from_kb": {"answer": "a", "source": "s"},
        "need_more_info": {"missing": "номер заявки"},
        "decline": {"reason": "поза скоупом"},
    }
    for name in TEMPLATES:
        result = registry["reply_with_template"].handler(
            ctx, {"template": name, "fields": fields[name]}
        )
        assert result["sent"] and "{" not in result["body"], name


# --- schedule_meeting -----------------------------------------------------

def test_meeting_outside_working_window_is_refused(ctx, registry):
    with pytest.raises(ToolError, match="10:00–17:00"):
        registry["schedule_meeting"].handler(
            ctx, {"title": "Ранній синк", "starts_at": "2026-09-01 08:00"}
        )


def test_meeting_at_lunch_is_refused(ctx, registry):
    """
    Правило з регламенту перевіряє код: модель уміє переказати регламент і тут
    же його порушити, бо для неї це текст, а не обмеження.
    """
    with pytest.raises(ToolError, match="обідній"):
        registry["schedule_meeting"].handler(
            ctx, {"title": "Обід із CFO", "starts_at": "2026-09-01 12:30"}
        )


def test_meeting_with_bad_time_format_is_refused(ctx, registry):
    with pytest.raises(ToolError, match="YYYY-MM-DD HH:MM"):
        registry["schedule_meeting"].handler(
            ctx, {"title": "Синк", "starts_at": "завтра о третій"}
        )


def test_valid_meeting_is_stored(ctx, registry, store):
    result = registry["schedule_meeting"].handler(
        ctx, {"title": "Демо", "starts_at": "2026-09-01 15:00", "duration_min": 45,
              "participants": ["Олена", "Ігор"]}
    )
    assert result["scheduled"]
    assert store.side_effects("s-test")["meetings"][0]["duration_min"] == 45


# --- escalate -------------------------------------------------------------

def test_escalation_without_a_reason_is_refused(ctx, registry):
    with pytest.raises(ToolError, match="причини"):
        registry["escalate_to_human"].handler(ctx, {})


# --- сховище --------------------------------------------------------------

def test_request_hash_ignores_time_but_not_content():
    a = make_request("Дайте доступ", request_id="1")
    b = make_request("Дайте доступ", request_id="2")
    c = make_request("Дайте інший доступ", request_id="3")
    assert request_hash(a) == request_hash(b), "той самий лист, доставлений двічі"
    assert request_hash(a) != request_hash(c)


def test_store_roundtrip_and_queue(tmp_path):
    store = AgentStore(tmp_path / "s.sqlite3")
    from src.schema import Session
    from src.store import utc_now

    session = Session(session_id="s-1", request=make_request(), created_at=utc_now(),
                      updated_at=utc_now(), status="awaiting_confirmation")
    store.save(session)

    assert store.get("s-1").status == "awaiting_confirmation"
    assert [s.session_id for s in store.awaiting()] == ["s-1"]
    assert store.counts_by_status() == {"awaiting_confirmation": 1}
    store.close()
