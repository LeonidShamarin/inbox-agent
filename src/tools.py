"""
Інструменти агента та їхній реєстр.

**Що вимагає підтвердження людини і чому.** Спокуса позначити
`requires_confirmation` на всьому, що щось змінює, — і саме так human-in-the-loop
перетворюється на штамп: коли підтвердження просять двадцять разів на день,
людина тисне «підтвердити» не читаючи, і запобіжник перестає працювати саме
тоді, коли він потрібен.

Тому межа проведена не по «читає / пише», а по **зворотності й видимості**:

| Інструмент | Підтвердження | Чому |
|---|---|---|
| `search_knowledge_base` | ні | лише читає |
| `create_task` | ні | внутрішня задача, видаляється одним рухом |
| `reply_with_template` | **так** | текст іде людині за межі команди, назад не забереш |
| `schedule_meeting` | **так** | займає чужий час у чужих календарях |
| `escalate_to_human` | ні | це і є передача людині, безпечний кінець |

Помилитись у бік «зайве підтвердження» дешево лише на папері; на практиці це
вимикає увагу. Помилитись у бік «виконали без питання» дорого завжди.

Кожен обробник повертає словник — його бачить і модель (як результат виклику),
і людина в трасі. Виняток усередині обробника не валить сесію: цикл ловить його
й записує як крок з помилкою, щоб агент міг спробувати інший шлях.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

from src.retrieval import KnowledgeBase
from src.store import AgentStore

# Шаблони відповідей. Текст листа не вигадується моделлю з нуля: вона обирає
# шаблон і заповнює поля. Так відповідь від компанії лишається передбачуваною,
# а перевіряти людині треба кілька значень, а не весь текст.
TEMPLATES: dict[str, str] = {
    "acknowledge": (
        "Доброго дня, {name}!\n\n"
        "Отримали ваш запит, взяли в роботу. Повернемось із відповіддю до {due}.\n\n"
        "З повагою,\nкоманда підтримки"
    ),
    "answer_from_kb": (
        "Доброго дня, {name}!\n\n"
        "{answer}\n\n"
        "Джерело: {source}\n\n"
        "Якщо потрібні деталі — напишіть, уточнимо.\n\n"
        "З повагою,\nкоманда підтримки"
    ),
    "need_more_info": (
        "Доброго дня, {name}!\n\n"
        "Щоб узяти запит у роботу, бракує деталей: {missing}.\n\n"
        "Підкажіть, будь ласка, і ми одразу продовжимо.\n\n"
        "З повагою,\nкоманда підтримки"
    ),
    "decline": (
        "Доброго дня, {name}!\n\n"
        "На жаль, цей запит ми не можемо виконати: {reason}.\n\n"
        "З повагою,\nкоманда підтримки"
    ),
}

PRIORITIES = ("low", "medium", "high", "critical")


class ToolError(Exception):
    """Помилка виконання інструмента, зрозуміла моделі (не збій пайплайна)."""


@dataclass
class ToolContext:
    """Усе, що обробник має право чіпати. Прямого доступу до сесії немає навмисно."""

    store: AgentStore
    kb: KnowledgeBase
    session_id: str
    sender: str = "невідомо"
    today: Optional[date] = None

    def now(self) -> date:
        return self.today or date.today()


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[ToolContext, dict], dict]
    # Перевірка аргументів без побічних ефектів. Викликається ПЕРЕД паузою на
    # підтвердження: людину не можна питати про дію, яка все одно впаде.
    # Те, що повертає preview, показується в картці підтвердження.
    preview: Optional[Callable[[ToolContext, dict], dict]] = None
    requires_confirmation: bool = False
    # Термінальний інструмент завершує сесію: після нього агентові немає що
    # робити, і давати моделі ще один хід означало б дозволити їй «передумати»
    # вже після ескалації.
    terminal: bool = False
    # Людям у черзі показується саме це, а не назва функції.
    confirmation_title: str = ""
    examples: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Обробники
# --------------------------------------------------------------------------

def _search_kb(ctx: ToolContext, args: dict) -> dict:
    query = str(args.get("query", "")).strip()
    if not query:
        raise ToolError("порожній запит до бази знань")
    hits = ctx.kb.search(query, top_k=int(args.get("top_k", 3)))
    if not hits:
        # Порожній результат — це відповідь, а не збій. Саме він має підштовхнути
        # агента до ескалації замість вигадування.
        return {
            "found": False,
            "query": query,
            "hint": "У базі знань нічого не знайдено. Не вигадуй відповідь — "
            "передай запит людині через escalate_to_human.",
        }
    return {"found": True, "query": query, "results": hits}


def _create_task(ctx: ToolContext, args: dict) -> dict:
    title = str(args.get("title", "")).strip()
    if not title:
        raise ToolError("задача без назви")
    priority = str(args.get("priority", "medium")).lower()
    if priority not in PRIORITIES:
        raise ToolError(f"невідомий пріоритет {priority!r}, дозволені: {', '.join(PRIORITIES)}")

    due = args.get("due_date")
    if due:
        try:
            parsed = datetime.strptime(str(due), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ToolError(f"дата у форматі YYYY-MM-DD, отримано {due!r}") from exc
        if parsed < ctx.now():
            raise ToolError(f"дедлайн {due} у минулому, сьогодні {ctx.now().isoformat()}")

    payload = {
        "title": title,
        "assignee": args.get("assignee"),
        "priority": priority,
        "due_date": due,
        "description": args.get("description"),
    }
    task_id = ctx.store.add_task(ctx.session_id, payload)
    return {"created": True, "task_id": task_id, **payload}


def _reply_preview(ctx: ToolContext, args: dict) -> dict:
    """
    Перевіряє аргументи й збирає текст листа, НЕ записуючи нічого.

    Існує окремо від `_reply` через реальний випадок з eval: агент попросив
    підтвердження на відповідь, у якій бракувало обов'язкового поля шаблону.
    Людина натиснула «підтвердити» — і аж тоді інструмент відмовився. Тобто
    її потурбували заради дії, яка не могла виконатись у принципі.

    Тепер усе, що можна перевірити без побічних ефектів, перевіряється до
    паузи. Побічний бонус: у картці підтвердження видно готовий текст листа,
    а не сирі аргументи — людина погоджує те, що реально піде.
    """
    template = str(args.get("template", "")).strip()
    if template not in TEMPLATES:
        raise ToolError(
            f"невідомий шаблон {template!r}, доступні: {', '.join(sorted(TEMPLATES))}"
        )
    fields = dict(args.get("fields") or {})
    fields.setdefault("name", ctx.sender)
    fields.setdefault("due", (ctx.now() + timedelta(days=1)).isoformat())
    try:
        body = TEMPLATES[template].format(**fields)
    except KeyError as exc:
        raise ToolError(
            f"для шаблону {template!r} бракує поля {exc.args[0]!r} у fields"
        ) from exc
    return {
        "recipient": args.get("recipient") or ctx.sender,
        "template": template,
        "body": body,
    }


def _reply(ctx: ToolContext, args: dict) -> dict:
    prepared = _reply_preview(ctx, args)
    reply_id = ctx.store.add_reply(
        ctx.session_id, {**args, "recipient": prepared["recipient"]}, prepared["body"]
    )
    return {"sent": True, "reply_id": reply_id, **prepared}


def _schedule_preview(ctx: ToolContext, args: dict) -> dict:
    """Ті самі перевірки, що й у `_schedule`, але без запису в календар."""
    title = str(args.get("title", "")).strip()
    if not title:
        raise ToolError("зустріч без теми")
    starts_at = str(args.get("starts_at", "")).strip()
    try:
        moment = datetime.strptime(starts_at, "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ToolError(f"час у форматі 'YYYY-MM-DD HH:MM', отримано {starts_at!r}") from exc

    # Правила з регламенту перевіряє КОД, а не модель: зустрічі 10:00–17:00,
    # обід 12:00–13:00 вільний. Модель уміє переказати регламент і тут же його
    # порушити, бо для неї це текст, а не обмеження.
    if not (10 <= moment.hour < 17):
        raise ToolError(
            f"{moment:%H:%M} поза робочим вікном зустрічей 10:00–17:00 (див. регламент комунікації)"
        )
    if moment.hour == 12:
        raise ToolError("12:00–13:00 — обідній час, зустрічі не призначаються")

    duration = int(args.get("duration_min", 30))
    if duration <= 0 or duration > 180:
        raise ToolError("тривалість має бути від 1 до 180 хвилин")

    participants = args.get("participants") or [ctx.sender]
    return {
        "title": title,
        "participants": participants,
        "starts_at": starts_at,
        "duration_min": duration,
    }


def _schedule(ctx: ToolContext, args: dict) -> dict:
    prepared = _schedule_preview(ctx, args)
    meeting_id = ctx.store.add_meeting(ctx.session_id, prepared)
    return {"scheduled": True, "meeting_id": meeting_id, **prepared}


def _escalate(ctx: ToolContext, args: dict) -> dict:
    reason = str(args.get("reason", "")).strip()
    if not reason:
        raise ToolError("ескалація без причини — людині нема з чим працювати")
    return {
        "escalated": True,
        "reason": reason,
        "to": args.get("to") or "черговий менеджер",
        "summary": args.get("summary"),
    }


# --------------------------------------------------------------------------
# Реєстр
# --------------------------------------------------------------------------

def build_registry() -> dict[str, ToolSpec]:
    specs = [
        ToolSpec(
            name="search_knowledge_base",
            description=(
                "Шукає відповідь у внутрішніх регламентах компанії (відпустки, доступи, "
                "закупівлі, IT, зустрічі). Викликай ПЕРШИМ, якщо запит схожий на питання "
                "про правила. Повертає фрагменти з посиланням на розділ."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Пошуковий запит українською"},
                    "top_k": {"type": "integer", "description": "Скільки фрагментів, 1–5"},
                },
                "required": ["query"],
            },
            handler=_search_kb,
        ),
        ToolSpec(
            name="create_task",
            description=(
                "Створює внутрішню задачу в трекері. Використовуй, коли запит вимагає "
                "роботи від команди, а не просто відповіді."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Коротка назва задачі"},
                    "assignee": {"type": "string", "description": "Команда або людина"},
                    "priority": {
                        "type": "string",
                        "enum": list(PRIORITIES),
                        "description": "Пріоритет",
                    },
                    "due_date": {"type": "string", "description": "Дедлайн, YYYY-MM-DD"},
                    "description": {"type": "string", "description": "Суть задачі"},
                },
                "required": ["title"],
            },
            handler=_create_task,
        ),
        ToolSpec(
            name="reply_with_template",
            description=(
                "Надсилає відповідь автору запиту за одним із шаблонів: "
                "acknowledge (взяли в роботу), answer_from_kb (відповідь із регламенту, "
                "обов'язково з джерелом), need_more_info (бракує деталей), decline (відмова). "
                "Текст іде людині — виклик потребує підтвердження."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "enum": sorted(TEMPLATES),
                        "description": "Який шаблон використати",
                    },
                    "recipient": {"type": "string", "description": "Кому"},
                    "fields": {
                        "type": "object",
                        "description": "Поля шаблону: name, answer, source, missing, reason, due",
                        "properties": {
                            "name": {"type": "string"},
                            "answer": {"type": "string"},
                            "source": {"type": "string"},
                            "missing": {"type": "string"},
                            "reason": {"type": "string"},
                            "due": {"type": "string"},
                        },
                    },
                },
                "required": ["template"],
            },
            handler=_reply,
            preview=_reply_preview,
            requires_confirmation=True,
            confirmation_title="Надіслати відповідь автору запиту",
        ),
        ToolSpec(
            name="schedule_meeting",
            description=(
                "Ставить зустріч у календарі учасників. Робоче вікно 10:00–17:00, "
                "обід 12:00–13:00 зайнятий. Займає чужий час — потребує підтвердження."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Тема зустрічі"},
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Учасники",
                    },
                    "starts_at": {"type": "string", "description": "Початок, 'YYYY-MM-DD HH:MM'"},
                    "duration_min": {"type": "integer", "description": "Тривалість у хвилинах"},
                },
                "required": ["title", "starts_at"],
            },
            handler=_schedule,
            preview=_schedule_preview,
            requires_confirmation=True,
            confirmation_title="Поставити зустріч у календарі",
        ),
        ToolSpec(
            name="escalate_to_human",
            description=(
                "Передає запит людині. Викликай, коли в базі знань відповіді немає, "
                "запит неоднозначний або стосується грошей, безпеки чи звільнення. "
                "Краще передати людині, ніж вигадати відповідь."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Чому потрібна людина"},
                    "to": {"type": "string", "description": "Кому передати"},
                    "summary": {"type": "string", "description": "Стисло суть запиту"},
                },
                "required": ["reason"],
            },
            handler=_escalate,
            terminal=True,
        ),
    ]
    return {spec.name: spec for spec in specs}


IRREVERSIBLE = tuple(
    name for name, spec in build_registry().items() if spec.requires_confirmation
)
