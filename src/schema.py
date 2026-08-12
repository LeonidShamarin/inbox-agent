"""
Контракт даних агента.

Центральне поняття — **сесія**: один вхідний запит і повний слід того, що з ним
робив агент. Сесія серіалізується цілком і зберігається після кожного кроку,
тому вона одночасно:

* стан виконання (де зупинились, скільки бюджету лишилось);
* журнал для людини (`plan → tool → args → result → next`);
* точка відновлення після підтвердження людини або після падіння процесу.

Саме тому крок — це запис у списку, а не рядок у логах. Лог можна загубити,
у нього не можна повернутись і за ним не видно, чому агент вирішив саме так.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Channel = Literal["telegram", "email", "webhook", "web"]

# running                — агент працює або готовий до наступного кроку
# awaiting_confirmation  — уперся в незворотну дію і ЧЕКАЄ рішення людини
# done                   — дійшов до відповіді
# aborted                — вичерпав бюджет, впав або людина відхилила дію
SessionStatus = Literal["running", "awaiting_confirmation", "done", "aborted"]

StepKind = Literal[
    "plan",                  # модель пояснила, що збирається робити
    "tool_call",             # виклик інструмента з аргументами
    "tool_result",           # що інструмент повернув
    "confirmation_request",  # пауза: потрібна згода людини
    "human_decision",        # що людина відповіла
    "final",                 # підсумкова відповідь агента
    "abort",                 # запобіжник спрацював
]

Decision = Literal["approve", "reject"]


class InboundRequest(BaseModel):
    """Вхідний запит. Канал не впливає на логіку — лише на те, звідки він прийшов."""

    request_id: str
    channel: Channel = "web"
    sender: str = "невідомо"
    text: str
    received_at: str


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    """
    Один запис траси. «next» у формулі `plan → tool → args → result → next` —
    це наступний елемент списку: тримати явне посилання означало б мати два
    джерела правди про порядок.
    """

    index: int
    kind: StepKind
    at: str

    # Чому агент так вирішив. Для kind="plan" — його власне формулювання,
    # для решти — пояснення від пайплайна (напр. причина зупинки).
    note: Optional[str] = None

    tool: Optional[ToolCall] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    # Непрозорий підпис міркування від Gemini 3, у base64.
    #
    # Він потрібен не нам, а API: коли виклик функції повертається назад в
    # історії, підпис має бути при ньому, інакше наступний хід падає з
    # 400 INVALID_ARGUMENT «Function call is missing a thought_signature».
    # Саме тому історія відновлюється з траси разом із цим полем: без нього
    # пауза на підтвердженні ламала б розмову, яку відновлюють після неї.
    signature: Optional[str] = None

    decision: Optional[Decision] = None
    decided_by: Optional[str] = None

    tokens_in: int = 0
    tokens_out: int = 0
    duration_s: float = 0.0

    def summary(self) -> str:
        """Один рядок для CLI-виводу траси."""
        if self.tool is not None:
            args = ", ".join(f"{k}={v!r}" for k, v in list(self.tool.args.items())[:3])
            return f"{self.kind}: {self.tool.name}({args})"
        if self.decision:
            return f"{self.kind}: {self.decision}"
        return f"{self.kind}: {(self.note or '')[:80]}"


class BudgetState(BaseModel):
    """
    Скільки витрачено. Ліміти лежать у конфігу, витрати — тут, у сесії:
    після перезапуску процесу бюджет не має починатися з нуля, інакше
    «ліміт на задачу» перетворюється на «ліміт на спробу».
    """

    steps: int = 0
    tokens: int = 0
    elapsed_s: float = 0.0
    tool_calls: int = 0


class Session(BaseModel):
    session_id: str
    request: InboundRequest
    status: SessionStatus = "running"
    steps: list[Step] = Field(default_factory=list)
    budget: BudgetState = Field(default_factory=BudgetState)

    created_at: str
    updated_at: str

    # Заповнюється, коли status="awaiting_confirmation": що саме агент хоче
    # зробити. Людина бачить точні аргументи, а не переказ намірів.
    pending_tool: Optional[ToolCall] = None
    pending_reason: Optional[str] = None

    final_answer: Optional[str] = None
    abort_reason: Optional[str] = None

    def add(self, step: Step) -> Step:
        step.index = len(self.steps)
        self.steps.append(step)
        return step

    def tool_calls(self) -> list[ToolCall]:
        return [s.tool for s in self.steps if s.kind == "tool_call" and s.tool is not None]

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tool_calls()]

    def to_trace_dict(self) -> dict:
        """Компактний JSON-слід для `output/` і для README."""
        return {
            "session_id": self.session_id,
            "request": self.request.model_dump(),
            "status": self.status,
            "budget": self.budget.model_dump(),
            "final_answer": self.final_answer,
            "abort_reason": self.abort_reason,
            "steps": [
                {
                    "index": s.index,
                    "kind": s.kind,
                    "note": s.note,
                    "tool": s.tool.model_dump() if s.tool else None,
                    "result": s.result,
                    "error": s.error,
                    "decision": s.decision,
                    "tokens": {"in": s.tokens_in, "out": s.tokens_out},
                    "duration_s": round(s.duration_s, 3),
                }
                for s in self.steps
            ],
        }
