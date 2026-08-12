"""
Нативний tool calling через `google-genai`, без фреймворку.

Чому без фреймворку. Весь агент — це цикл «спитати модель → виконати те, що
вона попросила → віддати результат назад». Фреймворк ховає саме цей цикл, а
разом із ним і місця, де треба ставити запобіжники: рахувати кроки, рахувати
токени, зупинятися перед незворотною дією. Тут цикл видно (`agent.py`), і
кожен запобіжник стоїть там, де він потрібен.

**Автовиклик функцій вимкнено явно.** SDK вміє сам виконувати інструменти й
повертати готову відповідь — і це рівно те, чого агентові з human-in-the-loop
робити не можна: він виконав би `reply_with_template` не спитавши нікого.

Історія розмови тримається в нейтральних `Turn`, а не в типах SDK: завдяки
цьому в тестах на місце клієнта стає фейк, і жоден тест не потребує ні ключа,
ні мережі.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.schema import ToolCall
from src.tools import ToolSpec

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_RPM = 5

TRANSPORT_MAX_ATTEMPTS = 4
TRANSPORT_BASE_DELAY = 2.0
MAX_SERVER_RETRY_DELAY = 65.0

_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")

SYSTEM_PROMPT = """\
Ти — асистент вхідного потоку запитів у компанії. До тебе потрапляють запити \
від співробітників і зовнішніх людей: питання про регламенти, прохання щось \
зробити, скарги, неоднозначні повідомлення.

Твоя робота — не переказати запит, а **довести його до результату** наявними \
інструментами.

Порядок дій:

1. Якщо це питання про правила компанії — спершу `search_knowledge_base`. \
Не відповідай із власних знань: правила в кожній компанії свої.
2. Якщо знайшов відповідь — надішли її через `reply_with_template` із шаблоном \
`answer_from_kb`, обов'язково вказавши джерело з результату пошуку.
3. Якщо запит вимагає роботи команди — `create_task` із конкретною назвою.
4. Якщо просять зустріч — `schedule_meeting`.
5. Якщо в базі знань відповіді немає або запит неоднозначний — \
`escalate_to_human`. Це не поразка, а правильний хід.
6. Запити про гроші, безпеку і звільнення теж ідуть людині — але **спершу** \
перевір регламент: якщо там є чітка інструкція, що робити просто зараз \
(наприклад, куди писати при компрометації пароля), надішли її, і лише потім \
ескалюй. Людина, яка чекає на відповідь із заблокованим акаунтом, від самої \
ескалації нічого не отримує.

Правила:

- **Не вигадуй фактів про компанію.** Якщо пошук нічого не дав — ескалація.
- Не викликай той самий інструмент двічі з тими самими аргументами.
- Коли задача завершена, поверни коротке текстове резюме того, що зроблено, \
без виклику інструментів.
- Відповідай українською.
"""


class LLMError(Exception):
    """Вичерпані всі транспортні спроби."""


class DailyQuotaExceeded(Exception):
    """Вичерпано денну квоту — до кінця доби прогін не відновиться."""


@dataclass
class Turn:
    """Один хід розмови в нейтральному вигляді, без типів SDK."""

    role: Literal["user", "model", "tool"]
    text: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_name: Optional[str] = None
    tool_result: Optional[dict[str, Any]] = None
    # base64 підпису міркування; потрібен лише для ролі "model" з викликом.
    signature: Optional[str] = None


@dataclass
class ModelDecision:
    """
    Що модель вирішила на цьому кроці: або викликати інструмент, або відповісти.

    Обидва поля порожні — теж валідний стан (модель промовчала); цикл трактує
    його як завершення, а не як привід ходити далі.
    """

    tool: Optional[ToolCall] = None
    text: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    raw_parts: list[str] = field(default_factory=list)
    # Підпис міркування, який API вимагає повернути разом із викликом.
    signature: Optional[str] = None


class AgentLLM(Protocol):
    """Контракт, від якого залежить цикл агента. У тестах — фейк."""

    async def decide(self, history: list[Turn], tools: list[ToolSpec]) -> ModelDecision:
        ...


def _server_retry_delay(exc: Exception) -> Optional[float]:
    match = _RETRY_DELAY_RE.search(str(exc))
    return min(float(match.group(1)), MAX_SERVER_RETRY_DELAY) if match else None


def _is_daily_quota(exc: Exception) -> bool:
    """
    Не всі 429 однакові: хвилинну квоту перечекати можна, денну — ні. Gemini
    повертає `retryDelay` у ВСІХ 429, тому дивимось саме на quotaId.
    """
    return "PerDay" in str(exc)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        if getattr(exc, "code", None) != 429:
            return False
        return not _is_daily_quota(exc)
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))


class _RateLimiter:
    """Рознесення викликів у часі під квоту безкоштовного тіру (5 запитів/хв)."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = now + self._interval


class GeminiAgentLLM:
    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        rpm: int = DEFAULT_RPM,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._limiter = _RateLimiter(rpm)
        self.daily_quota_hit = False

    # -- конвертація ------------------------------------------------------

    @staticmethod
    def _to_contents(history: list[Turn]) -> list[types.Content]:
        contents: list[types.Content] = []
        for turn in history:
            if turn.role == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=turn.text or "")])
                )
            elif turn.role == "model":
                if turn.tool_call is not None:
                    # thought_signature повертається разом із викликом. Gemini 3
                    # без нього відповідає 400 INVALID_ARGUMENT на НАСТУПНОМУ
                    # ході: перший виклик проходить, а розмова далі не їде.
                    part = types.Part(
                        function_call=types.FunctionCall(
                            name=turn.tool_call.name, args=turn.tool_call.args
                        )
                    )
                    if turn.signature:
                        part.thought_signature = base64.b64decode(turn.signature)
                    contents.append(types.Content(role="model", parts=[part]))
                else:
                    contents.append(
                        types.Content(role="model", parts=[types.Part(text=turn.text or "")])
                    )
            else:
                # Результат інструмента повертається роллю "user": для моделі це
                # вхідні дані, а не її власний хід.
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=turn.tool_name or "unknown",
                                response=turn.tool_result or {},
                            )
                        ],
                    )
                )
        return contents

    @staticmethod
    def _to_declarations(tools: list[ToolSpec]) -> list[types.Tool]:
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=spec.name,
                        description=spec.description,
                        parameters=spec.parameters,
                    )
                    for spec in tools
                ]
            )
        ]

    # -- один хід ---------------------------------------------------------

    async def decide(self, history: list[Turn], tools: list[ToolSpec]) -> ModelDecision:
        config = types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            temperature=self._temperature,
            tools=self._to_declarations(tools),
            # Ключовий рядок: SDK не має права виконувати інструменти сам.
            # Інакше він виконав би незворотну дію, не спитавши людину.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )
        contents = self._to_contents(history)
        last_error: Optional[Exception] = None

        for attempt in range(TRANSPORT_MAX_ATTEMPTS):
            if self.daily_quota_hit:
                raise DailyQuotaExceeded(f"денна квота вичерпана для {self._model_name}")
            try:
                await self._limiter.acquire()
                started = time.monotonic()
                response = await self._client.aio.models.generate_content(
                    model=self._model_name, contents=contents, config=config
                )
                elapsed = time.monotonic() - started
                return self._parse(response, elapsed)
            except Exception as exc:  # noqa: BLE001 — тип розбирається нижче
                if _is_daily_quota(exc):
                    self.daily_quota_hit = True
                    raise DailyQuotaExceeded(
                        f"денна квота вичерпана для {self._model_name}"
                    ) from exc
                if not _is_retryable(exc):
                    raise
                last_error = exc
                if attempt == TRANSPORT_MAX_ATTEMPTS - 1:
                    break
                delay = _server_retry_delay(exc) or TRANSPORT_BASE_DELAY * (2**attempt)
                delay += random.uniform(0, 1)
                logger.warning(
                    "transport error (%d/%d), retry in %.1fs: %s",
                    attempt + 1, TRANSPORT_MAX_ATTEMPTS, delay, str(exc)[:200],
                )
                await asyncio.sleep(delay)

        raise LLMError(f"транспорт не витримав {TRANSPORT_MAX_ATTEMPTS} спроб: {last_error}")

    @staticmethod
    def _parse(response: Any, elapsed: float) -> ModelDecision:
        usage = getattr(response, "usage_metadata", None)
        decision = ModelDecision(
            tokens_in=getattr(usage, "prompt_token_count", 0) or 0,
            tokens_out=getattr(usage, "candidates_token_count", 0) or 0,
            latency_s=elapsed,
        )

        candidates = getattr(response, "candidates", None) or []
        parts = []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []

        texts: list[str] = []
        for part in parts:
            call = getattr(part, "function_call", None)
            if call is not None and getattr(call, "name", None):
                # Беремо ПЕРШИЙ виклик і ігноруємо решту: паралельні виклики
                # ламають і бюджет по кроках, і підтвердження — людина мала б
                # погоджувати їх пачкою, не бачачи, який від чого залежить.
                if decision.tool is None:
                    decision.tool = ToolCall(name=call.name, args=dict(call.args or {}))
                    raw_signature = getattr(part, "thought_signature", None)
                    if raw_signature:
                        decision.signature = base64.b64encode(raw_signature).decode("ascii")
                continue
            text = getattr(part, "text", None)
            if text:
                texts.append(text)

        decision.raw_parts = texts
        if texts:
            decision.text = "\n".join(texts).strip()
        return decision
