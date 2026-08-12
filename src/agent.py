"""
Цикл агента.

Увесь агент — це `while`: спитати модель → перевірити запобіжники → виконати те,
що вона попросила → віддати результат назад. Тут цей цикл написаний явно, бо
саме в ньому живуть три речі, які фреймворк ховає:

1. **Бюджети.** Перевіряються ПЕРЕД викликом моделі, а не після: сенс ліміту в
   тому, щоб не зробити зайвий виклик, а не в тому, щоб порахувати вже зроблені.
2. **Пауза на незворотній дії.** Агент не блокує потік і не чекає на людину в
   пам'яті: він зберігає стан і **виходить**. Рішення людини приходить окремим
   викликом `resume()`, можливо, наступного дня і точно в іншому процесі.
3. **Історія відновлюється з траси.** Розмова з моделлю не тримається в пам'яті
   об'єкта — вона щоразу збирається зі збережених кроків. Інакше пауза на
   підтвердженні працювала б лише доти, доки живий процес.

Кожен крок зберігається одразу після виконання. Падіння на третьому кроці
лишає перші два в базі — без цього неможливо зрозуміти, на чому агент зламався.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable, Optional

from src.config import AgentConfig
from src.llm import AgentLLM, DailyQuotaExceeded, ModelDecision, Turn
from src.retrieval import KnowledgeBase
from src.schema import Decision, InboundRequest, Session, Step, ToolCall
from src.store import AgentStore, utc_now
from src.tools import ToolContext, ToolError, ToolSpec, build_registry

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        llm: AgentLLM,
        store: AgentStore,
        kb: KnowledgeBase,
        cfg: Optional[AgentConfig] = None,
        registry: Optional[dict[str, ToolSpec]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.llm = llm
        self.store = store
        self.kb = kb
        self.cfg = cfg or AgentConfig()
        self.registry = registry if registry is not None else build_registry()
        self.clock = clock

    # -- запуск ------------------------------------------------------------

    async def handle(self, request: InboundRequest) -> Session:
        """
        Точка входу для нового запиту.

        Ідемпотентність: той самий запит, доставлений вебхуком двічі, не
        породжує другу сесію і другий створений тікет. At-least-once — це
        нормальна поведінка черг, а не рідкісний збій.
        """
        existing = self.store.find_by_request(request)
        if existing is not None:
            logger.info("запит уже оброблявся, сесія %s", existing.session_id)
            return existing

        session = Session(
            session_id=f"s-{uuid.uuid4().hex[:10]}",
            request=request,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        self.store.save(session)
        return await self.run(session)

    # -- головний цикл -----------------------------------------------------

    async def run(self, session: Session) -> Session:
        """Крутить цикл, доки не завершиться, не впреться в бюджет або не спитає людину."""
        started = self.clock()

        while True:
            spent = session.budget.elapsed_s + (self.clock() - started)
            reason = self._budget_exceeded(session, spent)
            if reason:
                session.budget.elapsed_s = spent
                return self._abort(session, reason)

            try:
                decision = await self.llm.decide(self._history(session), list(self.registry.values()))
            except DailyQuotaExceeded as exc:
                session.budget.elapsed_s = session.budget.elapsed_s + (self.clock() - started)
                return self._abort(session, f"денна квота вичерпана: {exc}")
            except (MemoryError, RecursionError):
                # Стан процесу, а не помилка задачі. Загорнути в abort означало б
                # дати циклу поїхати далі й добити машину.
                raise
            except Exception as exc:  # noqa: BLE001
                session.budget.elapsed_s = session.budget.elapsed_s + (self.clock() - started)
                return self._abort(session, f"виклик моделі провалився: {type(exc).__name__}: {exc}")

            session.budget.steps += 1
            session.budget.tokens += decision.tokens_in + decision.tokens_out

            # --- модель нічого не викликає: це кінець ---
            if decision.tool is None:
                session.add(
                    Step(
                        index=0,
                        kind="final",
                        at=utc_now(),
                        note=decision.text or "модель завершила без відповіді",
                        tokens_in=decision.tokens_in,
                        tokens_out=decision.tokens_out,
                        duration_s=decision.latency_s,
                    )
                )
                session.status = "done"
                session.final_answer = decision.text
                session.budget.elapsed_s += self.clock() - started
                self.store.save(session)
                return session

            self._record_plan(session, decision)

            spec = self.registry.get(decision.tool.name)
            if spec is None:
                # Модель вигадала інструмент. Це не привід падати: повертаємо їй
                # помилку тим самим каналом, що й будь-який інший результат.
                self._record_tool_result(
                    session,
                    decision.tool,
                    result={"error": f"інструмента {decision.tool.name!r} не існує",
                            "available": sorted(self.registry)},
                    error=f"невідомий інструмент {decision.tool.name!r}",
                )
                self.store.save(session)
                continue

            loop_reason = self._looping(session, decision.tool)
            if loop_reason:
                session.budget.elapsed_s += self.clock() - started
                return self._abort(session, loop_reason)

            if spec.requires_confirmation and self.cfg.require_confirmation:
                # Аргументи перевіряються ДО паузи. Інакше буває так: людина
                # підтверджує відповідь, і аж тоді інструмент відмовляється,
                # бо в шаблоні бракує поля. Її потурбували заради дії, яка не
                # могла виконатись — а це найшвидший спосіб привчити людину
                # тиснути «підтвердити» не читаючи.
                preview, preview_error = self._preview(session, spec, decision.tool)
                if preview_error is not None:
                    self._record_tool_result(
                        session,
                        decision.tool,
                        result={"error": preview_error,
                                "hint": "Виправ аргументи і виклич інструмент ще раз."},
                        error=preview_error,
                    )
                    self.store.save(session)
                    continue

                session.add(
                    Step(
                        index=0,
                        kind="confirmation_request",
                        at=utc_now(),
                        note=spec.confirmation_title or f"потрібне підтвердження: {spec.name}",
                        tool=decision.tool,
                        # Людина бачить те, що реально відбудеться: готовий текст
                        # листа, а не сирі аргументи виклику.
                        result=preview,
                    )
                )
                session.status = "awaiting_confirmation"
                session.pending_tool = decision.tool
                session.pending_reason = spec.confirmation_title or spec.name
                session.budget.elapsed_s += self.clock() - started
                self.store.save(session)
                logger.info("сесія %s чекає на людину: %s", session.session_id, spec.name)
                return session

            self._execute(session, spec, decision.tool)
            self.store.save(session)

            if spec.terminal:
                # Термінальний інструмент завершує сесію: давати моделі ще один
                # хід після ескалації означало б дозволити їй передумати вже
                # після того, як запит передали людині.
                session.status = "done"
                session.final_answer = self._terminal_answer(session)
                session.budget.elapsed_s += self.clock() - started
                self.store.save(session)
                return session

    # -- рішення людини ----------------------------------------------------

    async def resume(
        self, session: Session, decision: Decision, decided_by: str = "людина"
    ) -> Session:
        """
        Продовжує сесію після рішення людини.

        Відмова НЕ завершує сесію. Модель дізнається про неї тим самим каналом,
        що й про будь-який інший результат інструмента, і може обрати інший шлях —
        зазвичай ескалацію. Обривати роботу на «ні» означало б залишити запит
        без відповіді взагалі.
        """
        if session.status != "awaiting_confirmation" or session.pending_tool is None:
            raise ValueError(f"сесія {session.session_id} не чекає на підтвердження")

        pending = session.pending_tool
        session.add(
            Step(
                index=0,
                kind="human_decision",
                at=utc_now(),
                note=f"{decided_by}: {'підтверджено' if decision == 'approve' else 'відхилено'}",
                tool=pending,
                decision=decision,
                decided_by=decided_by,
            )
        )
        session.pending_tool = None
        session.pending_reason = None
        session.status = "running"

        if decision == "approve":
            spec = self.registry[pending.name]
            self._execute(session, spec, pending)
            self.store.save(session)
            if spec.terminal:
                session.status = "done"
                self.store.save(session)
                return session
        else:
            self._record_tool_result(
                session,
                pending,
                result={
                    "performed": False,
                    "rejected_by": decided_by,
                    "hint": "Людина відхилила цю дію. Не повторюй її з тими самими "
                    "аргументами — обери інший шлях або передай запит людині.",
                },
                error="відхилено людиною",
            )
            self.store.save(session)

        return await self.run(session)

    # -- внутрішнє ---------------------------------------------------------

    def _budget_exceeded(self, session: Session, elapsed: float) -> Optional[str]:
        if session.budget.steps >= self.cfg.max_steps:
            return (
                f"вичерпано ліміт кроків: {session.budget.steps} з {self.cfg.max_steps}. "
                "Агент не дійшов до відповіді — запит потребує людини."
            )
        if session.budget.tokens >= self.cfg.max_tokens:
            return (
                f"вичерпано ліміт токенів: {session.budget.tokens} з {self.cfg.max_tokens}"
            )
        if elapsed >= self.cfg.timeout_s:
            return f"вичерпано таймаут: {elapsed:.1f} с з {self.cfg.timeout_s} с"
        return None

    def _looping(self, session: Session, call: ToolCall) -> Optional[str]:
        """
        Окремий запобіжник від ходіння по колу.

        Ліміту кроків недостатньо: агент може шість разів поспіль викликати той
        самий пошук із тим самим запитом і чесно вкластися в бюджет. Однакові
        аргументи означають однаковий результат — це вже не робота.
        """
        same = sum(
            1
            for previous in session.tool_calls()
            if previous.name == call.name and previous.args == call.args
        )
        if same >= self.cfg.max_same_tool_calls:
            return (
                f"інструмент {call.name} викликано {same + 1} раз(и) з тими самими "
                "аргументами — агент ходить по колу"
            )
        return None

    def _record_plan(self, session: Session, decision: ModelDecision) -> None:
        assert decision.tool is not None  # викликається лише разом з інструментом
        # Модель не завжди супроводжує виклик текстом. Тоді в трасі лишається
        # хоча б назва інструмента — порожній крок «plan» читати неможливо.
        note = decision.text or f"викликати {decision.tool.name}"
        session.add(
            Step(
                index=0,
                kind="plan",
                at=utc_now(),
                note=note,
                tokens_in=decision.tokens_in,
                tokens_out=decision.tokens_out,
                duration_s=decision.latency_s,
            )
        )
        # Підпис зберігається разом із кроком, а не в пам'яті об'єкта: історію
        # для наступного ходу збирають із траси, зокрема й після паузи на
        # підтвердженні в іншому процесі.
        session.add(
            Step(
                index=0,
                kind="tool_call",
                at=utc_now(),
                tool=decision.tool,
                signature=decision.signature,
            )
        )

    def _record_tool_result(
        self,
        session: Session,
        call: ToolCall,
        result: dict,
        error: Optional[str] = None,
        duration: float = 0.0,
    ) -> None:
        session.add(
            Step(
                index=0,
                kind="tool_result",
                at=utc_now(),
                tool=call,
                result=result,
                error=error,
                duration_s=duration,
            )
        )

    def _preview(
        self, session: Session, spec: ToolSpec, call: ToolCall
    ) -> tuple[Optional[dict], Optional[str]]:
        """
        Суха перевірка аргументів перед підтвердженням.

        Повертає (прев'ю, None) або (None, текст помилки). Інструмент без
        `preview` вважається таким, що перевірити наперед неможливо — тоді
        людина бачить сирі аргументи, як і раніше.
        """
        if spec.preview is None:
            return None, None
        ctx = ToolContext(
            store=self.store,
            kb=self.kb,
            session_id=session.session_id,
            sender=session.request.sender,
        )
        try:
            return spec.preview(ctx, call.args), None
        except ToolError as exc:
            return None, str(exc)
        except (MemoryError, RecursionError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("перевірка аргументів %s впала", spec.name)
            return None, f"{type(exc).__name__}: {exc}"

    def _execute(self, session: Session, spec: ToolSpec, call: ToolCall) -> None:
        ctx = ToolContext(
            store=self.store,
            kb=self.kb,
            session_id=session.session_id,
            sender=session.request.sender,
        )
        started = time.monotonic()
        try:
            result = spec.handler(ctx, call.args)
            error = None
        except ToolError as exc:
            # Помилка, зрозуміла моделі: неправильні аргументи, порушений
            # регламент. Вона побачить текст і зможе виправитись сама.
            result = {"error": str(exc)}
            error = str(exc)
        except (MemoryError, RecursionError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("інструмент %s впав", spec.name)
            result = {"error": f"{type(exc).__name__}: {exc}"}
            error = f"{type(exc).__name__}: {exc}"

        session.budget.tool_calls += 1
        self._record_tool_result(
            session, call, result=result, error=error, duration=time.monotonic() - started
        )

    @staticmethod
    def _terminal_answer(session: Session) -> str:
        """Людяне резюме останнього кроку — його показує UI як підсумок сесії."""
        last = session.steps[-1] if session.steps else None
        result = (last.result if last else None) or {}
        if result.get("escalated"):
            to = result.get("to") or "людині"
            return f"Передано {to}: {result.get('reason', 'без причини')}"
        return str(result.get("summary") or "Сесію завершено")

    def _abort(self, session: Session, reason: str) -> Session:
        session.add(Step(index=0, kind="abort", at=utc_now(), note=reason))
        session.status = "aborted"
        session.abort_reason = reason
        self.store.save(session)
        logger.warning("сесія %s зупинена: %s", session.session_id, reason)
        return session

    def _history(self, session: Session) -> list[Turn]:
        """
        Збирає розмову з моделлю зі збережених кроків.

        Саме тому пауза на підтвердженні переживає перезапуск процесу: історія
        не зберігається окремо, вона **виводиться** з траси, яка вже в базі.
        """
        turns: list[Turn] = [
            Turn(
                role="user",
                text=(
                    f"Канал: {session.request.channel}\n"
                    f"Від: {session.request.sender}\n"
                    f"Запит:\n{session.request.text}"
                ),
            )
        ]
        for step in session.steps:
            if step.kind == "tool_call" and step.tool is not None:
                turns.append(
                    Turn(role="model", tool_call=step.tool, signature=step.signature)
                )
            elif step.kind == "tool_result" and step.tool is not None:
                turns.append(
                    Turn(role="tool", tool_name=step.tool.name, tool_result=step.result or {})
                )
        return turns
