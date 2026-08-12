"""
CLI агента-оркестратора вхідного потоку.

    python main.py serve                      # HTTP + веб-інтерфейс на :7860
    python main.py ask "Скільки днів відпустки?"   # один запит, трасa в консоль
    python main.py inbox                      # прогнати демо-інбокс
    python main.py eval --retrieval-only      # метрики пошуку, без мережі
    python main.py eval                       # + сценарії агента (потрібен ключ)

Env: GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.config import AgentConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("inbox-agent")

DEFAULT_KB = Path("data/kb")
DEFAULT_DB = Path("state/agent.sqlite3")


def _use_system_trust_store() -> None:
    """
    httpx (усередині google-genai) довіряє лише бандлу `certifi`. У мережах з
    TLS-інспекцією корпоративний корінь є у сховищі ОС, але не в certifi — і всі
    виклики падають з CERTIFICATE_VERIFY_FAILED. truststore бере довіру звідти,
    де вона реально налаштована. Перевірку сертифікатів це не послаблює.

    Викликається ОДИН раз у `main()`, а не в кожній команді: у сусідньому
    проекті рівно одна команда лишилася без цього виклику, і сорок документів
    поспіль упали з помилкою сертифіката, хоча решта команд працювала.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        logger.debug("truststore недоступний, лишаємось на certifi")


def _require_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        logger.error("GEMINI_API_KEY не заданий. Створи .env за зразком .env.example.")
        sys.exit(2)
    return key


def _build_agent(args: argparse.Namespace):
    from src.agent import Agent
    from src.llm import GeminiAgentLLM
    from src.retrieval import KnowledgeBase
    from src.store import AgentStore

    cfg = AgentConfig()
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "max_steps", None):
        cfg.max_steps = args.max_steps
    if getattr(args, "no_confirmation", False):
        cfg.require_confirmation = False

    llm = GeminiAgentLLM(api_key=_require_key(), model_name=cfg.model, rpm=cfg.rpm,
                         temperature=cfg.temperature)
    store = AgentStore(args.db)
    kb = KnowledgeBase.from_dir(args.kb)
    return Agent(llm=llm, store=store, kb=kb, cfg=cfg), store


def _print_trace(session) -> None:
    print(f"\nсесія {session.session_id} — {session.status}")
    for step in session.steps:
        print(f"  {step.index:>2}. {step.summary()}")
        if step.error:
            print(f"      помилка: {step.error}")
    print(
        f"  бюджет: {session.budget.steps} кроків, {session.budget.tokens} токенів, "
        f"{session.budget.elapsed_s:.1f} с"
    )
    if session.pending_tool:
        print(f"  ЧЕКАЄ НА ЛЮДИНУ: {session.pending_tool.name}")
        print(f"    аргументи: {json.dumps(session.pending_tool.args, ensure_ascii=False)}")
    if session.final_answer:
        print(f"  підсумок: {session.final_answer}")
    if session.abort_reason:
        print(f"  зупинено: {session.abort_reason}")


async def _run_ask(args: argparse.Namespace) -> int:
    from src.schema import InboundRequest
    from src.store import utc_now

    agent, store = _build_agent(args)
    request = InboundRequest(
        request_id=f"cli-{uuid.uuid4().hex[:8]}",
        channel="web",
        sender=args.sender,
        text=args.text,
        received_at=utc_now(),
    )
    session = await agent.handle(request)

    if session.status == "awaiting_confirmation" and args.approve:
        print("\n[--approve] підтверджую дію від імені людини")
        session = await agent.resume(session, "approve", decided_by="cli")

    _print_trace(session)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(
            json.dumps(session.to_trace_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  трасa: {args.trace}")
    store.close()
    return 0


async def _run_inbox(args: argparse.Namespace) -> int:
    """Прогін демо-інбоксу: кілька типових запитів підряд."""
    import yaml

    from src.schema import InboundRequest
    from src.store import utc_now

    agent, store = _build_agent(args)
    data = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))
    for item in data["scenarios"][: args.limit] if args.limit else data["scenarios"]:
        request = InboundRequest(
            request_id=item["id"],
            channel=item.get("channel", "web"),
            sender=item.get("sender", "невідомо"),
            text=item["text"].strip(),
            received_at=utc_now(),
        )
        session = await agent.handle(request)
        _print_trace(session)
    store.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    return asyncio.run(_run_ask(args))


def cmd_inbox(args: argparse.Namespace) -> int:
    return asyncio.run(_run_inbox(args))


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from src.app import create_app

    app = create_app(db_path=args.db, kb_dir=args.kb)
    port = args.port or int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from src.evaluate import run_eval

    return asyncio.run(run_eval(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inbox-agent", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", type=Path, default=DEFAULT_DB)
        p.add_argument("--kb", type=Path, default=DEFAULT_KB)
        p.add_argument("--model", default=None)
        p.add_argument("--max-steps", type=int, default=None)
        p.add_argument("--no-confirmation", action="store_true",
                       help="Виконувати незворотні дії без людини (лише для порівняння)")

    p_ask = sub.add_parser("ask", help="Обробити один запит")
    p_ask.add_argument("text")
    p_ask.add_argument("--sender", default="Користувач")
    p_ask.add_argument("--approve", action="store_true",
                       help="Одразу підтвердити дію, якщо агент зупиниться")
    p_ask.add_argument("--trace", type=Path, default=None, help="Куди зберегти трасу")
    common(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    p_inbox = sub.add_parser("inbox", help="Прогнати демо-інбокс зі сценаріїв")
    p_inbox.add_argument("--scenarios", type=Path, default=Path("eval/scenarios.yaml"))
    p_inbox.add_argument("--limit", type=int, default=None)
    common(p_inbox)
    p_inbox.set_defaults(func=cmd_inbox)

    p_serve = sub.add_parser("serve", help="HTTP API + веб-інтерфейс")
    p_serve.add_argument("--port", type=int, default=None)
    common(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_eval = sub.add_parser("eval", help="Метрики пошуку і поведінки агента")
    p_eval.add_argument("--retrieval-only", action="store_true",
                        help="Тільки пошук — без мережі й без ключа")
    p_eval.add_argument("--retrieval-questions", type=Path,
                        default=Path("eval/retrieval_questions.yaml"))
    p_eval.add_argument("--scenarios", type=Path, default=Path("eval/scenarios.yaml"))
    p_eval.add_argument("--report", type=Path, default=Path("output/eval-report.json"))
    p_eval.add_argument("--markdown", type=Path, default=Path("output/eval-report.md"))
    p_eval.add_argument("--db", type=Path, default=Path("state/eval.sqlite3"))
    p_eval.add_argument("--kb", type=Path, default=DEFAULT_KB)
    p_eval.add_argument("--model", default=None)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main() -> int:
    load_dotenv()
    # До розбору аргументів і до будь-якого мережевого виклику.
    _use_system_trust_store()
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
