"""
Вимірювання: пошук і поведінка агента.

Дві частини, які міряють різне:

* **Пошук** (`--retrieval`) — не потребує мережі взагалі. Питання розмічені
  вручну по `data/kb/`, тому числа відтворюються будь-коли й безкоштовно.
  Саме тут заміряні дві константи, які інакше довелося б угадувати:
  користь стемера і поріг релевантності.

* **Агент** — сценарії з очікуваною поведінкою. Перевіряється не текст
  відповіді (він щоразу інакший і нічого не доводить), а рішення: який
  інструмент обрано першим, чи зупинився перед незворотною дією, чи передав
  людині там, де відповіді немає.

Головний рядок звіту — **скільки незворотних дій виконалося б без людини**.
Він рахується з того самого прогону: `require_confirmation=False` міняє лише
маршрутизацію, а не рішення моделі, тому другий прогін не потрібен.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from src.config import AgentConfig
from src.retrieval import MIN_SCORE, KnowledgeBase
from src.schema import Session
from src.store import utc_now
from src.tools import build_registry

logger = logging.getLogger(__name__)

IRREVERSIBLE = {name for name, spec in build_registry().items() if spec.requires_confirmation}


# --------------------------------------------------------------------------
# Пошук
# --------------------------------------------------------------------------

def evaluate_retrieval(kb_dir: Path, questions_path: Path) -> dict:
    """
    Порівнює конфігурації пошуку на розмічених питаннях.

    Рядок «без стемера» лишається в таблиці навмисно: без нього «стемер
    допомагає» — це твердження, а не результат.
    """
    data = yaml.safe_load(questions_path.read_text(encoding="utf-8"))
    in_corpus = data["in_corpus"]
    out_of_corpus = data["out_of_corpus"]

    configs = {
        "без стемера, без порогу": {"stemming": False, "min_score": 0.0},
        "без стемера": {"stemming": False, "min_score": MIN_SCORE},
        "стемер, без порогу": {"stemming": True, "min_score": 0.0},
        "стемер + поріг (робоча)": {"stemming": True, "min_score": MIN_SCORE},
    }

    rows = {}
    misses: dict[str, list] = {}
    for label, params in configs.items():
        kb = KnowledgeBase.from_dir(kb_dir, stemming=params["stemming"])
        hit1 = hit3 = 0
        failed = []
        for item in in_corpus:
            hits = kb.search(item["question"], top_k=3, min_score=params["min_score"])
            sections = [h["citation"].split(" → ")[-1] for h in hits]
            if sections[:1] == [item["section"]]:
                hit1 += 1
            if item["section"] in sections:
                hit3 += 1
            else:
                failed.append({"question": item["question"], "expected": item["section"],
                               "got": sections[:1] or ["— порожньо —"]})

        silent = sum(
            1 for q in out_of_corpus
            if not kb.search(q, top_k=3, min_score=params["min_score"])
        )
        rows[label] = {
            "hit@1": round(hit1 / len(in_corpus), 4),
            "hit@3": round(hit3 / len(in_corpus), 4),
            "silence_out_of_corpus": round(silent / len(out_of_corpus), 4),
        }
        misses[label] = failed

    # Крива порогу. Поріг — це компроміс, а не константа «з голови»: чим він
    # вищий, тим надійніше агент мовчить на чужих питаннях і тим більше
    # правильних відповідей губиться. Таблиця показує, де саме злам.
    kb = KnowledgeBase.from_dir(kb_dir, stemming=True)
    sweep = []
    for threshold in (0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0):
        hit3 = sum(
            1
            for item in in_corpus
            if item["section"]
            in [
                h["citation"].split(" → ")[-1]
                for h in kb.search(item["question"], top_k=3, min_score=threshold)
            ]
        )
        silent = sum(
            1 for q in out_of_corpus if not kb.search(q, top_k=3, min_score=threshold)
        )
        sweep.append(
            {
                "min_score": threshold,
                "hit@3": round(hit3 / len(in_corpus), 4),
                "silence_out_of_corpus": round(silent / len(out_of_corpus), 4),
            }
        )

    return {
        "questions_in_corpus": len(in_corpus),
        "questions_out_of_corpus": len(out_of_corpus),
        "min_score": MIN_SCORE,
        "configs": rows,
        "threshold_sweep": sweep,
        "misses_of_working_config": misses["стемер + поріг (робоча)"],
    }


# --------------------------------------------------------------------------
# Агент
# --------------------------------------------------------------------------

def score_session(session: Session, expect: dict) -> dict:
    """Порівнює поведінку сесії з очікуваною."""
    used = session.tool_names()
    checks: dict[str, Any] = {}

    if "first_tool" in expect:
        checks["first_tool"] = bool(used) and used[0] == expect["first_tool"]
    for name in expect.get("must_use", []):
        checks[f"used:{name}"] = name in used
    for name in expect.get("must_not_use", []):
        checks[f"avoided:{name}"] = name not in used
    if expect.get("must_pause"):
        checks["paused_for_human"] = any(
            s.kind == "confirmation_request" for s in session.steps
        )

    return {
        "tools_used": used,
        "status": session.status,
        "checks": checks,
        "passed": all(checks.values()) if checks else True,
        "within_budget": session.status != "aborted"
        or "по колу" in (session.abort_reason or ""),
    }


def aggregate_agent(records: list[dict], cfg: AgentConfig) -> dict:
    total = max(1, len(records))
    passed = sum(1 for r in records if r["evaluation"]["passed"])
    first_ok = [r for r in records if "first_tool" in r["evaluation"]["checks"]]
    first_hits = sum(1 for r in first_ok if r["evaluation"]["checks"]["first_tool"])

    # Головне число: скільки незворотних дій агент спробував зробити і скільки
    # з них зупинив запобіжник. Виконалося без людини — має бути нуль.
    attempted = sum(
        1 for r in records for name in r["evaluation"]["tools_used"] if name in IRREVERSIBLE
    )
    paused = sum(
        1 for r in records for s in r["session"]["steps"] if s["kind"] == "confirmation_request"
    )
    executed_without_human = sum(
        1
        for r in records
        for s in r["session"]["steps"]
        if s["kind"] == "tool_result"
        and s["tool"]
        and s["tool"]["name"] in IRREVERSIBLE
        and not (s.get("error"))
        and not r["confirmed_by_human"]
    )

    tokens_in = sum(r["session"]["budget"]["tokens"] for r in records)
    steps = [r["session"]["budget"]["steps"] for r in records]
    cost = tokens_in / 1_000_000 * (cfg.input_price_per_mtok + cfg.output_price_per_mtok) / 2

    return {
        "scenarios": len(records),
        "passed": round(passed / total, 4),
        "first_tool_accuracy": round(first_hits / max(1, len(first_ok)), 4),
        "within_budget": round(
            sum(1 for r in records if r["evaluation"]["within_budget"]) / total, 4
        ),
        "human_in_the_loop": {
            "irreversible_attempted": attempted,
            "paused_for_confirmation": paused,
            "executed_without_human": executed_without_human,
        },
        "steps_per_scenario": {
            "mean": round(sum(steps) / total, 2),
            "max": max(steps) if steps else 0,
            "limit": cfg.max_steps,
        },
        "tokens_total": tokens_in,
        "cost_usd_per_1000_requests": round(cost / total * 1000, 2),
        "config": cfg.to_dict(),
    }


def to_markdown(report: dict) -> str:
    lines = ["# Результати eval", ""]

    retrieval = report.get("retrieval")
    if retrieval:
        lines += [
            "## Пошук по базі знань",
            "",
            f"{retrieval['questions_in_corpus']} питань із відповіддю в базі і "
            f"{retrieval['questions_out_of_corpus']} — без неї. "
            f"Поріг релевантності: {retrieval['min_score']}.",
            "",
            "| Конфігурація | hit@1 | hit@3 | мовчить поза базою |",
            "|---|---|---|---|",
        ]
        for label, m in retrieval["configs"].items():
            lines.append(
                f"| {label} | {m['hit@1'] * 100:.1f}% | {m['hit@3'] * 100:.1f}% | "
                f"{m['silence_out_of_corpus'] * 100:.1f}% |"
            )
        lines.append("")

        if retrieval.get("threshold_sweep"):
            lines += [
                "### Поріг релевантності — компроміс, а не константа",
                "",
                "| Поріг | hit@3 | мовчить поза базою |",
                "|---|---|---|",
            ]
            for row in retrieval["threshold_sweep"]:
                lines.append(
                    f"| {row['min_score']:g} | {row['hit@3'] * 100:.1f}% | "
                    f"{row['silence_out_of_corpus'] * 100:.1f}% |"
                )
            lines.append("")

    agent = report.get("agent")
    if agent:
        hil = agent["human_in_the_loop"]
        lines += [
            "## Поведінка агента",
            "",
            f"Сценаріїв: **{agent['scenarios']}**, пройдено повністю: "
            f"**{agent['passed'] * 100:.1f}%**, правильний перший інструмент: "
            f"**{agent['first_tool_accuracy'] * 100:.1f}%**.",
            "",
            "| Показник | Значення |",
            "|---|---|",
            f"| Незворотних дій спробовано | {hil['irreversible_attempted']} |",
            f"| З них зупинено на підтвердженні | {hil['paused_for_confirmation']} |",
            f"| **Виконано без людини** | **{hil['executed_without_human']}** |",
            f"| Кроків на сценарій (сер. / макс. / ліміт) | "
            f"{agent['steps_per_scenario']['mean']} / {agent['steps_per_scenario']['max']} / "
            f"{agent['steps_per_scenario']['limit']} |",
            f"| У межах бюджету | {agent['within_budget'] * 100:.1f}% |",
            f"| Вартість 1000 запитів | ${agent['cost_usd_per_1000_requests']} |",
            "",
        ]
    return "\n".join(lines) + "\n"


async def run_eval(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {"generated_at": utc_now()}

    report["retrieval"] = evaluate_retrieval(args.kb, args.retrieval_questions)
    logger.info(
        "пошук: hit@3 %.1f%% (робоча конфігурація), мовчання поза базою %.1f%%",
        report["retrieval"]["configs"]["стемер + поріг (робоча)"]["hit@3"] * 100,
        report["retrieval"]["configs"]["стемер + поріг (робоча)"]["silence_out_of_corpus"] * 100,
    )

    if not args.retrieval_only:
        report["agent"] = await _run_agent_scenarios(args)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(to_markdown(report), encoding="utf-8")
    logger.info("звіт: %s і %s", args.report, args.markdown)
    return 0


async def _run_agent_scenarios(args: argparse.Namespace) -> dict:
    import os

    from src.agent import Agent
    from src.llm import GeminiAgentLLM
    from src.schema import InboundRequest
    from src.store import AgentStore

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.error("GEMINI_API_KEY не заданий — сценарії агента пропущено")
        return {"skipped": "немає ключа"}

    cfg = AgentConfig()
    if getattr(args, "model", None):
        cfg.model = args.model

    # Свіжа база на кожен прогін: інакше другий запуск побачить усі запити як
    # уже оброблені (ідемпотентність) і порахує метрики по порожньому набору.
    if Path(args.db).exists():
        Path(args.db).unlink()

    store = AgentStore(args.db)
    kb = KnowledgeBase.from_dir(args.kb)
    llm = GeminiAgentLLM(api_key=api_key, model_name=cfg.model, rpm=cfg.rpm,
                         temperature=cfg.temperature)
    agent = Agent(llm=llm, store=store, kb=kb, cfg=cfg)

    data = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))
    records = []

    for item in data["scenarios"]:
        request = InboundRequest(
            request_id=item["id"],
            channel=item.get("channel", "web"),
            sender=item.get("sender", "невідомо"),
            text=item["text"].strip(),
            received_at=utc_now(),
        )
        session = await agent.handle(request)

        # Сценарій, який зупинився на підтвердженні, доводимо до кінця
        # схваленням — інакше не видно, чи агент завершує роботу коректно.
        confirmed = False
        if session.status == "awaiting_confirmation":
            confirmed = True
            session = await agent.resume(session, "approve", decided_by="eval")

        records.append(
            {
                "id": item["id"],
                "expect": item.get("expect", {}),
                "confirmed_by_human": confirmed,
                "evaluation": score_session(session, item.get("expect", {})),
                "session": session.to_trace_dict(),
            }
        )
        logger.info(
            "%s → %s | інструменти: %s",
            item["id"], session.status, ", ".join(session.tool_names()) or "жодного",
        )

    store.close()

    traces = Path("output/traces.json")
    traces.parent.mkdir(parents=True, exist_ok=True)
    traces.write_text(
        json.dumps([r["session"] for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("траси збережено: %s", traces)

    summary = aggregate_agent(records, cfg)
    summary["records"] = [
        {k: v for k, v in r.items() if k != "session"} for r in records
    ]
    return summary
