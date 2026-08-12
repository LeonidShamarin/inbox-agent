"""
Стан агента: сесії, черга підтверджень, побічні ефекти інструментів.

Чому SQLite, а не пам'ять процесу. Сесія, що чекає на рішення людини, живе
рівно стільки, скільки людина не заходить у чергу — тобто до наступного ранку.
Тримати її в пам'яті означає, що перезапуск сервісу тихо втрачає всі паузи:
запити зникають, а відправник вважає, що його прийняли в роботу.

Друге: **бюджет має переживати перезапуск**. Якщо витрати живуть у пам'яті,
«ліміт токенів на задачу» перетворюється на «ліміт на спробу», і агент, який
падає й перезапускається, обходить власний запобіжник. Тому `budget` лежить
усередині серіалізованої сесії.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.schema import InboundRequest, Session

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL UNIQUE,
    channel      TEXT NOT NULL,
    sender       TEXT,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    payload      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

-- Побічні ефекти інструментів лежать окремо від сесій: задача, створена
-- агентом, має пережити видалення сесії так само, як створений тікет
-- переживає видалення листа, з якого його завели.
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    title       TEXT NOT NULL,
    assignee    TEXT,
    priority    TEXT,
    due_date    TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS meetings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    title       TEXT NOT NULL,
    participants TEXT,
    starts_at   TEXT,
    duration_min INTEGER
);

CREATE TABLE IF NOT EXISTS replies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    recipient   TEXT,
    template    TEXT,
    body        TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request_hash(request: InboundRequest) -> str:
    """
    Ключ ідемпотентності: канал, відправник і текст.

    Час навмисно не входить — той самий лист, доставлений вебхуком двічі
    (а це нормальна поведінка будь-якої черги з at-least-once), не має
    породжувати другу сесію і другий створений тікет.
    """
    raw = f"{request.channel}|{request.sender}|{request.text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AgentStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI обслуговує запити з пулу потоків,
        # а всі записи тут короткі й проходять через цей один модуль.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if str(path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- сесії -------------------------------------------------------------

    def find_by_request(self, request: InboundRequest) -> Optional[Session]:
        cur = self.conn.execute(
            "SELECT payload FROM sessions WHERE request_hash = ?", (request_hash(request),)
        )
        row = cur.fetchone()
        return Session.model_validate_json(row["payload"]) if row else None

    def save(self, session: Session) -> None:
        """
        Викликається ПІСЛЯ КОЖНОГО кроку, а не в кінці. Падіння на третьому
        кроці має лишати перші два в трасі: без цього неможливо зрозуміти,
        на чому саме агент зламався.
        """
        session.updated_at = utc_now()
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, request_hash, channel, sender, status, created_at, updated_at, payload) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                session.session_id,
                request_hash(session.request),
                session.request.channel,
                session.request.sender,
                session.status,
                session.created_at,
                session.updated_at,
                session.model_dump_json(),
            ),
        )
        self.conn.commit()

    def get(self, session_id: str) -> Optional[Session]:
        cur = self.conn.execute(
            "SELECT payload FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cur.fetchone()
        return Session.model_validate_json(row["payload"]) if row else None

    def awaiting(self, limit: int = 50) -> list[Session]:
        """Черга того, що чекає на людину, — найстаріше першим."""
        cur = self.conn.execute(
            "SELECT payload FROM sessions WHERE status = 'awaiting_confirmation' "
            "ORDER BY created_at LIMIT ?",
            (limit,),
        )
        return [Session.model_validate_json(r["payload"]) for r in cur.fetchall()]

    def recent(self, limit: int = 50) -> list[Session]:
        cur = self.conn.execute(
            "SELECT payload FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [Session.model_validate_json(r["payload"]) for r in cur.fetchall()]

    def counts_by_status(self) -> dict[str, int]:
        cur = self.conn.execute("SELECT status, COUNT(*) AS n FROM sessions GROUP BY status")
        return {r["status"]: r["n"] for r in cur.fetchall()}

    # -- побічні ефекти інструментів ---------------------------------------

    def add_task(self, session_id: str, args: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO tasks (session_id, created_at, title, assignee, priority, due_date, description) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                session_id,
                utc_now(),
                str(args.get("title", "")),
                args.get("assignee"),
                args.get("priority"),
                args.get("due_date"),
                args.get("description"),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_meeting(self, session_id: str, args: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO meetings (session_id, created_at, title, participants, starts_at, duration_min) "
            "VALUES (?,?,?,?,?,?)",
            (
                session_id,
                utc_now(),
                str(args.get("title", "")),
                json.dumps(args.get("participants", []), ensure_ascii=False),
                args.get("starts_at"),
                args.get("duration_min"),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_reply(self, session_id: str, args: dict, body: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO replies (session_id, created_at, recipient, template, body) "
            "VALUES (?,?,?,?,?)",
            (session_id, utc_now(), args.get("recipient"), args.get("template"), body),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def side_effects(self, session_id: str) -> dict:
        """Що агент реально наробив у зовнішньому світі в межах цієї сесії."""
        out = {}
        for table in ("tasks", "meetings", "replies"):
            cur = self.conn.execute(
                f"SELECT * FROM {table} WHERE session_id = ? ORDER BY id", (session_id,)
            )
            rows = [dict(r) for r in cur.fetchall()]
            if rows:
                out[table] = rows
        return out

    def close(self) -> None:
        self.conn.close()
