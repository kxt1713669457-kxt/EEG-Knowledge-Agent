"""小型本机元数据存储；原论文、历史任务和EEG产物保持各自的存储职责。"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from app.shared.runtime.logger import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "output/runtime/workspace.sqlite3"


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def transaction():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(kind,id))")
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def read(kind, identity, connection=None):
    if connection is None:
        with transaction() as connection:
            return read(kind, identity, connection)
    row = connection.execute("SELECT data FROM records WHERE kind=? AND id=?", (kind, identity)).fetchone()
    return json.loads(row[0]) if row else None


def write(kind, identity, data, connection=None):
    if connection is None:
        with transaction() as connection:
            return write(kind, identity, data, connection)
    connection.execute("INSERT INTO records(kind,id,data) VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data",
                       (kind, identity, json.dumps(data, ensure_ascii=False)))
    return data


def all_records(kind):
    with transaction() as connection:
        return [json.loads(row[0]) for row in connection.execute("SELECT data FROM records WHERE kind=?", (kind,))]


def new_record(kind, **fields):
    identity = uuid.uuid4().hex
    return write(kind, identity, {"id": identity, "created_at": now(), "updated_at": now(), **fields})


def ensure_session(session_id, title=""):
    with transaction() as connection:
        row = read("session", session_id, connection)
        if row is None:
            row = {"id": session_id, "title": title[:80] or "新的研究会话", "manual_title": False,
                   "project_id": None, "archived": False, "deleted_at": None, "created_at": now()}
        elif title and not row.get("manual_title") and row["title"] == "新的研究会话":
            row["title"] = title[:80]
        row["updated_at"] = now()
        return write("session", session_id, row, connection)
