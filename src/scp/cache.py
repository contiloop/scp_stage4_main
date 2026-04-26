"""SQLite teacher API cache.

Cache key = sha256(source_text || student_output || probe_output ||
                   teacher_model || prompt_hash || decoding_json).
Same input -> same cache hit, no duplicate API calls across resumes.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS teacher_cache (
    cache_key TEXT PRIMARY KEY,
    provider TEXT,
    model TEXT,
    prompt_version TEXT,
    prompt_hash TEXT,
    request_json TEXT,
    response_json TEXT,
    teacher_output TEXT,
    teacher_action TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    created_at TEXT
);
"""


def make_cache_key(
    source_text: str,
    student_output: str,
    probe_output: str | None,
    teacher_model: str,
    prompt_hash: str,
    decoding: dict,
) -> str:
    h = hashlib.sha256()
    payload = json.dumps(
        {
            "src": source_text,
            "stu": student_output,
            "probe": probe_output or "",
            "model": teacher_model,
            "prompt_hash": prompt_hash,
            "decoding": decoding,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


class TeacherCache:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.executescript(SCHEMA)

    def get(self, cache_key: str) -> dict | None:
        cur = self._conn.execute(
            "SELECT provider, model, prompt_version, prompt_hash, request_json,"
            " response_json, teacher_output, teacher_action, input_tokens,"
            " output_tokens, cost_usd, latency_ms, created_at"
            " FROM teacher_cache WHERE cache_key=?",
            (cache_key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        keys = [
            "provider", "model", "prompt_version", "prompt_hash", "request_json",
            "response_json", "teacher_output", "teacher_action", "input_tokens",
            "output_tokens", "cost_usd", "latency_ms", "created_at",
        ]
        out: dict[str, Any] = dict(zip(keys, row))
        for k in ("request_json", "response_json"):
            if out[k]:
                try:
                    out[k] = json.loads(out[k])
                except Exception:
                    pass
        return out

    def put(self, cache_key: str, record: dict) -> None:
        rec = dict(record)
        for k in ("request_json", "response_json"):
            v = rec.get(k)
            if v is not None and not isinstance(v, str):
                rec[k] = json.dumps(v, ensure_ascii=False)
        rec.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._conn.execute(
            "INSERT OR REPLACE INTO teacher_cache"
            " (cache_key, provider, model, prompt_version, prompt_hash,"
            "  request_json, response_json, teacher_output, teacher_action,"
            "  input_tokens, output_tokens, cost_usd, latency_ms, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cache_key,
                rec.get("provider"),
                rec.get("model"),
                rec.get("prompt_version"),
                rec.get("prompt_hash"),
                rec.get("request_json"),
                rec.get("response_json"),
                rec.get("teacher_output"),
                rec.get("teacher_action"),
                rec.get("input_tokens"),
                rec.get("output_tokens"),
                rec.get("cost_usd"),
                rec.get("latency_ms"),
                rec.get("created_at"),
            ),
        )

    def close(self) -> None:
        self._conn.close()
