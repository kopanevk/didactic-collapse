from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_SCHEMA_VERSION = "judge_cache_v1"


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_cache_key(
    *,
    provider: str,
    model_name: str,
    call_type: str,
    key_payload: dict[str, Any],
    schema_version: str = _CACHE_SCHEMA_VERSION,
) -> str:
    canonical = json.dumps(
        {
            "provider": provider,
            "model_name": model_name,
            "call_type": call_type,
            "schema_version": schema_version,
            "key_payload": key_payload,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return stable_sha256(canonical)


@dataclass(frozen=True)
class JudgeCacheRecord:
    key: str
    provider: str
    model_name: str
    call_type: str
    schema_version: str
    key_payload_json: str
    raw_response: str | None
    parsed_payload_json: str | None
    repair_applied: bool
    repair_actions: tuple[str, ...]
    created_at: str


class JudgeResultCache:
    """SQLite-backed persistent cache for judge requests/results."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    call_type TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    key_payload_json TEXT NOT NULL,
                    raw_response TEXT,
                    parsed_payload_json TEXT,
                    repair_applied INTEGER NOT NULL DEFAULT 0,
                    repair_actions_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )

    def get(self, cache_key: str) -> JudgeCacheRecord | None:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT
                        cache_key,
                        provider,
                        model_name,
                        call_type,
                        schema_version,
                        key_payload_json,
                        raw_response,
                        parsed_payload_json,
                        repair_applied,
                        repair_actions_json,
                        created_at
                    FROM judge_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("judge_cache_get_failed path=%s detail=%s", self._path, exc)
            return None

        if row is None:
            return None

        try:
            repair_actions_raw = json.loads(row[9] or "[]")
            if not isinstance(repair_actions_raw, list):
                raise ValueError("repair_actions_json is not list")
            repair_actions = tuple(str(x) for x in repair_actions_raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge_cache_corrupt_entry key=%s detail=%s", cache_key, exc)
            return None

        return JudgeCacheRecord(
            key=str(row[0]),
            provider=str(row[1]),
            model_name=str(row[2]),
            call_type=str(row[3]),
            schema_version=str(row[4]),
            key_payload_json=str(row[5]),
            raw_response=row[6] if row[6] is None else str(row[6]),
            parsed_payload_json=row[7] if row[7] is None else str(row[7]),
            repair_applied=bool(int(row[8])),
            repair_actions=repair_actions,
            created_at=str(row[10]),
        )

    def put(
        self,
        *,
        cache_key: str,
        provider: str,
        model_name: str,
        call_type: str,
        key_payload_json: str,
        raw_response: str | None,
        parsed_payload_json: str | None,
        repair_applied: bool,
        repair_actions: tuple[str, ...],
        schema_version: str = _CACHE_SCHEMA_VERSION,
    ) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO judge_cache (
                        cache_key,
                        provider,
                        model_name,
                        call_type,
                        schema_version,
                        key_payload_json,
                        raw_response,
                        parsed_payload_json,
                        repair_applied,
                        repair_actions_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        provider,
                        model_name,
                        call_type,
                        schema_version,
                        key_payload_json,
                        raw_response,
                        parsed_payload_json,
                        1 if repair_applied else 0,
                        json.dumps(list(repair_actions), ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except sqlite3.Error as exc:
            logger.warning("judge_cache_put_failed path=%s detail=%s", self._path, exc)
