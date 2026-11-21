"""SQLite persistence for prediction history (plain SQL via sqlite3).

Every query is parameterised. A new connection is opened per call, which
keeps this safe under multi-worker servers such as gunicorn.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .detector import Prediction

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT    NOT NULL,
    label       TEXT    NOT NULL CHECK (label IN ('phishing', 'legitimate')),
    probability REAL    NOT NULL,
    threshold   REAL    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
"""


class PredictionStore:
    """Small data-access class around the `predictions` table."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or os.environ.get("PHISHING_DB") or DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, p: Prediction) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO predictions (url, label, probability, threshold) "
                "VALUES (?, ?, ?, ?)",
                (p.url, p.label, p.probability, p.threshold),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, url, label, probability, created_at "
                "FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Aggregate counts with GROUP BY."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT label, COUNT(*) AS n, AVG(probability) AS avg_prob "
                "FROM predictions GROUP BY label"
            ).fetchall()
        by_label = {r["label"]: {"count": r["n"], "avg_probability": round(r["avg_prob"], 4)}
                    for r in rows}
        total = sum(v["count"] for v in by_label.values())
        return {"total": total, "by_label": by_label}

    def to_dataframe(self) -> pd.DataFrame:
        """Whole table as a DataFrame, for ad-hoc analysis."""
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM predictions ORDER BY id", conn)
