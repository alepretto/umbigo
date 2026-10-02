"""SQLite local. Candle, log de chamada e resultado de treino/backtest."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from Models.base import Candle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "umbigo.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_log (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    coin TEXT,
    interval TEXT,
    start_ms INTEGER,
    end_ms INTEGER,
    http_status INTEGER,
    latency_ms REAL,
    rows INTEGER,
    truncated INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS candles (
    coin TEXT NOT NULL,
    interval TEXT NOT NULL,
    t INTEGER NOT NULL,
    o REAL NOT NULL,
    h REAL NOT NULL,
    l REAL NOT NULL,
    c REAL NOT NULL,
    v REAL NOT NULL,
    n INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (coin, interval, t)
);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    coin TEXT NOT NULL,
    interval TEXT NOT NULL,
    created_at TEXT NOT NULL,
    train_start INTEGER NOT NULL,
    train_end INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY,
    artifact_id INTEGER NOT NULL,
    coin TEXT NOT NULL,
    interval TEXT NOT NULL,
    created_at TEXT NOT NULL,
    test_start INTEGER NOT NULL,
    test_end INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES model_artifacts(id)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    side INTEGER NOT NULL,
    t_entry INTEGER NOT NULL,
    t_exit INTEGER NOT NULL,
    entry_px REAL NOT NULL,
    exit_px REAL NOT NULL,
    pnl REAL NOT NULL,
    bars INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def log_api(
        self,
        *,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        http_status: int | None,
        latency_ms: float,
        rows: int,
        truncated: bool = False,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO api_log (
                created_at, coin, interval, start_ms, end_ms,
                http_status, latency_ms, rows, truncated, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                coin,
                interval,
                start_ms,
                end_ms,
                http_status,
                latency_ms,
                rows,
                int(truncated),
                error,
            ),
        )
        self.conn.commit()

    def upsert_candles(self, coin: str, interval: str, candles: list[Candle]) -> int:
        if not candles:
            return 0
        fetched = utc_now()
        self.conn.executemany(
            """
            INSERT INTO candles (coin, interval, t, o, h, l, c, v, n, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(coin, interval, t) DO UPDATE SET
                o=excluded.o, h=excluded.h, l=excluded.l, c=excluded.c,
                v=excluded.v, n=excluded.n, fetched_at=excluded.fetched_at
            """,
            [
                (coin, interval, c.t, c.o, c.h, c.l, c.c, c.v, c.n, fetched)
                for c in candles
            ],
        )
        self.conn.commit()
        return len(candles)

    def load_candles(
        self,
        coin: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Candle]:
        sql = "SELECT t, o, h, l, c, v, n FROM candles WHERE coin = ? AND interval = ?"
        args: list = [coin, interval]
        if start_ms is not None:
            sql += " AND t >= ?"
            args.append(start_ms)
        if end_ms is not None:
            sql += " AND t <= ?"
            args.append(end_ms)
        sql += " ORDER BY t"
        rows = self.conn.execute(sql, args).fetchall()
        return [Candle(r["t"], r["o"], r["h"], r["l"], r["c"], r["v"], r["n"]) for r in rows]

    def save_artifact(
        self,
        *,
        name: str,
        coin: str,
        interval: str,
        train_start: int,
        train_end: int,
        params: dict,
        metrics: dict,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO model_artifacts (
                name, coin, interval, created_at, train_start, train_end,
                params_json, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                coin,
                interval,
                utc_now(),
                train_start,
                train_end,
                json.dumps(params, sort_keys=True),
                json.dumps(metrics, sort_keys=True),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_artifact(self, artifact_id: int) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"artifact {artifact_id} não existe")
        return row

    def latest_artifact(self, coin: str, interval: str, name: str = "tsmom") -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT * FROM model_artifacts
            WHERE coin = ? AND interval = ? AND name = ?
            ORDER BY id DESC LIMIT 1
            """,
            (coin, interval, name),
        ).fetchone()

    def save_backtest(
        self,
        *,
        artifact_id: int,
        coin: str,
        interval: str,
        test_start: int,
        test_end: int,
        metrics: dict,
        trades: list[dict],
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO backtest_runs (
                artifact_id, coin, interval, created_at, test_start, test_end, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                coin,
                interval,
                utc_now(),
                test_start,
                test_end,
                json.dumps(metrics, sort_keys=True),
            ),
        )
        run_id = int(cur.lastrowid)
        self.conn.executemany(
            """
            INSERT INTO backtest_trades (
                run_id, side, t_entry, t_exit, entry_px, exit_px, pnl, bars
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    t["side"],
                    t["t_entry"],
                    t["t_exit"],
                    t["entry_px"],
                    t["exit_px"],
                    t["pnl"],
                    t["bars"],
                )
                for t in trades
            ],
        )
        self.conn.commit()
        return run_id
