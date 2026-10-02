"""Puxa candle e deixa o rastro da chamada no SQLite."""

from __future__ import annotations

import time
from dataclasses import dataclass

from Client.hyperliquid import fetch_candles
from Client.store import Store

# Chama desde o começo. A API é que corta em ~5000; o prefixo sem trade sai na esteira.
DEFAULT_START_MS = 0


@dataclass(frozen=True)
class SyncStats:
    coin: str
    interval: str
    rows: int
    truncated: bool
    first_t: int | None
    last_t: int | None


def sync_market(
    store: Store,
    coin: str,
    interval: str,
    *,
    start_ms: int = DEFAULT_START_MS,
    end_ms: int | None = None,
    pause_s: float = 0.2,
) -> SyncStats:
    end = int(time.time() * 1000) if end_ms is None else end_ms

    def on_page(page) -> None:
        store.log_api(
            coin=coin,
            interval=interval,
            start_ms=start_ms,
            end_ms=end,
            http_status=page.status,
            latency_ms=page.latency_ms,
            rows=page.rows,
            error=page.error,
        )

    candles, truncated = fetch_candles(
        coin, interval, start_ms, end, pause_s=pause_s, on_page=on_page
    )

    # A última barra ainda aberta mente. Tira se o close dela está no futuro.
    now_ms = int(time.time() * 1000)
    interval_ms = candles[-1].t - candles[-2].t if len(candles) >= 2 else 0
    if candles and interval_ms > 0 and candles[-1].t + interval_ms > now_ms + 1_000:
        candles = candles[:-1]

    store.upsert_candles(coin, interval, candles)
    if truncated:
        store.log_api(
            coin=coin,
            interval=interval,
            start_ms=start_ms,
            end_ms=end,
            http_status=200,
            latency_ms=0,
            rows=len(candles),
            truncated=True,
            error="histórico truncado no teto de ~5000 candles da Hyperliquid",
        )
    return SyncStats(
        coin=coin,
        interval=interval,
        rows=len(candles),
        truncated=truncated,
        first_t=candles[0].t if candles else None,
        last_t=candles[-1].t if candles else None,
    )
