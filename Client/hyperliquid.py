"""Cliente da API pública de info da Hyperliquid. Sem chave e sem ordem."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from Models.base import INTERVAL_MS, Candle

INFO_URL = "https://api.hyperliquid.xyz/info"
# A doc diz 5000. Na prática a resposta vem com ~5000 e pico. Acima disso é corte.
HISTORY_CAP = 4500


class HyperliquidError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Page:
    status: int
    latency_ms: float
    rows: int
    error: str | None = None


def parse_candle(raw: dict) -> Candle:
    return Candle(
        t=int(raw["t"]),
        o=float(raw["o"]),
        h=float(raw["h"]),
        l=float(raw["l"]),
        c=float(raw["c"]),
        v=float(raw["v"]),
        n=int(raw["n"]),
    )


def history_truncated(start_ms: int, candles: list[Candle], interval: str) -> bool:
    if len(candles) < HISTORY_CAP:
        return False
    return candles[0].t > start_ms + 2 * INTERVAL_MS[interval]


def fetch_candles(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    transport=None,
    pause_s: float = 0.2,
    on_page=None,
) -> tuple[list[Candle], bool]:
    """Baixa o range. Pagina se a API devolver bloco parcial; marca se bateu o teto de histórico."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"intervalo inválido: {interval}")
    if end_ms <= start_ms:
        raise ValueError("end_ms tem que ser maior que start_ms")

    post = transport or _post
    interval_ms = INTERVAL_MS[interval]
    found: dict[int, Candle] = {}
    cursor = start_ms
    truncated = False

    for page in range(40):
        payload, meta = post(coin, interval, cursor, end_ms)
        if on_page is not None:
            on_page(meta)
        if meta.error:
            raise HyperliquidError(meta.error, meta.status)
        batch = [parse_candle(row) for row in payload]
        batch = [c for c in batch if start_ms <= c.t <= end_ms]
        batch.sort(key=lambda c: c.t)
        if not batch:
            break
        for candle in batch:
            found[candle.t] = candle
        if page == 0 and history_truncated(start_ms, batch, interval):
            truncated = True
        last = batch[-1].t
        if last + interval_ms >= end_ms or len(batch) < 400:
            break
        nxt = last + 1
        if nxt <= cursor:
            break
        cursor = nxt
        if pause_s:
            time.sleep(pause_s)

    candles = [found[t] for t in sorted(found)]
    if history_truncated(start_ms, candles, interval):
        truncated = True
    return candles, truncated


def _post(coin: str, interval: str, start_ms: int, end_ms: int) -> tuple[list, Page]:
    body = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
        },
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        INFO_URL,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "umbigo/0.1"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        latency = (time.perf_counter() - started) * 1000
        detail = exc.read().decode("utf-8", "replace")[:400]
        return [], Page(exc.code, latency, 0, error=detail or str(exc))
    except urllib.error.URLError as exc:
        latency = (time.perf_counter() - started) * 1000
        return [], Page(0, latency, 0, error=str(exc.reason))

    latency = (time.perf_counter() - started) * 1000
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], Page(status, latency, 0, error=f"json inválido: {exc}")
    if isinstance(payload, dict):
        return [], Page(status, latency, 0, error=json.dumps(payload)[:400])
    if not isinstance(payload, list):
        return [], Page(status, latency, 0, error=f"resposta inesperada: {type(payload).__name__}")
    return payload, Page(status, latency, len(payload))
