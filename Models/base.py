"""Tipos do modelo estático. Sem IO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


# Hyperliquid: no máximo ~5000 candles recentes por intervalo.
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

# Cripto não fecha no fim de semana.
BARS_PER_YEAR: dict[str, float] = {
    name: 365.0 * 86_400_000 / ms for name, ms in INTERVAL_MS.items()
}

COINS = ("BTC", "ETH", "SOL")


@dataclass(frozen=True)
class Candle:
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float
    n: int


def drop_untraded_prefix(candles: list[Candle]) -> list[Candle]:
    """Corta o backfill da HL (preço com n=0) antes do mercado existir de verdade."""
    for i, candle in enumerate(candles):
        if candle.n > 0 and candle.v > 0:
            return candles[i:]
    return []


class StaticModel(Protocol):
    name: str

    def grid(self, interval: str, n_bars: int) -> list[dict]:
        """Combinações candidatas. Quem escolhe é a esteira, só no treino."""

    def signals(self, candles: list[Candle], params: dict) -> list[float]:
        """Posição em [-cap, cap] conhecida no close de cada barra. Causal."""
