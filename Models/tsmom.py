"""Time-series momentum com vol alvo. Parâmetro congela depois do treino."""

from __future__ import annotations

import math

from Models.base import BARS_PER_YEAR, Candle

# Risco escolhido, não otimizado. Sharpe é quase invariante a escala constante;
# deixar a grade mexer nisso só veste overfitting de alavancagem.
VOL_TARGET = 0.30
LEVERAGE_CAP = 1.0


class TSMOM:
    name = "tsmom"

    def grid(self, interval: str, n_bars: int) -> list[dict]:
        if interval not in BARS_PER_YEAR:
            raise ValueError(f"intervalo desconhecido: {interval}")
        day = max(1, round(BARS_PER_YEAR[interval] / 365.0))
        # Janela tem que caber no treino. 35% da série é o teto; senão a grade é fantasia.
        max_window = max(day, int(n_bars * 0.35))
        lookbacks = _uniq(min(days * day, max_window) for days in (10, 20, 40, 60, 90))
        vol_windows = _uniq(min(max(5, days * day), max_window) for days in (20, 40, 60))
        lookbacks = [n for n in lookbacks if n >= 2]
        vol_windows = [n for n in vol_windows if n >= 5]
        if not lookbacks or not vol_windows:
            raise ValueError(
                f"histórico curto demais para treinar ({n_bars} barras em {interval})"
            )
        return [
            {
                "lookback": lookback,
                "vol_window": vol_window,
                "vol_target": VOL_TARGET,
                "leverage_cap": LEVERAGE_CAP,
            }
            for lookback in lookbacks
            for vol_window in vol_windows
        ]

    def signals(self, candles: list[Candle], params: dict) -> list[float]:
        lookback = int(params["lookback"])
        vol_window = int(params["vol_window"])
        vol_target = float(params["vol_target"])
        cap = float(params["leverage_cap"])
        interval_bars = params.get("_bars_per_year")
        if lookback < 1 or vol_window < 2:
            raise ValueError("lookback >= 1 e vol_window >= 2")
        if cap <= 0 or vol_target <= 0:
            raise ValueError("vol_target e leverage_cap precisam ser positivos")

        closes = [c.c for c in candles]
        n = len(closes)
        rets = [0.0] * n
        for i in range(1, n):
            prev = closes[i - 1]
            rets[i] = closes[i] / prev - 1.0 if prev else 0.0

        bars_per_year = float(interval_bars) if interval_bars else _infer_bars_per_year(candles)
        ann = math.sqrt(bars_per_year)
        out = [0.0] * n
        warmup = max(lookback, vol_window)
        for i in range(warmup, n):
            mom = closes[i] / closes[i - lookback] - 1.0 if closes[i - lookback] else 0.0
            if mom == 0.0:
                continue
            vol = _stdev(rets[i - vol_window + 1 : i + 1]) * ann
            if vol <= 1e-12:
                continue
            scale = min(cap, vol_target / vol)
            out[i] = math.copysign(scale, mom)
        return out


def _infer_bars_per_year(candles: list[Candle]) -> float:
    if len(candles) < 2:
        return BARS_PER_YEAR["1d"]
    step = candles[1].t - candles[0].t
    if step <= 0:
        return BARS_PER_YEAR["1d"]
    return 365.0 * 86_400_000 / step


def _stdev(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def _uniq(values) -> list[int]:
    return sorted({int(v) for v in values})
