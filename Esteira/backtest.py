"""Backtest sem lookahead. Sinal no close, retorno do close seguinte."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from Models.base import BARS_PER_YEAR, Candle


@dataclass(frozen=True)
class Cost:
    # Tier base de taker na Hyperliquid, mais 1 bp de escorregada.
    # Não é o teu tier. Passa --fee-bps se a conta for outra.
    fee_bps: float = 4.5
    slippage_bps: float = 1.0

    @property
    def rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000


@dataclass(frozen=True)
class Trade:
    side: int
    t_entry: int
    t_exit: int
    entry_px: float
    exit_px: float
    pnl: float
    bars: int

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "t_entry": self.t_entry,
            "t_exit": self.t_exit,
            "entry_px": self.entry_px,
            "exit_px": self.exit_px,
            "pnl": self.pnl,
            "bars": self.bars,
        }


@dataclass
class BacktestResult:
    metrics: dict
    trades: list[Trade] = field(default_factory=list)
    equity_end: float = 1.0


def run_backtest(
    candles: list[Candle],
    signals: list[float],
    interval: str,
    cost: Cost,
    *,
    eval_start: int,
    eval_end: int | None = None,
) -> BacktestResult:
    """Avalia signals[i] ganhando o retorno da barra i+1, dentro de [eval_start, eval_end]."""
    n = len(candles)
    if len(signals) != n:
        raise ValueError("signals e candles precisam ter o mesmo tamanho")
    if interval not in BARS_PER_YEAR:
        raise ValueError(f"intervalo desconhecido: {interval}")
    if n < 3:
        raise ValueError("precisa de pelo menos 3 candles")

    last = n - 1 if eval_end is None else min(eval_end, n - 1)
    first = max(1, eval_start)
    if last < first:
        raise ValueError("janela de avaliação vazia")

    rets = [0.0] * n
    for i in range(1, n):
        prev = candles[i - 1].c
        rets[i] = candles[i].c / prev - 1.0 if prev else 0.0

    # pos que ganha o retorno da barra i foi decidida no close anterior.
    held = [0.0] * n
    for i in range(1, n):
        held[i] = signals[i - 1]

    bar_pnl: list[float] = []
    costs: list[float] = []
    for i in range(first, last + 1):
        prev_pos = held[i - 1] if i >= 1 else 0.0
        fee = cost.rate * abs(held[i] - prev_pos)
        costs.append(fee)
        bar_pnl.append(held[i] * rets[i] - fee)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for pnl in bar_pnl:
        equity *= 1.0 + pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)

    trades = _trades(candles, held, rets, first, last)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    bars = len(bar_pnl)
    years = bars / BARS_PER_YEAR[interval]
    sharpe = _sharpe(bar_pnl, BARS_PER_YEAR[interval])
    exposure_bars = sum(1 for i in range(first, last + 1) if held[i] != 0.0)

    metrics = {
        "bars": bars,
        "trades": len(trades),
        "total_return": equity - 1.0,
        "cagr": (equity ** (1.0 / years) - 1.0) if years > 0 and equity > 0 else None,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": (len(wins) / len(trades)) if trades else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "exposure": exposure_bars / bars if bars else 0.0,
        "avg_abs_position": sum(abs(held[i]) for i in range(first, last + 1)) / bars,
        "fees_drag": sum(costs),
        "turnover": sum(abs(held[i] - held[i - 1]) for i in range(first, last + 1)),
        "funding": "excluded",
        "fee_bps": cost.fee_bps,
        "slippage_bps": cost.slippage_bps,
        "equity_end": equity,
    }
    return BacktestResult(metrics=_clean(metrics), trades=trades, equity_end=equity)


def _trades(
    candles: list[Candle], held: list[float], rets: list[float], first: int, last: int
) -> list[Trade]:
    trades: list[Trade] = []
    start: int | None = None
    side = 0
    for i in range(first, last + 2):
        pos = held[i] if i <= last else 0.0
        sign = 1 if pos > 0 else -1 if pos < 0 else 0
        if start is None:
            if sign != 0:
                start = i
                side = sign
            continue
        if sign != side:
            a = start
            b = i - 1
            entry_i = a - 1
            pnl = sum(held[k] * rets[k] for k in range(a, b + 1))
            trades.append(
                Trade(
                    side=side,
                    t_entry=candles[entry_i].t,
                    t_exit=candles[b].t,
                    entry_px=candles[entry_i].c,
                    exit_px=candles[b].c,
                    pnl=pnl,
                    bars=b - a + 1,
                )
            )
            start = i if sign != 0 else None
            side = sign
    return trades


def _sharpe(pnls: list[float], bars_per_year: float) -> float | None:
    n = len(pnls)
    if n < 2:
        return None
    mu = sum(pnls) / n
    var = sum((x - mu) ** 2 for x in pnls) / (n - 1)
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return None
    return mu / sd * math.sqrt(bars_per_year)


def _clean(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out
