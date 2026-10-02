"""Escolhe parâmetro no treino e congela. O teste não vota."""

from __future__ import annotations

import json
from dataclasses import dataclass

from Client.store import ROOT, Store
from Esteira.backtest import Cost, run_backtest
from Models.base import Candle, StaticModel, drop_untraded_prefix


@dataclass(frozen=True)
class Fit:
    name: str
    params: dict
    train_start: int
    train_end: int
    metrics: dict
    candles: list[Candle]


def train_end_index(candles: list[Candle], train_frac: float) -> int:
    if not 0.5 <= train_frac <= 0.9:
        raise ValueError("train_frac entre 0.5 e 0.9")
    if len(candles) < 10:
        raise ValueError("série curta demais")
    span = candles[-1].t - candles[0].t
    cutoff = candles[0].t + train_frac * span
    idx = 0
    for i, candle in enumerate(candles):
        if candle.t <= cutoff:
            idx = i
    # Deixa sobrar barra pro retorno do último sinal de treino e pro teste.
    return min(idx, len(candles) - 2)


def fit_static(
    model: StaticModel,
    candles: list[Candle],
    interval: str,
    cost: Cost,
    *,
    train_frac: float = 0.7,
    train_end_t: int | None = None,
) -> Fit:
    series = drop_untraded_prefix(candles)
    if len(series) < 30:
        raise ValueError("depois de cortar o backfill sem trade, sobrou quase nada")

    if train_end_t is None:
        end_idx = train_end_index(series, train_frac)
    else:
        end_idx = 0
        for i, candle in enumerate(series):
            if candle.t <= train_end_t:
                end_idx = i
        end_idx = min(end_idx, len(series) - 2)
    if end_idx < 20:
        raise ValueError("janela de treino curta demais")

    grid = model.grid(interval, end_idx + 1)
    scored: list[dict] = []
    for params in grid:
        signals = model.signals(series, params)
        result = run_backtest(
            series, signals, interval, cost, eval_start=1, eval_end=end_idx
        )
        m = result.metrics
        scored.append(
            {
                "params": params,
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "trades": m["trades"],
                "total_return": m["total_return"],
            }
        )

    viable = []
    for row in scored:
        drawdown = row["max_drawdown"]
        if drawdown is None:
            drawdown = -1.0
        if (row["trades"] or 0) >= 8 and drawdown >= -0.40 and row["sharpe"] is not None:
            viable.append(row)
    relaxed = False
    pool = viable
    if not pool:
        relaxed = True
        pool = [row for row in scored if row["sharpe"] is not None and (row["trades"] or 0) >= 5]
    if not pool:
        raise ValueError("nenhuma combinação produziu trade suficiente no treino")

    def sort_key(row: dict) -> tuple:
        # Sharpe manda. Empate: menos drawdown (mais perto de zero), depois menos parâmetro.
        return (
            row["sharpe"],
            row["max_drawdown"],
            -row["params"]["lookback"],
        )

    winner = max(pool, key=sort_key)
    sharpes = sorted(row["sharpe"] for row in scored if row["sharpe"] is not None)
    median = sharpes[len(sharpes) // 2] if sharpes else None
    metrics = {
        "model": model.name,
        "interval": interval,
        "train_bars": end_idx,
        "candidates": len(scored),
        "viable": len(viable),
        "constraint_relaxed": relaxed,
        "train_sharpe": winner["sharpe"],
        "train_max_drawdown": winner["max_drawdown"],
        "train_return": winner["total_return"],
        "train_trades": winner["trades"],
        "sharpe_median": median,
        "sharpe_gap_vs_median": (
            None if median is None or winner["sharpe"] is None else round(winner["sharpe"] - median, 6)
        ),
        "grid": scored,
        "funding": "excluded",
        "fee_bps": cost.fee_bps,
        "slippage_bps": cost.slippage_bps,
    }
    return Fit(
        name=model.name,
        params=winner["params"],
        train_start=series[0].t,
        train_end=series[end_idx].t,
        metrics=_round_tree(metrics),
        candles=series,
    )


def persist_fit(
    store: Store,
    fit: Fit,
    coin: str,
    interval: str,
    artifact_dir=None,
) -> int:
    artifact_id = store.save_artifact(
        name=fit.name,
        coin=coin,
        interval=interval,
        train_start=fit.train_start,
        train_end=fit.train_end,
        params=fit.params,
        metrics=fit.metrics,
    )
    directory = ROOT / "Models" / "artifacts" if artifact_dir is None else artifact_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{coin}_{interval}_{artifact_id}.json"
    path.write_text(
        json.dumps(
            {
                "id": artifact_id,
                "name": fit.name,
                "coin": coin,
                "interval": interval,
                "train_start": fit.train_start,
                "train_end": fit.train_end,
                "params": fit.params,
                "metrics": {k: v for k, v in fit.metrics.items() if k != "grid"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_id


def _round_tree(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _round_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_tree(v) for v in value]
    return value
