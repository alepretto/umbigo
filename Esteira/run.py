"""Liga o modelo congelado na janela que ele não viu."""

from __future__ import annotations

import json

from Client.store import Store
from Esteira.backtest import Cost, run_backtest
from Models import MODELS
from Models.base import drop_untraded_prefix


def run_artifact(store: Store, artifact_id: int, cost: Cost | None = None) -> dict:
    row = store.get_artifact(artifact_id)
    model_cls = MODELS.get(row["name"])
    if model_cls is None:
        raise KeyError(f"modelo desconhecido: {row['name']}")
    model = model_cls()
    params = json.loads(row["params_json"])
    cost = cost or Cost(fee_bps=row_fee(row), slippage_bps=row_slip(row))

    raw = store.load_candles(row["coin"], row["interval"])
    candles = drop_untraded_prefix(raw)
    if len(candles) < 3:
        raise ValueError("sem candle negociável para testar")

    signals = model.signals(candles, params)
    test_from = 1
    for i, candle in enumerate(candles):
        if candle.t > row["train_end"]:
            test_from = i
            break
    else:
        raise ValueError("não sobrou barra depois do treino")

    result = run_backtest(
        candles,
        signals,
        row["interval"],
        cost,
        eval_start=test_from,
        eval_end=len(candles) - 1,
    )
    run_id = store.save_backtest(
        artifact_id=artifact_id,
        coin=row["coin"],
        interval=row["interval"],
        test_start=candles[test_from].t,
        test_end=candles[-1].t,
        metrics=result.metrics,
        trades=[t.as_dict() for t in result.trades],
    )
    return {
        "run_id": run_id,
        "artifact_id": artifact_id,
        "coin": row["coin"],
        "interval": row["interval"],
        "params": params,
        "test_start": candles[test_from].t,
        "test_end": candles[-1].t,
        "metrics": result.metrics,
    }


def row_fee(row) -> float:
    metrics = json.loads(row["metrics_json"])
    return float(metrics.get("fee_bps", 4.5))


def row_slip(row) -> float:
    metrics = json.loads(row["metrics_json"])
    return float(metrics.get("slippage_bps", 1.0))
