"""Esteira de treino e backtest. Não manda ordem."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from Client.store import Store
from Client.sync import sync_market
from Esteira.backtest import Cost
from Esteira.run import run_artifact
from Esteira.train import fit_static, persist_fit
from Models import MODELS
from Models.base import COINS, INTERVAL_MS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="umbigo",
        description="Treina um modelo estático e roda o backtest fora da amostra. Sem ordem.",
    )
    parser.add_argument("--db", default=None, help="caminho do SQLite (default: data/umbigo.sqlite)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="baixa candle da Hyperliquid")
    _add_market_args(p_sync)

    p_train = sub.add_parser("train", help="escolhe parâmetro no treino e congela")
    _add_market_args(p_train)
    p_train.add_argument("--model", default="tsmom", choices=sorted(MODELS))
    p_train.add_argument("--train-frac", type=float, default=0.7)
    p_train.add_argument("--train-end", default=None, help="YYYY-MM-DD UTC, ignora --train-frac")
    _add_cost_args(p_train)

    p_test = sub.add_parser("backtest", help="roda o artefato congelado no período seguinte")
    p_test.add_argument("--artifact", type=int, default=None)
    p_test.add_argument("--coin", nargs="+", default=None)
    p_test.add_argument("--interval", default="1d", choices=sorted(INTERVAL_MS))
    p_test.add_argument("--model", default="tsmom", choices=sorted(MODELS))
    _add_cost_args(p_test, optional=True)

    p_run = sub.add_parser("run", help="sync + train + backtest")
    _add_market_args(p_run)
    p_run.add_argument("--model", default="tsmom", choices=sorted(MODELS))
    p_run.add_argument("--train-frac", type=float, default=0.7)
    p_run.add_argument("--train-end", default=None)
    _add_cost_args(p_run)

    args = parser.parse_args(argv)
    store = Store(args.db)
    try:
        if args.cmd == "sync":
            return cmd_sync(store, args)
        if args.cmd == "train":
            return cmd_train(store, args)
        if args.cmd == "backtest":
            return cmd_backtest(store, args)
        if args.cmd == "run":
            return cmd_run(store, args)
    finally:
        store.close()
    return 2


def cmd_sync(store: Store, args) -> int:
    for coin in _coins(args):
        stats = sync_market(store, coin, args.interval)
        flag = " truncado" if stats.truncated else ""
        print(
            f"sync {coin} {args.interval}: {stats.rows} candles "
            f"{_day(stats.first_t)} → {_day(stats.last_t)}{flag}"
        )
    return 0


def cmd_train(store: Store, args) -> int:
    ids = []
    for coin in _coins(args):
        ids.append(_train_one(store, args, coin))
    return 0 if ids else 1


def cmd_backtest(store: Store, args) -> int:
    if args.artifact is not None:
        report = run_artifact(store, args.artifact, _cost_override(args))
        _print_test(report)
        return 0
    coins = _coins(args) if args.coin else list(COINS)
    found = False
    for coin in coins:
        row = store.latest_artifact(coin, args.interval, args.model)
        if row is None:
            print(f"sem artefato para {coin} {args.interval}", file=sys.stderr)
            continue
        found = True
        _print_test(run_artifact(store, row["id"], _cost_override(args)))
    return 0 if found else 1


def cmd_run(store: Store, args) -> int:
    cmd_sync(store, args)
    for coin in _coins(args):
        artifact_id = _train_one(store, args, coin)
        _print_test(run_artifact(store, artifact_id, _cost(args)))
    return 0


def _train_one(store: Store, args, coin: str) -> int:
    candles = store.load_candles(coin, args.interval)
    if not candles:
        raise SystemExit(f"sem candle local de {coin} {args.interval}. Roda sync antes.")
    model = MODELS[args.model]()
    train_end_t = _parse_day(args.train_end) if getattr(args, "train_end", None) else None
    fit = fit_static(
        model,
        candles,
        args.interval,
        _cost(args),
        train_frac=args.train_frac,
        train_end_t=train_end_t,
    )
    artifact_id = persist_fit(store, fit, coin, args.interval)
    gap = fit.metrics.get("sharpe_gap_vs_median")
    relaxed = " (restrição afrouxada)" if fit.metrics.get("constraint_relaxed") else ""
    print(
        f"train {coin} {args.interval} #{artifact_id} {fit.name} "
        f"lookback={fit.params['lookback']} vol_window={fit.params['vol_window']} "
        f"vol_target={fit.params['vol_target']} cap={fit.params['leverage_cap']}"
    )
    print(
        f"  treino {_day(fit.train_start)} → {_day(fit.train_end)}  "
        f"sharpe {_num(fit.metrics.get('train_sharpe'))}  "
        f"maxDD {_pct(fit.metrics.get('train_max_drawdown'))}  "
        f"trades {fit.metrics.get('train_trades')}  "
        f"mediana {_num(fit.metrics.get('sharpe_median'))}  "
        f"gap {_num(gap)}{relaxed}"
    )
    return artifact_id


def _print_test(report: dict) -> None:
    m = report["metrics"]
    p = report["params"]
    print(
        f"teste {report['coin']} {report['interval']} artefato #{report['artifact_id']} "
        f"run #{report['run_id']}  {_day(report['test_start'])} → {_day(report['test_end'])}"
    )
    print(
        f"  ret {_pct(m.get('total_return'))}  cagr {_pct(m.get('cagr'))}  "
        f"sharpe {_num(m.get('sharpe'))}  maxDD {_pct(m.get('max_drawdown'))}  "
        f"trades {m.get('trades')}  win {_pct(m.get('win_rate'))}  "
        f"pf {_num(m.get('profit_factor'))}"
    )
    print(
        f"  exposição {_pct(m.get('exposure'))}  posição média {_num(m.get('avg_abs_position'))}  "
        f"taxa {m.get('fee_bps')}bp + slip {m.get('slippage_bps')}bp  "
        f"funding {m.get('funding')}  "
        f"lb {p.get('lookback')}/vol {p.get('vol_window')}"
    )


def _add_market_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coin", nargs="+", default=list(COINS))
    parser.add_argument("--interval", default="1d", choices=sorted(INTERVAL_MS))


def _add_cost_args(parser: argparse.ArgumentParser, *, optional: bool = False) -> None:
    parser.add_argument("--fee-bps", type=float, default=None if optional else 4.5)
    parser.add_argument("--slippage-bps", type=float, default=None if optional else 1.0)


def _coins(args) -> list[str]:
    coins = [c.upper() for c in args.coin]
    unknown = [c for c in coins if c not in COINS]
    if unknown:
        raise SystemExit(f"umbigo só opera {', '.join(COINS)}. Recebi {', '.join(unknown)}.")
    return coins


def _cost(args) -> Cost:
    return Cost(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)


def _cost_override(args) -> Cost | None:
    if args.fee_bps is None and args.slippage_bps is None:
        return None
    return Cost(
        fee_bps=4.5 if args.fee_bps is None else args.fee_bps,
        slippage_bps=1.0 if args.slippage_bps is None else args.slippage_bps,
    )


def _parse_day(text: str) -> int:
    day = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(day.timestamp() * 1000)


def _day(ms: int | None) -> str:
    if ms is None:
        return "-"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).date().isoformat()


def _pct(value) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.1f}%"


def _num(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
