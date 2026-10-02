import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Client.hyperliquid import Page, fetch_candles, history_truncated  # noqa: E402
from Client.store import Store  # noqa: E402
from Esteira.backtest import Cost, run_backtest  # noqa: E402
from Esteira.run import run_artifact  # noqa: E402
from Esteira.train import fit_static, persist_fit  # noqa: E402
from Models.base import INTERVAL_MS, Candle, drop_untraded_prefix  # noqa: E402
from Models.tsmom import TSMOM  # noqa: E402


def candles_from_prices(prices, t0=1_700_000_000_000, step=86_400_000, n=1, v=1.0):
    return [
        Candle(t0 + i * step, p, p * 1.01, p * 0.99, p, v, n) for i, p in enumerate(prices)
    ]


def wavy_prices(n, crash_from=None):
    """Sobe e desce em blocos, para o treino ter mais de um trade."""
    prices = []
    price = 100.0
    for i in range(n):
        if crash_from is not None and i >= crash_from:
            drift = -0.02
        else:
            drift = 0.012 if (i // 25) % 2 == 0 else -0.01
        price *= 1.0 + drift
        prices.append(price)
    return prices


def raw_candle(t, price=100.0):
    return {
        "t": t,
        "T": t + 3_599_999,
        "s": "BTC",
        "i": "1h",
        "o": str(price),
        "c": str(price),
        "h": str(price),
        "l": str(price),
        "v": "1",
        "n": 1,
    }


class CausalTests(unittest.TestCase):
    def test_signal_ignores_future_close(self):
        prices = [100, 101, 102, 103, 104, 110, 90, 80]
        params = {"lookback": 2, "vol_window": 3, "vol_target": 0.3, "leverage_cap": 1}
        prefix = TSMOM().signals(candles_from_prices(prices[:5]), params)
        full = TSMOM().signals(candles_from_prices(prices), params)
        self.assertEqual(prefix, full[:5])

    def test_same_bar_jump_is_not_earned(self):
        prices = [100.0] * 40 + [200.0]
        candles = candles_from_prices(prices)
        params = {"lookback": 5, "vol_window": 10, "vol_target": 0.3, "leverage_cap": 1.0}
        signals = TSMOM().signals(candles, params)
        result = run_backtest(
            candles, signals, "1d", Cost(fee_bps=0, slippage_bps=0), eval_start=1
        )
        self.assertAlmostEqual(result.metrics["total_return"], 0.0, places=6)
        self.assertEqual(signals[-2], 0.0)

    def test_fee_is_charged_on_turnover(self):
        candles = candles_from_prices([100, 100, 110, 110])
        signals = [0.0, 1.0, 0.0, 0.0]
        cost = Cost(fee_bps=10, slippage_bps=0)
        result = run_backtest(candles, signals, "1d", cost, eval_start=1)
        self.assertAlmostEqual(result.metrics["fees_drag"], 2 * cost.rate, places=6)
        self.assertLess(result.metrics["total_return"], 0.10)

    def test_fit_does_not_look_at_the_test_crash(self):
        clean = candles_from_prices(wavy_prices(400))
        ruined = candles_from_prices(wavy_prices(400, crash_from=300))
        left = fit_static(TSMOM(), clean, "1d", Cost(), train_frac=0.7)
        right = fit_static(TSMOM(), ruined, "1d", Cost(), train_frac=0.7)
        self.assertEqual(left.params, right.params)
        self.assertLess(right.train_end, ruined[-1].t)

    def test_backtest_does_not_refit(self):
        candles = candles_from_prices(wavy_prices(220))
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "umbigo.sqlite")
            store.upsert_candles("BTC", "1d", candles)
            fit = fit_static(TSMOM(), candles, "1d", Cost(), train_frac=0.7)
            artifact_id = persist_fit(
                store, fit, "BTC", "1d", artifact_dir=Path(tmp)
            )

            def boom(self, interval, n_bars):
                raise AssertionError("backtest não pode retreinar")

            original = TSMOM.grid
            TSMOM.grid = boom
            try:
                report = run_artifact(store, artifact_id, Cost())
            finally:
                TSMOM.grid = original
                store.close()
        self.assertGreater(report["metrics"]["bars"], 10)
        self.assertEqual(report["params"], fit.params)


class ClientTests(unittest.TestCase):
    def test_pagination_walks_forward(self):
        start = 1_700_000_000_000
        step = INTERVAL_MS["1h"]
        rows = [raw_candle(start + i * step, 100 + i) for i in range(1000)]

        def transport(coin, interval, start_ms, end_ms):
            page = [row for row in rows if start_ms <= row["t"] <= end_ms][:500]
            return page, Page(200, 1.0, len(page))

        candles, truncated = fetch_candles(
            "BTC",
            "1h",
            start,
            rows[-1]["t"],
            transport=transport,
            pause_s=0,
        )
        self.assertEqual(len(candles), 1000)
        self.assertFalse(truncated)
        self.assertEqual(candles[0].c, 100)
        self.assertEqual(candles[-1].c, 1099)

    def test_history_cap_is_flagged(self):
        step = INTERVAL_MS["1h"]
        end = 1_790_000_000_000
        first = end - 4999 * step
        rows = [raw_candle(first + i * step) for i in range(5000)]
        candles = [
            Candle(row["t"], 1, 1, 1, 1, 1, 1) for row in rows
        ]
        self.assertTrue(history_truncated(0, candles, "1h"))

    def test_store_roundtrip_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "umbigo.sqlite")
            store.log_api(
                coin="BTC",
                interval="1d",
                start_ms=0,
                end_ms=10,
                http_status=200,
                latency_ms=12.5,
                rows=2,
            )
            n = store.upsert_candles("BTC", "1d", candles_from_prices([10, 11, 12]))
            again = store.upsert_candles("BTC", "1d", candles_from_prices([10, 11, 13]))
            loaded = store.load_candles("BTC", "1d")
            logs = store.conn.execute("SELECT COUNT(*) AS n FROM api_log").fetchone()["n"]
            store.close()
        self.assertEqual(n, 3)
        self.assertEqual(again, 3)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[-1].c, 13)
        self.assertEqual(logs, 1)

    def test_drops_only_the_untraded_prefix(self):
        backfill = candles_from_prices([1, 2, 3], n=0, v=0)
        live = candles_from_prices([4, 5], t0=backfill[-1].t + 86_400_000)
        kept = drop_untraded_prefix(backfill + live)
        self.assertEqual([c.c for c in kept], [4, 5])


if __name__ == "__main__":
    unittest.main()
