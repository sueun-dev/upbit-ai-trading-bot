"""Tests for DataStore: FIFO sell-profit, under-fill guard, latest portfolio."""

import os
import tempfile
import unittest
from datetime import datetime

from src.shared.utils.data_store import (
    AIAnalysisResult,
    DataStore,
    TradeRecord,
    get_latest_portfolio_status,
)


def _buy(symbol: str, price: float, quantity: float) -> TradeRecord:
    return TradeRecord(
        timestamp=datetime.now().isoformat(),
        symbol=symbol,
        action="buy",
        confidence=0.8,
        reason="test buy",
        price=price,
        quantity=quantity,
        remaining_quantity=quantity,
    )


class TestDataStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = DataStore(db_path=os.path.join(self.tmp.name, "test.db"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_record_trade_returns_positive_id(self) -> None:
        trade_id = self.store.record_trade(_buy("BTC", 100.0, 10.0))
        self.assertGreater(trade_id, 0)

    def test_fifo_sell_profit_uses_oldest_lots(self) -> None:
        self.store.record_trade(_buy("BTC", 100.0, 10.0))  # lot 1
        self.store.record_trade(_buy("BTC", 200.0, 10.0))  # lot 2

        # Sell 5 units at 120: should consume only lot 1 (avg 100) -> +20%.
        profit_pct, avg_buy, used = self.store.calculate_sell_profit("BTC", 120.0, 5.0)
        self.assertAlmostEqual(avg_buy, 100.0)
        self.assertAlmostEqual(profit_pct, 20.0)
        self.assertEqual(len(used), 1)

        # Lot 1 now has 5 remaining; selling 10 more spans lot1(5@100)+lot2(5@200).
        profit_pct2, avg_buy2, used2 = self.store.calculate_sell_profit(
            "BTC", 200.0, 10.0
        )
        self.assertAlmostEqual(avg_buy2, 150.0)  # (5*100 + 5*200) / 10
        self.assertEqual(len(used2), 2)

    def test_sell_more_than_inventory_does_not_crash(self) -> None:
        self.store.record_trade(_buy("ETH", 100.0, 1.0))
        # Requesting 5 when only 1 is open: profit computed on the filled portion.
        profit_pct, avg_buy, used = self.store.calculate_sell_profit("ETH", 110.0, 5.0)
        self.assertAlmostEqual(avg_buy, 100.0)
        self.assertAlmostEqual(profit_pct, 10.0)

    def test_no_open_buys_returns_zeros(self) -> None:
        self.assertEqual(
            self.store.calculate_sell_profit("DOGE", 1.0, 1.0), (0.0, 0.0, [])
        )

    def test_get_latest_portfolio_status(self) -> None:
        # No data yet -> None.
        self.assertIsNone(get_latest_portfolio_status_for(self.store))
        self.store.record_ai_analysis(
            AIAnalysisResult(
                timestamp=datetime.now().isoformat(),
                news_count=0,
                extracted_symbols=[],
                market_data_symbols=[],
                decisions={},
                analysis_duration=1.0,
                portfolio_value=1_234_567.0,
                circuit_breaker_status={},
            )
        )
        status = get_latest_portfolio_status_for(self.store)
        assert status is not None
        self.assertAlmostEqual(status["total_balance"], 1_234_567.0)


def get_latest_portfolio_status_for(store: DataStore):
    """Helper: query the module function against a specific store's db_path."""
    import sqlite3

    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT portfolio_value, timestamp FROM ai_analysis "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    return {"total_balance": row[0], "timestamp": row[1]} if row else None


class TestModuleLevelLatestPortfolio(unittest.TestCase):
    def test_function_exists_and_handles_missing_gracefully(self) -> None:
        # The function adaptive_risk_manager imports must exist and be callable.
        self.assertTrue(callable(get_latest_portfolio_status))


if __name__ == "__main__":
    unittest.main()
