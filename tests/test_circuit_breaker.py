"""Tests for CircuitBreaker timestamp guarding and trade gating."""

import os
import tempfile
import unittest
from datetime import datetime

from src.core.clients.circuit_breaker import CircuitBreaker, _parse_timestamp


class TestParseTimestamp(unittest.TestCase):
    def test_valid_iso(self) -> None:
        self.assertEqual(
            _parse_timestamp("2024-01-02T03:04:05"), datetime(2024, 1, 2, 3, 4, 5)
        )

    def test_malformed_returns_min(self) -> None:
        self.assertEqual(_parse_timestamp("not-a-date"), datetime.min)

    def test_none_returns_min(self) -> None:
        self.assertEqual(_parse_timestamp(None), datetime.min)


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.breaker = CircuitBreaker()
        # Redirect persistence away from the package directory.
        self.breaker.data_file = os.path.join(self.tmp.name, "cb.json")
        self.breaker.trade_history = []
        self.breaker.circuit_open = False
        self.breaker.circuit_open_time = None

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fresh_breaker_allows_trading(self) -> None:
        self.assertTrue(self.breaker.can_trade())

    def test_malformed_timestamp_does_not_crash(self) -> None:
        # A bad timestamp in history must not raise on the hot trading path.
        self.breaker.trade_history = [
            {"timestamp": "garbage", "profit_loss_percent": -1.0}
        ]
        self.assertIsInstance(self.breaker.can_trade(), bool)
        self.assertIsInstance(self.breaker.get_status(), dict)

    def test_daily_loss_limit_blocks_trading(self) -> None:
        now = datetime.now().isoformat()
        self.breaker.trade_history = [
            {"timestamp": now, "profit_loss_percent": -6.0},
            {"timestamp": now, "profit_loss_percent": -6.0},
        ]
        # -12% cumulative exceeds the 10% daily loss limit.
        self.assertFalse(self.breaker.can_trade())


if __name__ == "__main__":
    unittest.main()
