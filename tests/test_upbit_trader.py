"""Tests for UpbitTrader dispatch and helpers (pyupbit mocked, no real orders)."""

import unittest
from unittest.mock import patch

from src.core.clients.upbit_trader import UpbitTrader


def _make_trader(mock_pyupbit):
    mock_pyupbit.get_tickers.return_value = ["KRW-BTC", "KRW-ETH"]
    return UpbitTrader(access_key="a", secret_key="b")


class TestUpbitTrader(unittest.TestCase):
    @patch("src.core.clients.upbit_trader.pyupbit")
    def test_market_code_uses_krw_prefix(self, mock_pyupbit) -> None:
        # Regression: the previous f"{"KRW-"}{symbol}" was Python-3.12-only syntax.
        trader = _make_trader(mock_pyupbit)
        self.assertEqual(trader._get_market_code("BTC"), "KRW-BTC")

    @patch("src.core.clients.upbit_trader.pyupbit")
    def test_averaging_amount_scales_with_confidence(self, mock_pyupbit) -> None:
        trader = _make_trader(mock_pyupbit)
        # base 30_000 * (1 + confidence)
        self.assertAlmostEqual(
            trader._calculate_averaging_amount({"confidence": 0.0}), 30_000
        )
        self.assertAlmostEqual(
            trader._calculate_averaging_amount({"confidence": 1.0}), 60_000
        )
        # confidence is clamped to [0, 1]
        self.assertAlmostEqual(
            trader._calculate_averaging_amount({"confidence": 5.0}), 60_000
        )

    @patch("src.core.clients.upbit_trader.pyupbit")
    def test_hold_action_places_no_order(self, mock_pyupbit) -> None:
        trader = _make_trader(mock_pyupbit)
        self.assertTrue(trader.execute_trade("BTC", "hold", "stay flat"))
        trader.upbit.buy_market_order.assert_not_called()
        trader.upbit.sell_market_order.assert_not_called()

    @patch("src.core.clients.upbit_trader.record_purchase")
    @patch("src.core.clients.upbit_trader.pyupbit")
    def test_buy_more_without_analysis_falls_back_to_regular_buy(
        self, mock_pyupbit, _mock_record
    ) -> None:
        trader = _make_trader(mock_pyupbit)
        mock_pyupbit.get_current_price.return_value = 1000.0
        trader.get_portfolio_status = lambda: {"available_krw": 100_000}  # type: ignore[method-assign]
        # buy_more with no averaging_analysis must place a regular buy, not crash.
        self.assertTrue(trader.execute_trade("BTC", "buy_more", "no analysis", {}))
        trader.upbit.buy_market_order.assert_called_once()

    @patch("src.core.clients.upbit_trader.record_purchase")
    @patch("src.core.clients.upbit_trader.pyupbit")
    def test_regular_buy_uses_risk_managed_amount(
        self, mock_pyupbit, _mock_record
    ) -> None:
        trader = _make_trader(mock_pyupbit)
        mock_pyupbit.get_current_price.return_value = 1000.0
        trader.get_portfolio_status = lambda: {"available_krw": 100_000}  # type: ignore[method-assign]

        self.assertTrue(
            trader.execute_trade(
                "BTC",
                "buy",
                "sized by risk manager",
                {"recommended_amount_krw": 75_000},
            )
        )

        trader.upbit.buy_market_order.assert_called_once_with("KRW-BTC", 75_000)

    @patch("src.core.clients.upbit_trader.pyupbit")
    def test_regular_buy_caps_amount_to_available_krw(self, mock_pyupbit) -> None:
        trader = _make_trader(mock_pyupbit)
        trader.get_portfolio_status = lambda: {"available_krw": 12_000}  # type: ignore[method-assign]

        amount = trader._resolve_buy_amount({"recommended_amount_krw": 75_000})

        self.assertEqual(amount, 12_000)

    @patch("src.core.clients.upbit_trader.pyupbit")
    def test_unknown_action_returns_false(self, mock_pyupbit) -> None:
        trader = _make_trader(mock_pyupbit)
        self.assertFalse(trader.execute_trade("BTC", "teleport", "nonsense"))


if __name__ == "__main__":
    unittest.main()
