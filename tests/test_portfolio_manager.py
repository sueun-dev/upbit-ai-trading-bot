"""Tests for portfolio_manager schema enrichment.

The producer must emit the keys every consumer reads (holdings, total_krw,
total_investment, and per-asset profit_loss(_percentage)/krw_value) — these were
previously missing, which silently broke stop-loss and held-coin analysis.
"""

import unittest
from unittest.mock import Mock, patch

from src.analysis.portfolio.portfolio_manager import get_portfolio_status


def _upbit_with(balances):
    upbit = Mock()
    upbit.get_balances.return_value = balances
    return upbit


class TestPortfolioSchema(unittest.TestCase):
    @patch("src.analysis.portfolio.portfolio_manager.pyupbit")
    def test_enriched_schema_has_all_consumer_keys(self, mock_pyupbit) -> None:
        mock_pyupbit.get_current_price.return_value = 68_000_000.0
        upbit = _upbit_with(
            [
                {"currency": "KRW", "balance": "500000", "avg_buy_price": "0"},
                {"currency": "BTC", "balance": "0.001", "avg_buy_price": "65000000"},
            ]
        )

        p = get_portfolio_status(upbit, api_key=None)

        # Top-level keys consumers depend on.
        for key in (
            "total_balance",
            "total_krw",
            "total_investment",
            "available_krw",
            "holdings",
            "assets",
        ):
            self.assertIn(key, p)
        self.assertEqual(p["total_krw"], p["total_balance"])
        self.assertIn("BTC", p["holdings"])
        self.assertEqual(p["available_krw"], 500000.0)

        # Per-asset enrichment used by the risk/stop-loss layers.
        btc = p["assets"]["BTC"]
        self.assertAlmostEqual(btc["krw_value"], 0.001 * 68_000_000)
        self.assertAlmostEqual(btc["profit_loss"], (68_000_000 - 65_000_000) * 0.001)
        self.assertAlmostEqual(
            btc["profit_loss_percentage"],
            (68_000_000 - 65_000_000) / 65_000_000 * 100,
        )

    @patch("src.analysis.portfolio.portfolio_manager.pyupbit")
    def test_total_investment_makes_pnl_correct(self, mock_pyupbit) -> None:
        # total_krw - total_investment should equal unrealized crypto P&L.
        mock_pyupbit.get_current_price.return_value = 70_000_000.0
        upbit = _upbit_with(
            [
                {"currency": "KRW", "balance": "100000", "avg_buy_price": "0"},
                {"currency": "BTC", "balance": "0.002", "avg_buy_price": "60000000"},
            ]
        )
        p = get_portfolio_status(upbit, api_key=None)
        unrealized = p["total_krw"] - p["total_investment"]
        self.assertAlmostEqual(unrealized, (70_000_000 - 60_000_000) * 0.002)

    @patch("src.analysis.portfolio.portfolio_manager.pyupbit")
    def test_no_holdings_skips_ai_and_returns_empty_assets(self, mock_pyupbit) -> None:
        upbit = _upbit_with(
            [{"currency": "KRW", "balance": "1000000", "avg_buy_price": "0"}]
        )
        p = get_portfolio_status(upbit, api_key=None)
        self.assertEqual(p["assets"], {})
        self.assertEqual(p["holdings"], {})
        self.assertEqual(p["total_balance"], 1_000_000.0)


if __name__ == "__main__":
    unittest.main()
