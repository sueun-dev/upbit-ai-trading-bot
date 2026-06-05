"""Tests for the risk layer: stop-loss triggers and adaptive position sizing.

Both modules had F821 NameErrors that silently disabled them
(risk_monitor returned a crash in its error path; adaptive_risk_manager always
fell back to the 3% default). These tests assert they actually work now.
"""

import unittest

from src.analysis.adaptive_risk_manager import AdaptiveRiskManager
from src.analysis.risk_monitor import RiskMonitor


def _portfolio_with_pnl(symbol: str, pnl_percent: float):
    return {
        "assets": {symbol: {"balance": 1.0, "profit_loss_percentage": pnl_percent}},
        "total_balance": 1_000_000,
        "krw_balance": 500_000,
        "available_krw": 500_000,
    }


class TestStopLossTriggers(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = RiskMonitor(api_key="test-key")

    def test_regular_stop_loss_triggers_partial_sell(self) -> None:
        triggers = self.monitor.check_stop_loss_triggers(
            _portfolio_with_pnl("BTC", -16), {}
        )
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["symbol"], "BTC")
        self.assertEqual(triggers[0]["recommended_action"], "partial_sell")

    def test_emergency_stop_loss_triggers_sell_all(self) -> None:
        triggers = self.monitor.check_stop_loss_triggers(
            _portfolio_with_pnl("BTC", -26), {}
        )
        self.assertEqual(triggers[0]["recommended_action"], "sell_all")

    def test_small_loss_does_not_trigger(self) -> None:
        triggers = self.monitor.check_stop_loss_triggers(
            _portfolio_with_pnl("BTC", -5), {}
        )
        self.assertEqual(triggers, [])


class TestAdaptivePositionSizing(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = AdaptiveRiskManager()
        self.portfolio = {
            "total_balance": 1_000_000,
            "krw_balance": 1_000_000,
            "available_krw": 1_000_000,
            "assets": {},
        }
        self.market = {"volatility_7d": 0.03}

    def _size(self, confidence: float, approved: bool = True) -> float:
        result = self.manager.calculate_optimal_position_size(
            {"confidence": confidence, "action": "buy"},
            self.portfolio,
            self.market,
            {"approved": approved, "risk_level": "low"},
        )
        return result["recommended_size"]

    def test_sizing_runs_and_returns_positive_amount(self) -> None:
        self.assertGreater(self._size(0.8), 0)

    def test_confidence_changes_size(self) -> None:
        # If the pipeline were still short-circuiting to the default (the old bug),
        # confidence would have no effect and these would be equal.
        self.assertNotEqual(self._size(0.9), self._size(0.2))

    def test_ai_rejection_reduces_size(self) -> None:
        # ai_adjusted = heat_adjusted * (1.0 if approved else AI_REJECTION_MULTIPLIER)
        self.assertGreater(
            self._size(0.8, approved=True), self._size(0.8, approved=False)
        )


if __name__ == "__main__":
    unittest.main()
