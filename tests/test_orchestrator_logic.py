"""Tests for TradingOrchestrator pure decision helpers.

These methods only use their arguments, so we bypass the heavy __init__
(which would construct the trader, AI systems and databases) via __new__.
"""

import unittest

from src.core.orchestrator.trading_orchestrator import TradingOrchestrator


def _orch() -> TradingOrchestrator:
    return TradingOrchestrator.__new__(TradingOrchestrator)


class TestValidateDecision(unittest.TestCase):
    def test_valid_decision_defaults_confidence(self) -> None:
        orch = _orch()
        decision = {"action": "buy", "reason": "rsi oversold"}
        self.assertTrue(orch._validate_decision("BTC", decision))
        self.assertEqual(decision["confidence"], 0.5)

    def test_none_confidence_is_defaulted(self) -> None:
        orch = _orch()
        decision = {"action": "hold", "reason": "neutral", "confidence": None}
        self.assertTrue(orch._validate_decision("BTC", decision))
        self.assertEqual(decision["confidence"], 0.5)

    def test_missing_reason_is_invalid(self) -> None:
        orch = _orch()
        self.assertFalse(orch._validate_decision("BTC", {"action": "buy"}))


class TestIsStopLossDecision(unittest.TestCase):
    def test_true_for_triggered_sell(self) -> None:
        orch = _orch()
        self.assertTrue(
            orch._is_stop_loss_decision(
                {"stop_loss_trigger": {"reason": "x"}, "action": "sell_all"}
            )
        )

    def test_false_when_no_trigger(self) -> None:
        orch = _orch()
        self.assertFalse(orch._is_stop_loss_decision({"action": "sell_all"}))

    def test_false_for_triggered_non_sell(self) -> None:
        orch = _orch()
        self.assertFalse(
            orch._is_stop_loss_decision(
                {"stop_loss_trigger": {"reason": "x"}, "action": "buy"}
            )
        )


class TestDecisionNotes(unittest.TestCase):
    def test_notes_include_averaging_and_original_action(self) -> None:
        orch = _orch()
        notes = orch._build_decision_notes(
            {"averaging_analysis": {"attempts_used": 1}, "original_action": "buy"}
        )
        self.assertIn("averaging", notes)
        self.assertIn("from buy", notes)

    def test_no_notes_for_plain_decision(self) -> None:
        orch = _orch()
        self.assertEqual(orch._build_decision_notes({"action": "buy"}), [])


if __name__ == "__main__":
    unittest.main()
