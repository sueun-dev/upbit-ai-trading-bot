"""Shared constants for the trading system (single source of truth).

Trading-action strings, the action groupings, their Korean labels, and the
default AI confidence were previously duplicated across the trader, orchestrator
and analysis modules. They live here so the vocabulary cannot drift.
"""

# Trading actions
ACTION_HOLD = "hold"
ACTION_BUY = "buy"
ACTION_BUY_MORE = "buy_more"
ACTION_SELL_ALL = "sell_all"
ACTION_PARTIAL_SELL = "partial_sell"

# Action groupings
VALID_ACTIONS = {
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_BUY_MORE,
    ACTION_SELL_ALL,
    ACTION_PARTIAL_SELL,
}
BUY_ACTIONS = {ACTION_BUY, ACTION_BUY_MORE}
SELL_ACTIONS = {ACTION_SELL_ALL, ACTION_PARTIAL_SELL}

# Korean labels for actions (used in trade analysis records)
ACTION_KOREAN_MAP = {
    ACTION_BUY: "매수",
    ACTION_SELL_ALL: "전량 매도",
    ACTION_PARTIAL_SELL: "부분 매도",
    ACTION_HOLD: "보유",
    ACTION_BUY_MORE: "추가 매수",
}

# Default AI confidence when a decision omits one
DEFAULT_CONFIDENCE = 0.5
