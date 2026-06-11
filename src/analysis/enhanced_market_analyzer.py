"""Enhanced Market Analyzer for comprehensive market analysis.

This module provides advanced market analysis capabilities including
technical indicators, trend analysis, and market sentiment evaluation.
"""

import logging
from datetime import datetime
from typing import Any, Dict

import numpy as np

# Volume analysis constants
VOLUME_NORMALIZATION_FACTOR = 2

# Volatility constants
VOLATILITY_NORMALIZATION_FACTOR = 10

# Trend analysis constants
TREND_NORMALIZATION_FACTOR = 10
BULLISH_THRESHOLD = 3  # At least 3 positive timeframes
BEARISH_THRESHOLD = 1  # At most 1 positive timeframe

# Signal generation thresholds
STRONG_BUY_THRESHOLD = 0.5
BUY_THRESHOLD = 0.2
STRONG_SELL_THRESHOLD = -0.5
SELL_THRESHOLD = -0.2

# Indicator weights for composite score
DEFAULT_INDICATOR_WEIGHTS = {
    "trend": 0.22,
    "momentum": 0.2,
    "trend_strength": 0.18,
    "volume": 0.16,
    "volatility": 0.14,
    "structure": 0.1,
}

# MACD normalization
MACD_HISTOGRAM_NORMALIZATION = 100

# Score weights for momentum
RSI_WEIGHT = 0.35
MACD_WEIGHT = 0.35
STOCHASTIC_WEIGHT = 0.15
MFI_WEIGHT = 0.15

logger = logging.getLogger(__name__)


class EnhancedMarketAnalyzer:
    """Enhanced market analyzer with comprehensive technical analysis.

    This simplified version uses pre-calculated data from EnhancedMarketData
    instead of recalculating indicators.
    """

    def __init__(self) -> None:
        """Initialize the enhanced market analyzer."""
        self.indicator_weights = DEFAULT_INDICATOR_WEIGHTS.copy()

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive market analysis.

        Args:
            data: Market data from EnhancedMarketData containing all indicators.

        Returns:
            Comprehensive analysis results with trading signal.
        """
        try:
            # Calculate trend score from price changes
            trend_score = self._calculate_trend_score_simple(data)

            # Calculate momentum score from RSI and MACD
            momentum_score = self._calculate_momentum_score_simple(data)

            # Calculate volume score from volume ratio
            volume_score = self._calculate_volume_score_simple(data)

            # Calculate volatility score
            volatility_score = self._calculate_volatility_score_simple(data)

            # Calculate directional trend strength from ADX/DI and EMA alignment
            trend_strength_score = self._calculate_trend_strength_score(data)

            # Calculate price-position structure from Donchian channel
            structure_score = self._calculate_structure_score(data)

            # Calculate composite score
            composite_score = self._calculate_composite_score(
                {
                    "trend": trend_score,
                    "momentum": momentum_score,
                    "trend_strength": trend_strength_score,
                    "volume": volume_score,
                    "volatility": volatility_score,
                    "structure": structure_score,
                }
            )

            # Generate trading signal
            signal = self._generate_signal_simple(composite_score)

            # Calculate confidence
            confidence = abs(composite_score)

            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": data.get("symbol", "UNKNOWN"),
                "current_price": data.get("current_price", 0),
                "composite_score": composite_score,
                "signal": signal,
                "confidence": confidence,
                "trend": self._get_trend_direction(data),
                "risk_flags": self._build_risk_flags(data),
                "scores": {
                    "trend": trend_score,
                    "momentum": momentum_score,
                    "trend_strength": trend_strength_score,
                    "volume": volume_score,
                    "volatility": volatility_score,
                    "structure": structure_score,
                },
            }

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            raise RuntimeError(f"Failed to analyze market data: {e}") from e

    def _calculate_trend_score_simple(self, data: Dict[str, Any]) -> float:
        """Calculate trend score from price changes."""
        # Use already calculated price changes (coerce explicit None to 0).
        price_changes = [
            data.get("price_1h_change") or 0,
            data.get("price_24h_change") or 0,
            data.get("price_7d_change") or 0,
            data.get("price_30d_change") or 0,
        ]

        # Average price change normalized
        avg_change = np.mean(price_changes)
        return float(np.tanh(avg_change / TREND_NORMALIZATION_FACTOR))

    def _calculate_momentum_score_simple(self, data: Dict[str, Any]) -> float:
        """Calculate momentum score from RSI and MACD."""
        # RSI score (0 at 50, positive when oversold, negative when overbought)
        rsi_14 = data.get("rsi_14", 50)
        rsi_score = np.clip((50 - rsi_14) / 35, -1, 1)

        # MACD score from histogram
        macd_histogram = data.get("macd_histogram", 0)
        macd_score = np.tanh(macd_histogram / MACD_HISTOGRAM_NORMALIZATION)

        stochastic_score = np.clip((50 - data.get("stoch_k", 50)) / 50, -1, 1)
        mfi_score = np.clip((50 - data.get("mfi_14", 50)) / 50, -1, 1)

        return float(
            rsi_score * RSI_WEIGHT
            + macd_score * MACD_WEIGHT
            + stochastic_score * STOCHASTIC_WEIGHT
            + mfi_score * MFI_WEIGHT
        )

    def _calculate_volume_score_simple(self, data: Dict[str, Any]) -> float:
        """Calculate volume score from volume ratio."""
        # Use pre-calculated relative volume plus KRW turnover z-score.
        volume_ratio = data.get("volume_ratio_24h_7d", 1.0)
        zscore = data.get("volume_zscore_30d", 0)
        ratio_score = np.tanh((volume_ratio - 1) * VOLUME_NORMALIZATION_FACTOR)
        zscore_score = np.tanh(zscore / 2)
        return float(ratio_score * 0.65 + zscore_score * 0.35)

    def _calculate_volatility_score_simple(self, data: Dict[str, Any]) -> float:
        """Calculate volatility score (lower volatility = higher score)."""
        # Favor tradable but not chaotic markets. ATR percent is tighter than raw
        # close-to-close volatility when available.
        volatility = data.get("atr_percent") or data.get("volatility_7d", 0)
        if volatility <= 0:
            return 0.0
        if volatility <= 2:
            return 0.7
        if volatility <= 7:
            return float(0.7 - ((volatility - 2) / 5) * 0.7)
        return float(-np.tanh((volatility - 7) / VOLATILITY_NORMALIZATION_FACTOR))

    def _calculate_trend_strength_score(self, data: Dict[str, Any]) -> float:
        """Calculate trend quality from ADX/DI and EMA alignment."""
        directional_strength = data.get("trend_strength_score", 0)
        ema_alignment = data.get("ema_alignment_score", 0)
        return float(np.clip(directional_strength * 0.65 + ema_alignment * 0.35, -1, 1))

    def _calculate_structure_score(self, data: Dict[str, Any]) -> float:
        """Score current location inside the Donchian channel."""
        position = data.get("donchian_position_20", 0.5)
        if position <= 0.2:
            return 0.45
        if position >= 0.92:
            return -0.35
        if position >= 0.75 and data.get("volume_zscore_30d", 0) > 1:
            return 0.2
        return float((0.5 - position) * 0.4)

    def _generate_signal_simple(self, composite_score: float) -> str:
        """Generate trading signal based on composite score."""
        if composite_score > STRONG_BUY_THRESHOLD:
            return "strong_buy"
        elif composite_score > BUY_THRESHOLD:
            return "buy"
        elif composite_score < STRONG_SELL_THRESHOLD:
            return "strong_sell"
        elif composite_score < SELL_THRESHOLD:
            return "sell"
        else:
            return "neutral"

    def _get_trend_direction(self, data: Dict[str, Any]) -> str:
        """Get trend direction from price changes."""
        price_changes = [
            data.get("price_1h_change", 0),
            data.get("price_24h_change", 0),
            data.get("price_7d_change", 0),
            data.get("price_30d_change", 0),
        ]
        positive_count = sum(1 for change in price_changes if change > 0)

        if (
            data.get("trend_strength_score", 0) > 0.35
            and data.get("ema_alignment_score", 0) > 0
        ):
            return "bullish"
        elif (
            data.get("trend_strength_score", 0) < -0.35
            and data.get("ema_alignment_score", 0) < 0
        ):
            return "bearish"
        elif positive_count >= BULLISH_THRESHOLD:
            return "bullish"
        elif positive_count <= BEARISH_THRESHOLD:
            return "bearish"
        else:
            return "neutral"

    def _build_risk_flags(self, data: Dict[str, Any]) -> list[str]:
        """Build compact flags for the AI prompt and logs."""
        flags = []
        if data.get("atr_percent", 0) >= 8 or data.get("volatility_7d", 0) >= 12:
            flags.append("high_volatility")
        if data.get("rsi_14", 50) >= 75 or data.get("mfi_14", 50) >= 80:
            flags.append("overbought")
        if data.get("rsi_14", 50) <= 25 or data.get("mfi_14", 50) <= 20:
            flags.append("oversold")
        if data.get("volume_24h_krw", 0) <= 0:
            flags.append("missing_turnover")
        if data.get("trend_strength_score", 0) < -0.5:
            flags.append("strong_downtrend")
        return flags

    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted composite score."""
        weighted_sum = sum(
            scores.get(indicator, 0) * self.indicator_weights.get(indicator, 0)
            for indicator in self.indicator_weights
        )
        return float(np.clip(weighted_sum, -1, 1))


# Create singleton instance
enhanced_market_analyzer = EnhancedMarketAnalyzer()
