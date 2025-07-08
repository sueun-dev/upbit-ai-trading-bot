"""Enhanced Market Analyzer for comprehensive market analysis.

This module provides advanced market analysis capabilities including
technical indicators, trend analysis, and market sentiment evaluation.
"""

import logging
from datetime import datetime
from typing import Any, Dict

import numpy as np


# Technical analysis constants
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_NEUTRAL = 50

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
    'trend': 0.3,
    'momentum': 0.25,
    'volume': 0.25,
    'volatility': 0.2
}

# MACD normalization
MACD_HISTOGRAM_NORMALIZATION = 100

# Score weights for momentum
RSI_WEIGHT = 0.5
MACD_WEIGHT = 0.5

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
            
            # Calculate composite score
            composite_score = self._calculate_composite_score({
                'trend': trend_score,
                'momentum': momentum_score,
                'volume': volume_score,
                'volatility': volatility_score
            })
            
            # Generate trading signal
            signal = self._generate_signal_simple(composite_score)
            
            # Calculate confidence
            confidence = abs(composite_score)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': data.get('symbol', 'UNKNOWN'),
                'current_price': data.get('current_price', 0),
                'composite_score': composite_score,
                'signal': signal,
                'confidence': confidence,
                'trend': self._get_trend_direction(data),
                'scores': {
                    'trend': trend_score,
                    'momentum': momentum_score,
                    'volume': volume_score,
                    'volatility': volatility_score
                }
            }
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            raise Exception(f"Failed to analyze market data: {e}")
    
    def _calculate_trend_score_simple(self, data: Dict[str, Any]) -> float:
        """Calculate trend score from price changes."""
        # Use already calculated price changes
        price_changes = [
            data.get('price_1h_change', 0),
            data.get('price_24h_change', 0),
            data.get('price_7d_change', 0),
            data.get('price_30d_change', 0)
        ]
        
        # Average price change normalized
        avg_change = np.mean(price_changes)
        return np.tanh(avg_change / TREND_NORMALIZATION_FACTOR)
    
    def _calculate_momentum_score_simple(self, data: Dict[str, Any]) -> float:
        """Calculate momentum score from RSI and MACD."""
        # RSI score (0 at 50, positive when oversold, negative when overbought)
        rsi_14 = data.get('rsi_14', 50)
        rsi_score = (50 - rsi_14) / 50  # Inverted: low RSI = positive score
        
        # MACD score from histogram
        macd_histogram = data.get('macd_histogram', 0)
        macd_score = np.tanh(macd_histogram / MACD_HISTOGRAM_NORMALIZATION)
        
        return (rsi_score * RSI_WEIGHT + macd_score * MACD_WEIGHT)
    
    def _calculate_volume_score_simple(self, data: Dict[str, Any]) -> float:
        """Calculate volume score from volume ratio."""
        # Use pre-calculated volume ratio
        volume_ratio = data.get('volume_ratio_24h_7d', 1.0)
        return np.tanh((volume_ratio - 1) * VOLUME_NORMALIZATION_FACTOR)
    
    def _calculate_volatility_score_simple(self, data: Dict[str, Any]) -> float:
        """Calculate volatility score (lower volatility = higher score)."""
        # Use pre-calculated volatility
        volatility = data.get('volatility_7d', 0)
        return 1 - np.tanh(volatility * VOLATILITY_NORMALIZATION_FACTOR / 10)
    
    def _generate_signal_simple(self, composite_score: float) -> str:
        """Generate trading signal based on composite score."""
        if composite_score > STRONG_BUY_THRESHOLD:
            return 'strong_buy'
        elif composite_score > BUY_THRESHOLD:
            return 'buy'
        elif composite_score < STRONG_SELL_THRESHOLD:
            return 'strong_sell'
        elif composite_score < SELL_THRESHOLD:
            return 'sell'
        else:
            return 'neutral'
    
    def _get_trend_direction(self, data: Dict[str, Any]) -> str:
        """Get trend direction from price changes."""
        price_changes = [
            data.get('price_1h_change', 0),
            data.get('price_24h_change', 0),
            data.get('price_7d_change', 0),
            data.get('price_30d_change', 0)
        ]
        positive_count = sum(1 for change in price_changes if change > 0)
        
        if positive_count >= BULLISH_THRESHOLD:
            return 'bullish'
        elif positive_count <= BEARISH_THRESHOLD:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted composite score."""
        weighted_sum = sum(
            scores.get(indicator, 0) * self.indicator_weights.get(indicator, 0)
            for indicator in self.indicator_weights
        )
        return np.clip(weighted_sum, -1, 1)


# Create singleton instance
enhanced_market_analyzer = EnhancedMarketAnalyzer()