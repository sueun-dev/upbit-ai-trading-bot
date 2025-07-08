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
VOLUME_INCREASE_THRESHOLD = 1.2
VOLUME_DECREASE_THRESHOLD = 0.8
VOLUME_NORMALIZATION_FACTOR = 2

# Volatility constants
HIGH_VOLATILITY_THRESHOLD = 0.1
LOW_VOLATILITY_THRESHOLD = 0.05
VOLATILITY_NORMALIZATION_FACTOR = 10

# Trend analysis constants
TREND_NORMALIZATION_FACTOR = 10
BULLISH_THRESHOLD = 3  # At least 3 positive timeframes
BEARISH_THRESHOLD = 1  # At most 1 positive timeframe
TREND_CONSISTENCY_NEUTRAL = 0.5
TREND_CONSISTENCY_FULL = 1.0

# Signal generation thresholds
STRONG_BUY_THRESHOLD = 0.5
BUY_THRESHOLD = 0.2
STRONG_SELL_THRESHOLD = -0.5
SELL_THRESHOLD = -0.2

# Confidence adjustment factors
VOLUME_CONFIRMED_MULTIPLIER = 1.2
VOLUME_UNCONFIRMED_MULTIPLIER = 0.8

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
    
    Provides advanced market analysis including:
    - Multi-timeframe trend analysis
    - Technical indicator confluence
    - Volume profile analysis
    - Market regime detection
    - Support/resistance identification
    """
    
    def __init__(self) -> None:
        """Initialize the enhanced market analyzer."""
        self.indicator_weights = DEFAULT_INDICATOR_WEIGHTS.copy()
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive market analysis.
        
        Args:
            data: Market data dictionary containing price, volume, and indicator data
            
        Returns:
            Comprehensive analysis results
        """
        try:
            return self._perform_analysis(data)
        except Exception as e:
            raise Exception(f"Unexpected error in EnhancedMarketAnalyzer") from e
    
    def _perform_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual market analysis.
        
        Args:
            data: Market data containing prices, volumes, and indicators.
            
        Returns:
            Comprehensive analysis results.
            
        Raises:
            AnalysisError: If analysis fails.
        """
        try:
            # Extract basic market data
            market_metrics = self._extract_market_metrics(data)
            
            # Extract technical indicators
            indicators = self._extract_technical_indicators(data)
            
            # Perform multi-aspect analysis
            analysis_results = self._perform_multi_aspect_analysis(data, indicators)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score({
                'trend': analysis_results['trend']['score'],
                'momentum': analysis_results['momentum']['score'],
                'volume': analysis_results['volume']['score'],
                'volatility': analysis_results['volatility']['score']
            })
            
            # Generate trading signal
            signal = self._generate_signal(
                composite_score,
                analysis_results['trend'],
                analysis_results['momentum']
            )
            
            return self._build_analysis_result(
                market_metrics,
                analysis_results,
                composite_score,
                signal
            )
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            raise Exception(f"Failed to analyze market data: {e}")
    
    def _extract_market_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic market metrics from data.
        
        Args:
            data: Raw market data.
            
        Returns:
            Dictionary of market metrics.
        """
        return {
            'current_price': float(data.get('current_price', 0)),
            'volume_24h': float(data.get('volume_24h', 0)),
            'price_change_24h': float(data.get('price_change_24h', 0))
        }
    
    def _extract_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract technical indicators from data.
        
        Args:
            data: Raw market data.
            
        Returns:
            Dictionary of technical indicators.
        """
        return {
            'rsi': data.get('rsi', {}),
            'macd': data.get('macd', {}),
            'bollinger': data.get('bollinger_bands', {})
        }
    
    def _perform_multi_aspect_analysis(
        self,
        data: Dict[str, Any],
        indicators: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform analysis across multiple aspects.
        
        Args:
            data: Market data.
            indicators: Technical indicators.
            
        Returns:
            Dictionary containing all analysis results.
        """
        return {
            'trend': self._analyze_trend(data),
            'momentum': self._analyze_momentum(indicators['rsi'], indicators['macd']),
            'volume': self._analyze_volume(data),
            'volatility': self._analyze_volatility(indicators['bollinger'], data),
            'support_resistance': self._identify_support_resistance(data)
        }
    
    def _build_analysis_result(
        self,
        market_metrics: Dict[str, float],
        analysis_results: Dict[str, Dict[str, Any]],
        composite_score: float,
        signal: str
    ) -> Dict[str, Any]:
        """Build final analysis result.
        
        Args:
            market_metrics: Basic market metrics.
            analysis_results: All analysis results.
            composite_score: Calculated composite score.
            signal: Generated trading signal.
            
        Returns:
            Complete analysis result.
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'current_price': market_metrics['current_price'],
            'price_change_24h': market_metrics['price_change_24h'],
            'volume_24h': market_metrics['volume_24h'],
            'trend': analysis_results['trend'],
            'momentum': analysis_results['momentum'],
            'volume': analysis_results['volume'],
            'volatility': analysis_results['volatility'],
            'support_resistance': analysis_results['support_resistance'],
            'composite_score': composite_score,
            'signal': signal,
            'confidence': self._calculate_confidence(
                composite_score,
                analysis_results['volume']
            )
        }
    
    def _analyze_trend(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price trend across multiple timeframes."""
        price_changes = {
            '1h': data.get('price_change_1h', 0),
            '24h': data.get('price_change_24h', 0),
            '7d': data.get('price_change_7d', 0),
            '30d': data.get('price_change_30d', 0)
        }
        
        # Calculate trend direction
        positive_count = sum(1 for change in price_changes.values() if change > 0)
        trend_direction = self._determine_trend_direction(positive_count)
        
        # Calculate trend consistency
        consistency = self._calculate_trend_consistency(price_changes.values())
        
        # Calculate trend score
        trend_score = self._calculate_trend_score(price_changes.values())
        
        return {
            'direction': trend_direction,
            'timeframes': price_changes,
            'consistency': consistency,
            'score': trend_score,
            'strength': abs(trend_score)
        }
    
    def _analyze_momentum(self, rsi: Dict[str, Any], macd: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators.
        
        Args:
            rsi: RSI indicator values.
            macd: MACD indicator values.
            
        Returns:
            Momentum analysis results.
        """
        # Analyze RSI
        rsi_analysis = self._analyze_rsi(rsi)
        
        # Analyze MACD
        macd_analysis = self._analyze_macd(macd)
        
        # Calculate combined momentum score
        momentum_score = self._calculate_momentum_score(
            rsi_analysis['score'],
            macd_analysis['score']
        )
        
        return {
            'rsi': rsi_analysis,
            'macd': macd_analysis,
            'score': momentum_score,
            'strength': abs(momentum_score)
        }
    
    def _analyze_volume(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume patterns and trends."""
        volume_24h = float(data.get('volume_24h', 0))
        avg_volume = float(data.get('volume_avg_7d', volume_24h))
        volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
        
        # Determine volume trend
        volume_trend = self._determine_volume_trend(volume_ratio)
        
        # Check volume confirmation
        price_change = data.get('price_change_24h', 0)
        volume_confirms = self._check_volume_confirmation(price_change, volume_ratio)
        
        # Calculate volume score
        volume_score = self._calculate_volume_score(volume_ratio)
        
        return {
            'current': volume_24h,
            'average': avg_volume,
            'ratio': volume_ratio,
            'trend': volume_trend,
            'confirms_price': volume_confirms,
            'score': volume_score
        }
    
    def _analyze_volatility(self, bollinger: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market volatility."""
        upper_band = bollinger.get('upper', 0)
        lower_band = bollinger.get('lower', 0)
        middle_band = bollinger.get('middle', 0)
        current_price = float(data.get('current_price', 0))
        
        # Calculate band width
        band_width = self._calculate_band_width(upper_band, lower_band, middle_band)
        
        # Calculate price position within bands
        price_position = self._calculate_price_position(
            current_price, upper_band, lower_band
        )
        
        # Calculate volatility score
        volatility_score = self._calculate_volatility_score(band_width)
        
        return {
            'band_width': band_width,
            'price_position': price_position,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'score': volatility_score,
            'level': self._determine_volatility_level(band_width)
        }
    
    def _identify_support_resistance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify key support and resistance levels."""
        current_price = float(data.get('current_price', 0))
        high_24h = float(data.get('high_24h', current_price))
        low_24h = float(data.get('low_24h', current_price))
        high_7d = float(data.get('high_7d', high_24h))
        low_7d = float(data.get('low_7d', low_24h))
        
        # Simple support/resistance identification
        resistance_levels = sorted([high_24h, high_7d])
        support_levels = sorted([low_24h, low_7d])
        
        # Find nearest levels
        nearest_resistance = self._find_nearest_resistance(
            resistance_levels, current_price, high_7d
        )
        nearest_support = self._find_nearest_support(
            support_levels, current_price, low_7d
        )
        
        # Calculate distances as percentages
        resistance_distance = self._calculate_price_distance(
            nearest_resistance, current_price
        )
        support_distance = self._calculate_price_distance(
            current_price, nearest_support
        )
        
        return {
            'resistance': {
                'nearest': nearest_resistance,
                'distance_pct': resistance_distance,
                'levels': resistance_levels
            },
            'support': {
                'nearest': nearest_support,
                'distance_pct': support_distance,
                'levels': support_levels
            }
        }
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted composite score."""
        weighted_sum = sum(
            scores.get(indicator, 0) * self.indicator_weights.get(indicator, 0)
            for indicator in self.indicator_weights
        )
        return np.clip(weighted_sum, -1, 1)
    
    def _generate_signal(
        self,
        composite_score: float,
        trend: Dict[str, Any],
        momentum: Dict[str, Any]
    ) -> str:
        """Generate trading signal based on analysis.
        
        Args:
            composite_score: Composite analysis score.
            trend: Trend analysis results.
            momentum: Momentum analysis results.
            
        Returns:
            Trading signal string.
        """
        # Check composite score thresholds
        if composite_score > STRONG_BUY_THRESHOLD:
            return 'strong_buy'
        elif composite_score > BUY_THRESHOLD:
            return 'buy'
        elif composite_score < STRONG_SELL_THRESHOLD:
            return 'strong_sell'
        elif composite_score < SELL_THRESHOLD:
            return 'sell'
        
        # Check for specific patterns
        return self._check_pattern_signals(trend, momentum)
    
    def _calculate_confidence(self, composite_score: float, volume: Dict[str, Any]) -> float:
        """Calculate confidence level for the analysis."""
        base_confidence = abs(composite_score)
        
        # Adjust for volume confirmation
        confidence = self._adjust_confidence_for_volume(
            base_confidence,
            volume['confirms_price']
        )
        
        return np.clip(confidence, 0, 1)
    
    def analyze_market_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Public method for comprehensive market analysis.
        
        Args:
            market_data: Market data to analyze.
            
        Returns:
            Comprehensive analysis results.
        """
        return self.analyze(market_data)
    
    # Helper methods for trend analysis
    def _determine_trend_direction(self, positive_count: int) -> str:
        """Determine trend direction based on positive timeframe count."""
        if positive_count >= BULLISH_THRESHOLD:
            return 'bullish'
        elif positive_count <= BEARISH_THRESHOLD:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_trend_consistency(self, changes: Any) -> float:
        """Calculate trend consistency across timeframes."""
        changes_list = list(changes)
        if all(c > 0 for c in changes_list) or all(c < 0 for c in changes_list):
            return TREND_CONSISTENCY_FULL
        return TREND_CONSISTENCY_NEUTRAL
    
    def _calculate_trend_score(self, changes: Any) -> float:
        """Calculate normalized trend score."""
        avg_change = np.mean(list(changes))
        return np.tanh(avg_change / TREND_NORMALIZATION_FACTOR)
    
    # Helper methods for momentum analysis
    def _analyze_rsi(self, rsi: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze RSI indicator."""
        rsi_14 = rsi.get('rsi_14', RSI_NEUTRAL)
        rsi_30 = rsi.get('rsi_30', RSI_NEUTRAL)
        
        signal = self._determine_rsi_signal(rsi_14)
        score = (rsi_14 - RSI_NEUTRAL) / RSI_NEUTRAL
        
        return {
            'value': rsi_14,
            'signal': signal,
            'divergence': abs(rsi_14 - rsi_30),
            'score': score
        }
    
    def _determine_rsi_signal(self, rsi_value: float) -> str:
        """Determine RSI signal."""
        if rsi_value < RSI_OVERSOLD:
            return 'oversold'
        elif rsi_value > RSI_OVERBOUGHT:
            return 'overbought'
        else:
            return 'neutral'
    
    def _analyze_macd(self, macd: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MACD indicator."""
        macd_line = macd.get('macd', 0)
        signal_line = macd.get('signal', 0)
        histogram = macd.get('histogram', 0)
        
        signal = 'bullish' if macd_line > signal_line and histogram > 0 else 'bearish'
        score = np.tanh(histogram / MACD_HISTOGRAM_NORMALIZATION)
        
        return {
            'value': macd_line,
            'signal': signal,
            'histogram': histogram,
            'score': score
        }
    
    def _calculate_momentum_score(self, rsi_score: float, macd_score: float) -> float:
        """Calculate combined momentum score."""
        return (rsi_score * RSI_WEIGHT + macd_score * MACD_WEIGHT)
    
    # Helper methods for volume analysis
    def _determine_volume_trend(self, volume_ratio: float) -> str:
        """Determine volume trend."""
        if volume_ratio > VOLUME_INCREASE_THRESHOLD:
            return 'increasing'
        elif volume_ratio < VOLUME_DECREASE_THRESHOLD:
            return 'decreasing'
        else:
            return 'stable'
    
    def _check_volume_confirmation(self, price_change: float, volume_ratio: float) -> bool:
        """Check if volume confirms price movement."""
        return (price_change > 0 and volume_ratio > 1) or (price_change < 0 and volume_ratio > 1)
    
    def _calculate_volume_score(self, volume_ratio: float) -> float:
        """Calculate volume score."""
        return np.tanh((volume_ratio - 1) * VOLUME_NORMALIZATION_FACTOR)
    
    # Helper methods for volatility analysis
    def _calculate_band_width(self, upper: float, lower: float, middle: float) -> float:
        """Calculate Bollinger Band width."""
        if middle > 0:
            return (upper - lower) / middle
        return 0
    
    def _calculate_price_position(self, price: float, upper: float, lower: float) -> float:
        """Calculate price position within bands."""
        if upper > lower:
            return (price - lower) / (upper - lower)
        return 0.5
    
    def _calculate_volatility_score(self, band_width: float) -> float:
        """Calculate volatility score (inverse relationship)."""
        return 1 - np.tanh(band_width * VOLATILITY_NORMALIZATION_FACTOR)
    
    def _determine_volatility_level(self, band_width: float) -> str:
        """Determine volatility level."""
        if band_width > HIGH_VOLATILITY_THRESHOLD:
            return 'high'
        elif band_width < LOW_VOLATILITY_THRESHOLD:
            return 'low'
        else:
            return 'medium'
    
    # Helper methods for support/resistance
    def _find_nearest_resistance(self, levels: list, current_price: float, default: float) -> float:
        """Find nearest resistance level above current price."""
        return min((r for r in levels if r > current_price), default=default)
    
    def _find_nearest_support(self, levels: list, current_price: float, default: float) -> float:
        """Find nearest support level below current price."""
        return max((s for s in levels if s < current_price), default=default)
    
    def _calculate_price_distance(self, price1: float, price2: float) -> float:
        """Calculate percentage distance between prices."""
        if price2 > 0:
            return ((price1 - price2) / price2) * 100
        return 0
    
    # Helper methods for signal generation
    def _check_pattern_signals(self, trend: Dict[str, Any], momentum: Dict[str, Any]) -> str:
        """Check for specific pattern-based signals."""
        if trend['direction'] == 'bullish' and momentum['rsi']['signal'] == 'oversold':
            return 'buy'
        elif trend['direction'] == 'bearish' and momentum['rsi']['signal'] == 'overbought':
            return 'sell'
        else:
            return 'neutral'
    
    # Helper methods for confidence calculation
    def _adjust_confidence_for_volume(self, base_confidence: float, volume_confirms: bool) -> float:
        """Adjust confidence based on volume confirmation."""
        if volume_confirms:
            return base_confidence * VOLUME_CONFIRMED_MULTIPLIER
        else:
            return base_confidence * VOLUME_UNCONFIRMED_MULTIPLIER


# Create singleton instance
enhanced_market_analyzer = EnhancedMarketAnalyzer()