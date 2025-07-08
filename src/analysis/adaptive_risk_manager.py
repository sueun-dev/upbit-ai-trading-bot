"""Adaptive Risk Manager for dynamic position sizing and risk management.

This module provides adaptive risk management capabilities that adjust
position sizes based on market conditions, portfolio performance, and AI confidence.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from src.infrastructure.config.settings import (
    DEFAULT_BUY_AMOUNT_KRW,
    MAX_INVEST_RATIO_PER_COIN,
    MAX_TOTAL_INVEST_RATIO,
    MIN_ORDER_KRW,
)

# Risk management constants
CONSERVATIVE_KELLY_FRACTION = 0.25
MIN_POSITION_SIZE_RATIO = 0.01  # 1% minimum
WIN_RATE_WINDOW = 20  # Last 20 trades for win rate calculation
PORTFOLIO_HEAT_DECAY_RATE = 0.9
MAX_PORTFOLIO_HEAT = 0.8
CIRCUIT_BREAKER_SIZE_MULTIPLIER = 0.5
MIN_TRADES_FOR_CIRCUIT_BREAKER = 10
MIN_TRADES_FOR_STATISTICAL_SIGNIFICANCE = 20

# Volatility thresholds
HIGH_VOLATILITY_THRESHOLD = 0.15
MEDIUM_VOLATILITY_THRESHOLD = 0.10
NORMAL_VOLATILITY_THRESHOLD = 0.05

# Performance thresholds
MIN_WIN_RATE = 0.3
MAX_DRAWDOWN = 0.15
MAX_CONSECUTIVE_LOSSES = 3
MAX_DAILY_LOSS = 0.10

# Position sizing multipliers
HIGH_VOLATILITY_MULTIPLIER = 0.7
MEDIUM_VOLATILITY_MULTIPLIER = 0.85
NORMAL_VOLATILITY_MULTIPLIER = 1.0
LOW_VOLATILITY_MULTIPLIER = 1.1
AI_REJECTION_MULTIPLIER = 0.7

logger = logging.getLogger(__name__)


class AdaptiveRiskManager:
    """Adaptive risk management system for dynamic position sizing.
    
    This class implements advanced risk management strategies including:
    - Kelly Criterion-based position sizing
    - Dynamic volatility adjustment
    - Portfolio heat management
    - Circuit breaker integration
    - Confidence-weighted sizing
    
    Attributes:
        portfolio_heat: Current portfolio heat level (0.0 to 1.0).
    """
    
    def __init__(self) -> None:
        """Initialize the adaptive risk manager."""
        self.portfolio_heat = 0.0
        logger.info("Adaptive Risk Manager initialized")
    
    def calculate_optimal_position_size(
        self,
        decision: Dict[str, Any],
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate optimal position size using adaptive Kelly Criterion.
        
        Args:
            decision: Trading decision with confidence score.
            portfolio: Current portfolio status including balances.
            market_data: Market data for the symbol.
            validation_result: Multi-AI validation result.
            
        Returns:
            Dict containing:
                - recommended_size: Recommended position size in KRW
                - position_ratio: Position size as ratio of portfolio
                - Various adjustment factors and reasoning
        """
        try:
            confidence = decision.get('confidence', 0.5)
            ai_approved = validation_result.get('approved', True)
            
            # Calculate position size through various adjustments
            base_size = self._calculate_base_size(portfolio)
            kelly_adjusted = self._apply_kelly_criterion(confidence, base_size)
            volatility_adjusted = self._adjust_for_volatility(kelly_adjusted, market_data)
            heat_adjusted = self._adjust_for_portfolio_heat(volatility_adjusted, portfolio)
            ai_adjusted = self._apply_ai_validation_adjustment(heat_adjusted, ai_approved)
            final_size = self._apply_position_limits(ai_adjusted, portfolio)
            
            # Convert to KRW amount
            recommended_krw = self._convert_to_krw_amount(final_size, portfolio)
            
            return self._build_sizing_result(
                recommended_krw=recommended_krw,
                final_size=final_size,
                base_size=base_size,
                kelly_adjusted=kelly_adjusted,
                volatility_adjusted=volatility_adjusted,
                heat_adjusted=heat_adjusted,
                ai_adjusted=ai_adjusted,
                confidence=confidence,
                market_data=market_data,
                portfolio=portfolio
            )
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            return self._get_default_sizing_result()
    
    def apply_circuit_breaker_limits(
        self,
        recommended_size: float,
        portfolio: Dict[str, Any],
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply circuit breaker limits to position size.
        
        Args:
            recommended_size: Recommended position size in KRW.
            portfolio: Current portfolio status.
            recent_trades: Recent trading history.
            
        Returns:
            Dict containing adjusted size and circuit breaker status.
        """
        recent_performance = self._calculate_recent_performance(recent_trades)
        circuit_status = self._check_circuit_conditions(recent_performance)
        
        adjusted_size = (
            recommended_size * CIRCUIT_BREAKER_SIZE_MULTIPLIER
            if circuit_status['triggered']
            else recommended_size
        )
        
        return {
            'original_size': recommended_size,
            'adjusted_size': adjusted_size,
            'circuit_breaker_triggered': circuit_status['triggered'],
            'reason': circuit_status['reason'],
            'recent_win_rate': recent_performance['win_rate'],
            'recent_drawdown': recent_performance['drawdown']
        }
    
    def _calculate_base_size(self, portfolio: Dict[str, Any]) -> float:
        """Calculate base position size based on portfolio cash ratio.
        
        Args:
            portfolio: Portfolio data with balance information.
            
        Returns:
            Base position size as ratio of portfolio.
        """
        total_balance = portfolio.get('total_balance', 0)
        krw_balance = portfolio.get('krw_balance', 0)
        
        if total_balance <= 0:
            return 0.03  # Default 3%
        
        cash_ratio = krw_balance / total_balance
        
        # More cash available = larger base position size
        if cash_ratio > 0.5:
            return 0.05  # 5% when lots of cash
        elif cash_ratio > 0.3:
            return 0.04  # 4% moderate cash
        else:
            return 0.03  # 3% when low on cash
    
    def _apply_kelly_criterion(self, confidence: float, base_size: float) -> float:
        """Apply Kelly Criterion for optimal position sizing.
        
        Uses simplified Kelly formula for 1:1 odds:
        Kelly fraction = 2p - 1, where p is probability of winning.
        
        Args:
            confidence: AI confidence score (0.0 to 1.0).
            base_size: Base position size ratio.
            
        Returns:
            Kelly-adjusted position size ratio.
        """
        kelly_fraction = 2 * confidence - 1
        conservative_kelly = kelly_fraction * CONSERVATIVE_KELLY_FRACTION
        
        if conservative_kelly <= 0:
            return base_size * 0.5  # Reduce size for low confidence
        
        return base_size * (1 + conservative_kelly)
    
    def _adjust_for_volatility(self, size: float, market_data: Dict[str, Any]) -> float:
        """Adjust position size based on market volatility.
        
        Higher volatility = smaller position size.
        
        Args:
            size: Current position size ratio.
            market_data: Market data including price range.
            
        Returns:
            Volatility-adjusted position size ratio.
        """
        volatility = self._calculate_volatility(market_data)
        multiplier = self._get_volatility_multiplier(volatility)
        return size * multiplier
    
    def _adjust_for_portfolio_heat(self, size: float, portfolio: Dict[str, Any]) -> float:
        """Adjust size based on portfolio heat (recent activity level).
        
        Args:
            size: Current position size ratio.
            portfolio: Portfolio data with asset information.
            
        Returns:
            Heat-adjusted position size ratio.
        """
        self._update_portfolio_heat(portfolio)
        
        if self.portfolio_heat > MAX_PORTFOLIO_HEAT:
            heat_reduction = 1 - (self.portfolio_heat - MAX_PORTFOLIO_HEAT)
            return size * max(0.5, heat_reduction)
        
        return size
    
    def _apply_ai_validation_adjustment(self, size: float, ai_approved: bool) -> float:
        """Apply adjustment based on AI validation result.
        
        Args:
            size: Current position size ratio.
            ai_approved: Whether AI validators approved the trade.
            
        Returns:
            AI-adjusted position size ratio.
        """
        return size * (1.0 if ai_approved else AI_REJECTION_MULTIPLIER)
    
    def _apply_position_limits(self, size: float, portfolio: Dict[str, Any]) -> float:
        """Apply minimum/maximum position limits.
        
        Args:
            size: Proposed position size ratio.
            portfolio: Portfolio data for exposure calculation.
            
        Returns:
            Limited position size ratio.
        """
        # Apply min/max position size
        size = max(MIN_POSITION_SIZE_RATIO, min(size, MAX_INVEST_RATIO_PER_COIN))
        
        # Check total exposure limit
        current_exposure = self._calculate_current_exposure(portfolio)
        if current_exposure + size > MAX_TOTAL_INVEST_RATIO:
            size = max(0, MAX_TOTAL_INVEST_RATIO - current_exposure)
        
        return size
    
    def _convert_to_krw_amount(self, size_ratio: float, portfolio: Dict[str, Any]) -> float:
        """Convert position size ratio to KRW amount.
        
        Args:
            size_ratio: Position size as ratio of portfolio.
            portfolio: Portfolio data with balance information.
            
        Returns:
            Position size in KRW.
        """
        total_balance = portfolio.get('total_balance', 0)
        krw_balance = portfolio.get('krw_balance', 0)
        
        desired_krw = size_ratio * total_balance
        available_krw = min(desired_krw, krw_balance)
        
        return max(MIN_ORDER_KRW, available_krw)
    
    def _update_portfolio_heat(self, portfolio: Dict[str, Any]) -> None:
        """Update portfolio heat metric based on position count.
        
        Heat increases with number of positions and decays over time.
        
        Args:
            portfolio: Portfolio data with asset information.
        """
        position_count = self._count_active_positions(portfolio)
        position_heat = position_count * 0.1
        
        # Decay existing heat and add new heat
        self.portfolio_heat = min(
            1.0,
            self.portfolio_heat * PORTFOLIO_HEAT_DECAY_RATE + position_heat
        )
    
    def _count_active_positions(self, portfolio: Dict[str, Any]) -> int:
        """Count number of active crypto positions.
        
        Args:
            portfolio: Portfolio data with asset information.
            
        Returns:
            Number of non-KRW positions with positive balance.
        """
        assets = portfolio.get('assets', {})
        return sum(
            1 for symbol, asset in assets.items()
            if symbol != 'KRW' and asset.get('balance', 0) > 0
        )
    
    def _calculate_current_exposure(self, portfolio: Dict[str, Any]) -> float:
        """Calculate current crypto exposure as ratio of portfolio.
        
        Args:
            portfolio: Portfolio data with balance information.
            
        Returns:
            Crypto exposure ratio (0.0 to 1.0).
        """
        total_balance = portfolio.get('total_balance', 0)
        krw_balance = portfolio.get('krw_balance', 0)
        
        if total_balance <= 0:
            return 0
        
        crypto_value = total_balance - krw_balance
        return crypto_value / total_balance
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate 24h volatility from market data.
        
        Args:
            market_data: Market data with high/low/current prices.
            
        Returns:
            Volatility as decimal (e.g., 0.1 for 10%).
        """
        high = market_data.get('high_24h', 0)
        low = market_data.get('low_24h', 0)
        current = market_data.get('current_price', 1)
        
        if current > 0:
            return (high - low) / current
        return 0.1  # Default 10%
    
    def _get_volatility_multiplier(self, volatility: float) -> float:
        """Get position size multiplier based on volatility level.
        
        Args:
            volatility: Calculated volatility value.
            
        Returns:
            Multiplier for position size adjustment.
        """
        if volatility > HIGH_VOLATILITY_THRESHOLD:
            return HIGH_VOLATILITY_MULTIPLIER
        elif volatility > MEDIUM_VOLATILITY_THRESHOLD:
            return MEDIUM_VOLATILITY_MULTIPLIER
        elif volatility > NORMAL_VOLATILITY_THRESHOLD:
            return NORMAL_VOLATILITY_MULTIPLIER
        else:
            return LOW_VOLATILITY_MULTIPLIER
    
    def _calculate_recent_performance(
        self,
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate recent trading performance metrics.
        
        Args:
            recent_trades: List of recent trade records.
            
        Returns:
            Dict with performance metrics including win rate and drawdown.
        """
        if not recent_trades:
            return self._get_default_performance_metrics()
        
        window_trades = recent_trades[-WIN_RATE_WINDOW:]
        
        win_rate = self._calculate_win_rate(window_trades)
        drawdown = self._calculate_max_drawdown(window_trades)
        avg_return = self._calculate_average_return(window_trades)
        
        return {
            'win_rate': win_rate,
            'drawdown': drawdown,
            'avg_return': avg_return,
            'trade_count': len(recent_trades),
            'trades': window_trades
        }
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trades.
        
        Args:
            trades: List of trade records with profit/loss.
            
        Returns:
            Win rate as decimal (0.0 to 1.0).
        """
        if not trades:
            return 0.5
        
        wins = sum(1 for t in trades if t.get('profit_loss', 0) > 0)
        return wins / len(trades)
    
    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown from trades.
        
        Args:
            trades: List of trade records with profit/loss.
            
        Returns:
            Maximum drawdown as decimal (e.g., 0.15 for 15%).
        """
        if not trades:
            return 0
        
        cumulative_returns = []
        running_total = 0
        
        for trade in trades:
            running_total += trade.get('profit_loss', 0)
            cumulative_returns.append(running_total)
        
        if not cumulative_returns:
            return 0
        
        peak = cumulative_returns[0]
        max_drawdown = 0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / abs(peak) if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_average_return(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate average return from trades.
        
        Args:
            trades: List of trade records with profit/loss.
            
        Returns:
            Average profit/loss per trade.
        """
        if not trades:
            return 0
        
        returns = [t.get('profit_loss', 0) for t in trades]
        return np.mean(returns)
    
    def _check_circuit_conditions(
        self,
        recent_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if circuit breaker conditions are met.
        
        Args:
            recent_performance: Performance metrics from recent trades.
            
        Returns:
            Dict with triggered status and reason.
        """
        trade_count = recent_performance.get('trade_count', 0)
        
        # Skip circuit breaker for initial trades
        if trade_count < MIN_TRADES_FOR_CIRCUIT_BREAKER:
            return {
                'triggered': False,
                'reason': f'Initial trading phase ({trade_count}/{MIN_TRADES_FOR_CIRCUIT_BREAKER} trades)'
            }
        
        triggered, reasons = self._evaluate_circuit_conditions(recent_performance)
        
        return {
            'triggered': triggered,
            'reason': '; '.join(reasons) if reasons else 'Normal conditions'
        }
    
    def _evaluate_circuit_conditions(
        self,
        performance: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Evaluate specific circuit breaker conditions.
        
        Args:
            performance: Recent performance metrics.
            
        Returns:
            Tuple of (triggered, list of reasons).
        """
        triggered = False
        reasons = []
        
        # Check win rate (only with sufficient trades)
        if (performance['win_rate'] < MIN_WIN_RATE and 
            performance['trade_count'] >= MIN_TRADES_FOR_STATISTICAL_SIGNIFICANCE):
            triggered = True
            reasons.append(f"Low win rate: {performance['win_rate']:.1%}")
        
        # Check drawdown
        if performance['drawdown'] > MAX_DRAWDOWN:
            triggered = True
            reasons.append(f"High drawdown: {performance['drawdown']:.1%}")
        
        # Check consecutive losses
        consecutive_losses = self._count_consecutive_losses(performance.get('trades', []))
        if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            triggered = True
            reasons.append(f"Consecutive losses: {consecutive_losses}")
        
        # Check daily loss
        daily_loss = self._calculate_daily_loss(performance.get('trades', []))
        if daily_loss > MAX_DAILY_LOSS:
            triggered = True
            reasons.append(f"Daily loss exceeds limit: {daily_loss:.1%}")
        
        return triggered, reasons
    
    def _count_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
        """Count consecutive losses from most recent trades.
        
        Args:
            trades: List of trade records.
            
        Returns:
            Number of consecutive losses.
        """
        if not trades:
            return 0
        
        consecutive = 0
        for trade in reversed(trades):
            if trade.get('profit_loss', 0) < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _calculate_daily_loss(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate today's loss percentage.
        
        Args:
            trades: List of trade records with timestamps.
            
        Returns:
            Daily loss as decimal (e.g., 0.1 for 10%).
        """
        if not trades:
            return 0
        
        today = datetime.now().date()
        today_loss = 0
        
        # Get portfolio value at start of day from first trade or current value
        start_of_day_value = self._get_start_of_day_portfolio_value(trades, today)
        if start_of_day_value <= 0:
            # Fallback to 1M KRW if unable to determine
            logger.warning("Unable to determine start of day portfolio value, using 1M KRW")
            start_of_day_value = 1_000_000
        
        for trade in trades:
            trade_date = self._parse_trade_date(trade.get('timestamp', ''))
            if trade_date == today:
                today_loss += trade.get('profit_loss', 0)
        
        return abs(today_loss) / start_of_day_value if today_loss < 0 else 0
    
    def _get_start_of_day_portfolio_value(self, trades: List[Dict[str, Any]], target_date: datetime.date) -> float:
        """Get portfolio value at start of day.
        
        Args:
            trades: List of trade records.
            target_date: The date to get start value for.
            
        Returns:
            Portfolio value at start of day.
        """
        # Find first trade of the day to estimate starting value
        for trade in sorted(trades, key=lambda x: x.get('timestamp', '')):
            trade_date = self._parse_trade_date(trade.get('timestamp', ''))
            if trade_date == target_date:
                # Use the portfolio value before this trade
                return trade.get('portfolio_value_before', 0)
        
        # If no trades today, try to get current portfolio value
        try:
            from src.shared.utils.data_store import get_latest_portfolio_status
            status = get_latest_portfolio_status()
            if status:
                return status.get('total_balance', 0)
        except Exception as e:
            logger.warning(f"Failed to get portfolio status: {e}")
        
        return 0
    
    def _parse_trade_date(self, timestamp: str) -> datetime.date:
        """Parse date from trade timestamp.
        
        Args:
            timestamp: ISO format timestamp string.
            
        Returns:
            Date object or None if parsing fails.
        """
        if not isinstance(timestamp, str):
            return None
        
        try:
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
        except (ValueError, AttributeError):
            return None
    
    def _generate_sizing_reasoning(
        self,
        confidence: float,
        market_data: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for position sizing.
        
        Args:
            confidence: AI confidence score.
            market_data: Market data for volatility.
            portfolio: Portfolio data for cash position.
            
        Returns:
            Reasoning string explaining size adjustments.
        """
        reasons = []
        
        # Confidence reasoning
        if confidence > 0.8:
            reasons.append("High confidence trade")
        elif confidence < 0.5:
            reasons.append("Low confidence - reduced size")
        
        # Volatility reasoning
        volatility = self._calculate_volatility(market_data)
        if volatility > HIGH_VOLATILITY_THRESHOLD:
            reasons.append("High volatility - reduced size")
        elif volatility < NORMAL_VOLATILITY_THRESHOLD:
            reasons.append("Low volatility - normal size")
        
        # Portfolio heat reasoning
        if self.portfolio_heat > MAX_PORTFOLIO_HEAT:
            reasons.append("Portfolio heat high - cooling period")
        
        # Cash position reasoning
        total_balance = portfolio.get('total_balance', 1)
        krw_balance = portfolio.get('krw_balance', 0)
        cash_ratio = krw_balance / total_balance if total_balance > 0 else 0
        
        if cash_ratio < 0.2:
            reasons.append("Low cash reserves")
        
        return "; ".join(reasons) if reasons else "Standard position sizing applied"
    
    def _build_sizing_result(
        self,
        recommended_krw: float,
        final_size: float,
        base_size: float,
        kelly_adjusted: float,
        volatility_adjusted: float,
        heat_adjusted: float,
        ai_adjusted: float,
        confidence: float,
        market_data: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build complete sizing result dictionary.
        
        Args:
            Various sizing values and adjustments.
            
        Returns:
            Complete sizing result with all metrics.
        """
        return {
            'recommended_size': recommended_krw,
            'position_ratio': final_size,
            'base_size': base_size,
            'kelly_adjustment': kelly_adjusted / base_size if base_size > 0 else 1,
            'volatility_adjustment': volatility_adjusted / kelly_adjusted if kelly_adjusted > 0 else 1,
            'heat_adjustment': heat_adjusted / volatility_adjusted if volatility_adjusted > 0 else 1,
            'ai_adjustment': ai_adjusted / heat_adjusted if heat_adjusted > 0 else 1,
            'reasoning': self._generate_sizing_reasoning(confidence, market_data, portfolio)
        }
    
    def _get_default_sizing_result(self) -> Dict[str, Any]:
        """Get default sizing result for error cases.
        
        Returns:
            Default sizing result dictionary.
        """
        return {
            'recommended_size': DEFAULT_BUY_AMOUNT_KRW,
            'position_ratio': 0.03,  # Default 3%
            'base_size': 0.03,
            'kelly_adjustment': 1.0,
            'volatility_adjustment': 1.0,
            'heat_adjustment': 1.0,
            'ai_adjustment': 1.0,
            'reasoning': 'Using default sizing due to calculation error'
        }
    
    def _get_default_performance_metrics(self) -> Dict[str, Any]:
        """Get default performance metrics when no trades available.
        
        Returns:
            Default performance metrics dictionary.
        """
        return {
            'win_rate': 0.5,
            'drawdown': 0,
            'avg_return': 0,
            'trade_count': 0,
            'trades': []
        }


# Create singleton instance
adaptive_risk_manager = AdaptiveRiskManager()