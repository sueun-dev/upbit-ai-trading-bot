"""Risk Monitor for real-time portfolio risk assessment.

This module provides continuous monitoring of portfolio risks including
position sizes, drawdowns, correlations, and stop-loss triggers.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from src.infrastructure.config.settings import (
    EMERGENCY_STOP_LOSS,
    MAX_INVEST_RATIO_PER_COIN,
    MAX_TOTAL_INVEST_RATIO,
    STOP_LOSS_THRESHOLD,
)
from src.shared.openai_client import OpenAIClient

# Risk thresholds
DRAWDOWN_WARNING_THRESHOLD = 0.15
DRAWDOWN_CRITICAL_THRESHOLD = 0.25
CORRELATION_HIGH_THRESHOLD = 0.8

# Risk score thresholds
RISK_SCORE_CRITICAL = 0.8
RISK_SCORE_HIGH = 0.6
RISK_SCORE_MEDIUM = 0.4

# Risk calculation weights
POSITION_SIZE_RISK_WEIGHT = 0.3
LOSS_RISK_WEIGHT = 0.5
VOLATILITY_RISK_WEIGHT = 0.2

# Risk calculation limits
MAX_LOSS_FOR_RISK_CALC = 0.25  # 25% loss
MAX_VOLATILITY_FOR_RISK_CALC = 0.1  # 10% daily range

# Market conditions
FREEFALL_THRESHOLD = -10  # 10% daily drop
HIGH_LOSS_THRESHOLD = -10  # 10% loss
LOW_CASH_RATIO_THRESHOLD = 0.2

# Analysis limits
MAX_RECENT_TRADES_FOR_DRAWDOWN = 20
MIN_POSITIONS_FOR_CORRELATION = 2
PERCENTAGE_DIVISOR = 100

# Default values
DEFAULT_RISK_LEVEL = 'low'
DEFAULT_CORRELATION = 0
DEFAULT_DRAWDOWN = 0
DEFAULT_VOLATILITY = 0
DEFAULT_HERFINDAHL = 0
DEFAULT_CASH_RATIO = 1

# Risk levels
RISK_LEVEL_CRITICAL = 'critical'
RISK_LEVEL_HIGH = 'high'
RISK_LEVEL_MEDIUM = 'medium'
RISK_LEVEL_LOW = 'low'
RISK_LEVEL_UNKNOWN = 'unknown'

# Risk assessment messages
MSG_CRITICAL_RISK = "⚠️ CRITICAL RISK: Consider reducing positions immediately"
MSG_HIGH_RISK = "⚠️ HIGH RISK: Avoid new positions, consider partial profit-taking"
MSG_POSITION_CRITICAL = "🚨 {}: Critical risk - consider reducing position"
MSG_POSITION_HIGH_LOSS = "⚠️ {}: High loss risk - monitor for stop-loss"
MSG_REDUCE_EXPOSURE = "💰 Reduce total exposure (currently {:.1%})"
MSG_LOW_CASH = "💵 Low cash reserves - consider taking some profits"
MSG_RISK_ACCEPTABLE = "✅ Risk levels acceptable"
MSG_RISK_ERROR = "Unable to assess risk - proceed with caution"

# Trigger types
TRIGGER_EMERGENCY = 'emergency'
TRIGGER_REGULAR = 'regular'

# Recommended actions
ACTION_SELL_ALL = 'sell_all'
ACTION_PARTIAL_SELL = 'partial_sell'

# Asset types
ASSET_KRW = 'KRW'

# AI settings
AI_TEMPERATURE = 0.1

logger = logging.getLogger(__name__)


class RiskMonitor:
    """Real-time risk monitoring system for trading portfolios.
    
    Monitors:
    - Individual position risks
    - Portfolio-wide risk metrics
    - Drawdown levels
    - Correlation risks
    - Stop-loss triggers
    """
    
    def __init__(self) -> None:
        """Initialize the risk monitor."""
        self.openai_client = None  # Will be set by initialize()
        self.risk_thresholds = {
            'position_size': MAX_INVEST_RATIO_PER_COIN,
            'total_exposure': MAX_TOTAL_INVEST_RATIO,
            'drawdown_warning': DRAWDOWN_WARNING_THRESHOLD,
            'drawdown_critical': DRAWDOWN_CRITICAL_THRESHOLD,
            'correlation_high': CORRELATION_HIGH_THRESHOLD
        }
        logger.info("Risk Monitor initialized")
    
    # USED
    def initialize(self, api_key: str) -> None:
        """Initialize with API key.
        
        Args:
            api_key: OpenAI API key
        """
        self.openai_client = OpenAIClient(api_key=api_key)
    
    # USED
    def monitor_active_positions(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Dict[str, Any]],
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Monitor all active positions for risks.
        
        Args:
            portfolio: Current portfolio status.
            market_data: Current market data for all positions.
            recent_trades: Recent trading history.
            
        Returns:
            Comprehensive risk assessment.
        """
        try:
            # Calculate individual position risks
            position_risks = self._assess_position_risks(portfolio, market_data)
            
            # Calculate portfolio-wide metrics
            portfolio_metrics = self._calculate_portfolio_metrics(portfolio, market_data)
            
            # Assess drawdown risk
            drawdown_risk = self._assess_drawdown_risk(portfolio, recent_trades)
            
            # Check correlation risks
            correlation_risk = self._assess_correlation_risk(portfolio, market_data)
            
            # Determine overall risk level
            overall_risk = self._determine_overall_risk(
                position_risks, portfolio_metrics, drawdown_risk, correlation_risk
            )
            
            # Get AI risk assessment
            ai_assessment = self._get_ai_risk_assessment(
                portfolio, market_data, overall_risk
            )
            
            return self._build_risk_assessment_result(
                position_risks, portfolio_metrics, drawdown_risk,
                correlation_risk, overall_risk, ai_assessment
            )
            
        except Exception as e:
            logger.error(f"Risk monitoring failed: {e}")
            return {
                'error': error,
                'overall_risk': RISK_LEVEL_UNKNOWN,
                'recommendations': [MSG_RISK_ERROR]
            }
    
    # USED
    def check_stop_loss_triggers(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check for positions that should trigger stop-loss.
        
        Args:
            portfolio: Current portfolio status.
            market_data: Current market data.
            
        Returns:
            List of stop-loss triggers.
        """
        triggers = []
        
        assets = portfolio.get('assets', {})
        for symbol, asset in assets.items():
            if symbol == ASSET_KRW:
                continue
                
            profit_loss_pct = float(asset.get('profit_loss_percentage', 0)) / PERCENTAGE_DIVISOR
            
            # Check stop-loss thresholds
            trigger = self._check_position_stop_loss(
                symbol, asset, profit_loss_pct, market_data.get(symbol, {})
            )
            
            if trigger:
                triggers.append(trigger)
        
        return triggers
    
    # USED
    def _check_position_stop_loss(
        self,
        symbol: str,
        asset: Dict[str, Any],
        profit_loss_pct: float,
        market_info: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Check if position should trigger stop-loss.
        
        Args:
            symbol: Asset symbol.
            asset: Asset information.
            profit_loss_pct: Profit/loss percentage.
            market_info: Market information.
            
        Returns:
            Stop-loss trigger or None.
        """
        if profit_loss_pct <= EMERGENCY_STOP_LOSS:
            return self._create_stop_loss_trigger(
                symbol, profit_loss_pct, TRIGGER_EMERGENCY, ACTION_SELL_ALL,
                f'Emergency stop-loss triggered at {profit_loss_pct:.1%}'
            )
        elif profit_loss_pct <= STOP_LOSS_THRESHOLD:
            if self._should_trigger_stop_loss(asset, market_info):
                return self._create_stop_loss_trigger(
                    symbol, profit_loss_pct, TRIGGER_REGULAR, ACTION_PARTIAL_SELL,
                    f'Stop-loss triggered at {profit_loss_pct:.1%}'
                )
        return None
    
    # USED
    def _create_stop_loss_trigger(
        self,
        symbol: str,
        loss_pct: float,
        trigger_type: str,
        action: str,
        reason: str
    ) -> Dict[str, Any]:
        """Create stop-loss trigger dictionary.
        
        Args:
            symbol: Asset symbol.
            loss_pct: Loss percentage.
            trigger_type: Type of trigger.
            action: Recommended action.
            reason: Reason for trigger.
            
        Returns:
            Stop-loss trigger dictionary.
        """
        return {
            'symbol': symbol,
            'loss_pct': loss_pct * 100,
            'trigger_type': trigger_type,
            'recommended_action': action,
            'reason': reason
        }
    
    # USED
    def _assess_position_risks(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Assess risk for each position.
        
        Args:
            portfolio: Portfolio data.
            market_data: Market data.
            
        Returns:
            Position risks dictionary.
        """
        position_risks = {}
        total_balance = portfolio.get('total_balance', 0)
        
        for symbol, asset in portfolio.get('assets', {}).items():
            if symbol == ASSET_KRW:
                continue
                
            position_risk = self._analyze_single_position_risk(
                asset, total_balance, market_data.get(symbol, {})
            )
            position_risks[symbol] = position_risk
        
        return position_risks
    
    # USED
    def _analyze_single_position_risk(
        self,
        asset: Dict[str, Any],
        total_balance: float,
        market_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze risk for a single position.
        
        Args:
            asset: Asset information.
            total_balance: Total portfolio balance.
            market_info: Market information.
            
        Returns:
            Position risk analysis.
        """
        position_value = float(asset.get('current_value', 0))
        position_ratio = position_value / total_balance if total_balance > 0 else 0
        profit_loss_pct = float(asset.get('profit_loss_percentage', 0)) / PERCENTAGE_DIVISOR
        
        # Get market volatility
        volatility = self._calculate_volatility(market_info)
        
        # Calculate risk score
        risk_score = self._calculate_position_risk_score(
            position_ratio, profit_loss_pct, volatility
        )
        
        return {
            'position_ratio': position_ratio,
            'profit_loss_pct': profit_loss_pct * 100,
            'volatility': volatility,
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score)
        }
    
    # USED
    def _calculate_portfolio_metrics(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate portfolio-wide risk metrics.
        
        Args:
            portfolio: Portfolio data.
            market_data: Market data.
            
        Returns:
            Portfolio metrics dictionary.
        """
        total_balance = portfolio.get('total_balance', 0)
        krw_balance = portfolio.get('krw_balance', 0)
        
        # Calculate exposure
        exposure_ratio = (total_balance - krw_balance) / total_balance if total_balance > 0 else 0
        cash_ratio = krw_balance / total_balance if total_balance > 0 else DEFAULT_CASH_RATIO
        
        # Calculate concentration
        herfindahl_index = self._calculate_concentration_index(portfolio)
        
        # Calculate portfolio volatility
        portfolio_volatility = self._calculate_portfolio_volatility(
            portfolio, market_data
        )
        
        # Count positions
        position_count = len([a for a in portfolio.get('assets', {}) if a != ASSET_KRW])
        
        return {
            'total_balance': total_balance,
            'exposure_ratio': exposure_ratio,
            'cash_ratio': cash_ratio,
            'concentration_index': herfindahl_index,
            'portfolio_volatility': portfolio_volatility,
            'position_count': position_count
        }
    
    def _calculate_exposure_ratio(self, total_balance: float, krw_balance: float) -> float:
        """Calculate crypto exposure ratio.
        
        Args:
            total_balance: Total portfolio balance.
            krw_balance: KRW balance.
            
        Returns:
            Exposure ratio.
        """
        if total_balance <= 0:
            return 0
        total_crypto_value = total_balance - krw_balance
        return total_crypto_value / total_balance
    
    # USED
    def _calculate_concentration_index(self, portfolio: Dict[str, Any]) -> float:
        """Calculate Herfindahl concentration index.
        
        Args:
            portfolio: Portfolio data.
            
        Returns:
            Herfindahl index.
        """
        position_values = []
        for symbol, asset in portfolio.get('assets', {}).items():
            if symbol != ASSET_KRW:
                position_values.append(float(asset.get('current_value', 0)))
        
        if not position_values:
            return DEFAULT_HERFINDAHL
            
        total_value = sum(position_values)
        if total_value <= 0:
            return DEFAULT_HERFINDAHL
            
        position_shares = [v / total_value for v in position_values]
        return sum(share ** 2 for share in position_shares)

    # USED
    def _assess_drawdown_risk(
        self,
        portfolio: Dict[str, Any],
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess drawdown risk from recent trading.
        
        Args:
            portfolio: Portfolio data.
            recent_trades: Recent trades.
            
        Returns:
            Drawdown risk assessment.
        """
        if not recent_trades:
            return {
            'current_drawdown': DEFAULT_DRAWDOWN,
            'max_drawdown': DEFAULT_DRAWDOWN,
            'risk_level': DEFAULT_RISK_LEVEL
        }
        
        # Calculate cumulative values
        cumulative_values = self._calculate_cumulative_values(portfolio, recent_trades)
        
        # Calculate drawdown metrics
        peak, max_drawdown, current_drawdown = self._calculate_drawdown_metrics(
            cumulative_values
        )
        
        # Determine risk level
        if max_drawdown > self.risk_thresholds['drawdown_critical']:
            risk_level = RISK_LEVEL_CRITICAL
        elif max_drawdown > self.risk_thresholds['drawdown_warning']:
            risk_level = RISK_LEVEL_HIGH
        else:
            risk_level = RISK_LEVEL_LOW
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'peak_value': peak,
            'risk_level': risk_level
        }
    
    # USED
    def _calculate_cumulative_values(
        self,
        portfolio: Dict[str, Any],
        recent_trades: List[Dict[str, Any]]
    ) -> List[float]:
        """Calculate cumulative portfolio values.
        
        Args:
            portfolio: Portfolio data.
            recent_trades: Recent trades.
            
        Returns:
            List of cumulative values.
        """
        cumulative_values = []
        current_value = portfolio.get('total_balance', 0)
        
        # Work backwards from current value
        for trade in reversed(recent_trades[-MAX_RECENT_TRADES_FOR_DRAWDOWN:]):
            trade_impact = self._calculate_trade_impact(trade)
            current_value -= trade_impact
            cumulative_values.append(current_value)
        
        cumulative_values.reverse()
        cumulative_values.append(portfolio.get('total_balance', 0))
        
        return cumulative_values
    
    def _calculate_trade_impact(self, trade: Dict[str, Any]) -> float:
        """Calculate impact of a trade on portfolio value.
        
        Args:
            trade: Trade data.
            
        Returns:
            Trade impact amount.
        """
        amount_krw = float(trade.get('amount_krw', 0))
        multiplier = 1 if trade.get('action') == 'buy' else -1
        return amount_krw * multiplier
    
    # USED
    def _calculate_drawdown_metrics(
        self,
        cumulative_values: List[float]
    ) -> tuple[float, float, float]:
        """Calculate drawdown metrics.
        
        Args:
            cumulative_values: List of cumulative values.
            
        Returns:
            Tuple of (peak, max_drawdown, current_drawdown).
        """
        if not cumulative_values:
            return 0, 0, 0
            
        peak = cumulative_values[0]
        max_drawdown = 0
        
        for value in cumulative_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        current_drawdown = (peak - cumulative_values[-1]) / peak if peak > 0 else 0
        
        return peak, max_drawdown, current_drawdown
    
    # USED
    def _assess_correlation_risk(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess correlation risk between positions.
        
        Args:
            portfolio: Portfolio data.
            market_data: Market data.
            
        Returns:
            Correlation risk assessment.
        """
        positions = [s for s in portfolio.get('assets', {}).keys() if s != ASSET_KRW]
        
        if len(positions) < MIN_POSITIONS_FOR_CORRELATION:
            return {
            'max_correlation': DEFAULT_CORRELATION,
            'risk_level': DEFAULT_RISK_LEVEL,
            'correlated_pairs': []
        }
        
        # Calculate correlations
        correlations, correlated_pairs = self._calculate_position_correlations(
            positions, market_data
        )
        
        max_correlation = max(correlations) if correlations else DEFAULT_CORRELATION
        avg_correlation = np.mean(correlations) if correlations else DEFAULT_CORRELATION
        
        return {
            'max_correlation': max_correlation,
            'avg_correlation': avg_correlation,
            'risk_level': RISK_LEVEL_HIGH if max_correlation > self.risk_thresholds['correlation_high'] else RISK_LEVEL_LOW,
            'correlated_pairs': correlated_pairs
        }
    
    # USED
    def _calculate_position_correlations(
        self,
        positions: List[str],
        market_data: Dict[str, Dict[str, Any]]
    ) -> tuple[List[float], List[Dict[str, Any]]]:
        """Calculate correlations between positions.
        
        Args:
            positions: List of position symbols.
            market_data: Market data.
            
        Returns:
            Tuple of (correlations list, correlated pairs list).
        """
        correlations = []
        correlated_pairs = []
        
        for i, symbol1 in enumerate(positions):
            for j, symbol2 in enumerate(positions[i+1:], i+1):
                correlation = self._estimate_correlation(
                    symbol1, symbol2, market_data
                )
                correlations.append(correlation)
                
                if correlation > self.risk_thresholds['correlation_high']:
                    correlated_pairs.append({
                        'pair': f"{symbol1}-{symbol2}",
                        'correlation': correlation
                    })
        
        return correlations, correlated_pairs
    
    def _estimate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        market_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """Estimate correlation between two assets.
        
        Args:
            symbol1: First asset symbol.
            symbol2: Second asset symbol.
            market_data: Market data.
            
        Returns:
            Estimated correlation.
        """
        data1 = market_data.get(symbol1, {})
        data2 = market_data.get(symbol2, {})
        
        change1 = float(data1.get('price_change_24h', 0))
        change2 = float(data2.get('price_change_24h', 0))
        
        # Simple correlation estimate
        if change1 * change2 > 0:  # Same direction
            max_change = max(abs(change1), abs(change2), 1)
            min_change = min(abs(change1), abs(change2))
            return min_change / max_change
        else:
            return DEFAULT_CORRELATION
    
    
    
    # USED
    def _calculate_volatility(self, market_info: Dict[str, Any]) -> float:
        """Calculate volatility from market data.
        
        Args:
            market_info: Market information.
            
        Returns:
            Volatility value.
        """
        high = float(market_info.get('high_24h', 0))
        low = float(market_info.get('low_24h', 0))
        current = float(market_info.get('current_price', 1))
        
        if current > 0:
            return (high - low) / current
        return DEFAULT_VOLATILITY
    
    # USED
    def _calculate_portfolio_volatility(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate portfolio-wide volatility.
        
        Args:
            portfolio: Portfolio data.
            market_data: Market data.
            
        Returns:
            Portfolio volatility.
        """
        weighted_volatilities = []
        total_value = portfolio.get('total_balance', 0)
        
        if total_value <= 0:
            return DEFAULT_VOLATILITY
        
        for symbol, asset in portfolio.get('assets', {}).items():
            if symbol == ASSET_KRW:
                continue
                
            position_value = float(asset.get('current_value', 0))
            weight = position_value / total_value
            
            market_info = market_data.get(symbol, {})
            volatility = self._calculate_volatility(market_info)
            
            weighted_volatilities.append(weight * volatility)
        
        return sum(weighted_volatilities)
    
    # USED
    def _calculate_position_risk_score(
        self,
        position_ratio: float,
        profit_loss_pct: float,
        volatility: float
    ) -> float:
        """Calculate risk score for a position (0-1).
        
        Args:
            position_ratio: Position size ratio.
            profit_loss_pct: Profit/loss percentage.
            volatility: Volatility.
            
        Returns:
            Risk score.
        """
        # Position size risk
        size_risk = min(position_ratio / self.risk_thresholds['position_size'], 1)
        # Loss risk
        if profit_loss_pct < 0:
            loss_risk =  min(abs(profit_loss_pct) / MAX_LOSS_FOR_RISK_CALC, 1)
        else: 0
        # Volatility risk
        vol_risk = min(volatility / MAX_VOLATILITY_FOR_RISK_CALC, 1)
        
        # Combined risk score
        return (
            size_risk * POSITION_SIZE_RISK_WEIGHT +
            loss_risk * LOSS_RISK_WEIGHT +
            vol_risk * VOLATILITY_RISK_WEIGHT
        )
    
    # USED
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level.
        
        Args:
            risk_score: Risk score.
            
        Returns:
            Risk level string.
        """
        if risk_score >= RISK_SCORE_CRITICAL:
            return RISK_LEVEL_CRITICAL
        elif risk_score >= RISK_SCORE_HIGH:
            return RISK_LEVEL_HIGH
        elif risk_score >= RISK_SCORE_MEDIUM:
            return RISK_LEVEL_MEDIUM
        else:
            return RISK_LEVEL_LOW
    
    def _determine_overall_risk(
        self,
        position_risks: Dict[str, Dict[str, Any]],
        portfolio_metrics: Dict[str, float],
        drawdown_risk: Dict[str, Any],
        correlation_risk: Dict[str, Any]
    ) -> str:
        """Determine overall portfolio risk level.
        
        Args:
            position_risks: Position risk assessments.
            portfolio_metrics: Portfolio metrics.
            drawdown_risk: Drawdown risk assessment.
            correlation_risk: Correlation risk assessment.
            
        Returns:
            Overall risk level.
        """
        risk_factors = []
        
        # Check position risks
        risk_factors.extend(self._assess_position_risk_factors(position_risks))
        
        # Check exposure
        risk_factors.extend(self._assess_exposure_risk_factors(portfolio_metrics))
        
        # Add other risk factors
        risk_factors.append(drawdown_risk['risk_level'])
        risk_factors.append(correlation_risk['risk_level'])
        
        return self._aggregate_risk_factors(risk_factors)
    
    # USED
    def _assess_position_risk_factors(
        self,
        position_risks: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Assess risk factors from position risks.
        
        Args:
            position_risks: Position risk assessments.
            
        Returns:
            List of risk factors.
        """
        risk_factors = []
        high_risk_positions = sum(
            1 for p in position_risks.values()
            if p['risk_level'] in [RISK_LEVEL_HIGH, RISK_LEVEL_CRITICAL]
        )
        
        if high_risk_positions > 0:
            if high_risk_positions == 1:
                risk_factors.append(RISK_LEVEL_HIGH)
            else:
                risk_factors.append(RISK_LEVEL_CRITICAL)
                
        return risk_factors
    
    def _assess_exposure_risk_factors(
        self,
        portfolio_metrics: Dict[str, float]
    ) -> List[str]:
        """Assess risk factors from portfolio exposure.
        
        Args:
            portfolio_metrics: Portfolio metrics.
            
        Returns:
            List of risk factors.
        """
        risk_factors = []
        if portfolio_metrics['exposure_ratio'] > self.risk_thresholds['total_exposure']:
            risk_factors.append(RISK_LEVEL_HIGH)
        return risk_factors
    
    def _aggregate_risk_factors(self, risk_factors: List[str]) -> str:
        """Aggregate risk factors into overall risk level.
        
        Args:
            risk_factors: List of risk factors.
            
        Returns:
            Overall risk level.
        """
        if RISK_LEVEL_CRITICAL in risk_factors:
            return RISK_LEVEL_CRITICAL
        elif risk_factors.count(RISK_LEVEL_HIGH) >= 2:
            return RISK_LEVEL_HIGH
        elif RISK_LEVEL_HIGH in risk_factors:
            return RISK_LEVEL_MEDIUM
        else:
            return RISK_LEVEL_LOW
    
    # USED
    def _generate_recommendations(
        self,
        position_risks: Dict[str, Dict[str, Any]],
        portfolio_metrics: Dict[str, float],
        overall_risk: str
    ) -> List[str]:
        """Generate risk management recommendations.
        
        Args:
            position_risks: Position risk assessments.
            portfolio_metrics: Portfolio metrics.
            overall_risk: Overall risk level.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        # Add overall risk recommendations
        recommendations.extend(self._get_overall_risk_recommendations(overall_risk))
        
        # Add position-specific recommendations
        recommendations.extend(self._get_position_recommendations(position_risks))
        
        # Add exposure recommendations
        recommendations.extend(self._get_exposure_recommendations(portfolio_metrics))
        
        # Add cash recommendations
        recommendations.extend(self._get_cash_recommendations(portfolio_metrics))
        
        return recommendations if recommendations else [MSG_RISK_ACCEPTABLE]
    
    # USED
    def _get_overall_risk_recommendations(self, overall_risk: str) -> List[str]:
        """Get recommendations based on overall risk.
        
        Args:
            overall_risk: Overall risk level.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        if overall_risk == RISK_LEVEL_CRITICAL:
            recommendations.append(MSG_CRITICAL_RISK)
        elif overall_risk == RISK_LEVEL_HIGH:
            recommendations.append(MSG_HIGH_RISK)
        return recommendations
    
    # USED
    def _get_position_recommendations(
        self,
        position_risks: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Get position-specific recommendations.
        
        Args:
            position_risks: Position risk assessments.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        for symbol, risk in position_risks.items():
            if risk['risk_level'] == RISK_LEVEL_CRITICAL:
                recommendations.append(MSG_POSITION_CRITICAL.format(symbol))
            elif risk['risk_level'] == RISK_LEVEL_HIGH and risk['profit_loss_pct'] < HIGH_LOSS_THRESHOLD:
                recommendations.append(MSG_POSITION_HIGH_LOSS.format(symbol))
        return recommendations
    
    # USED
    def _get_exposure_recommendations(
        self,
        portfolio_metrics: Dict[str, float]
    ) -> List[str]:
        """Get exposure-related recommendations.
        
        Args:
            portfolio_metrics: Portfolio metrics.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        if portfolio_metrics['exposure_ratio'] > self.risk_thresholds['total_exposure']:
            recommendations.append(
                MSG_REDUCE_EXPOSURE.format(portfolio_metrics['exposure_ratio'])
            )
        return recommendations
    
    # USED
    def _get_cash_recommendations(
        self,
        portfolio_metrics: Dict[str, float]
    ) -> List[str]:
        """Get cash-related recommendations.
        
        Args:
            portfolio_metrics: Portfolio metrics.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        if portfolio_metrics['cash_ratio'] < LOW_CASH_RATIO_THRESHOLD:
            recommendations.append(MSG_LOW_CASH)
        return recommendations
    
    # USED
    def _should_trigger_stop_loss(
        self,
        asset: Dict[str, Any],
        market_info: Dict[str, Any]
    ) -> bool:
        """Determine if stop-loss should be triggered.
        
        Args:
            asset: Asset information.
            market_info: Market information.
            
        Returns:
            True if stop-loss should be triggered.
        """
        # Check if market is in freefall
        price_change_24h = float(market_info.get('price_change_24h', 0))
        if price_change_24h < FREEFALL_THRESHOLD:
            return True
        
        # Default to trigger if threshold is met
        return True
    
    # USED
    def _get_ai_risk_assessment(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Dict[str, Any]],
        overall_risk: str
    ) -> str:
        """Get AI assessment of portfolio risk.
        
        Args:
            portfolio: Portfolio data.
            market_data: Market data.
            overall_risk: Overall risk level.
            
        Returns:
            AI risk assessment string.
        """
        try:
            positions_summary = self._prepare_positions_summary(portfolio)
            system_message = """You are a risk management expert. Provide a brief, actionable assessment of portfolio risk in 1-2 sentences."""
            prompt = self._create_ai_prompt(portfolio, overall_risk, positions_summary)
            
            result = self.openai_client.analyze_with_prompt(
                prompt=prompt,
                system_message=system_message,
                temperature=AI_TEMPERATURE
            )
            
            return result.get('response', 'Risk assessment unavailable')
            
        except Exception as e:
            logger.error(f"AI risk assessment failed: {e}")
            return "AI risk assessment unavailable"
    
    # USED
    def _prepare_positions_summary(self, portfolio: Dict[str, Any]) -> List[str]:
        """Prepare positions summary for AI.
        
        Args:
            portfolio: Portfolio data.
            
        Returns:
            List of position summaries.
        """
        positions = []
        for symbol, asset in portfolio.get('assets', {}).items():
            if symbol == ASSET_KRW:
                continue
            positions.append(
                f"{symbol}: {asset.get('current_value', 0):,.0f} KRW "
                f"({asset.get('profit_loss_percentage', 0):.1f}%)"
            )
        return positions
    
    # USED
    def _create_ai_prompt(
        self,
        portfolio: Dict[str, Any],
        overall_risk: str,
        positions: List[str]
    ) -> str:
        """Create prompt for AI risk assessment.
        
        Args:
            portfolio: Portfolio data.
            overall_risk: Overall risk level.
            positions: Position summaries.
            
        Returns:
            Prompt string.
        """
        return f"""Assess this portfolio risk:
        
            Overall Risk Level: {overall_risk}
            Total Value: {portfolio.get('total_balance', 0):,.0f} KRW
            Cash Available: {portfolio.get('krw_balance', 0):,.0f} KRW
            Positions: {', '.join(positions)}

            Provide a brief risk assessment and key action if needed.
        """
    
    # USED
    def _build_risk_assessment_result(
        self,
        position_risks: Dict[str, Dict[str, Any]],
        portfolio_metrics: Dict[str, float],
        drawdown_risk: Dict[str, Any],
        correlation_risk: Dict[str, Any],
        overall_risk: str,
        ai_assessment: str
    ) -> Dict[str, Any]:
        """Build risk assessment result.
        
        Args:
            position_risks: Position risk assessments.
            portfolio_metrics: Portfolio metrics.
            drawdown_risk: Drawdown risk assessment.
            correlation_risk: Correlation risk assessment.
            overall_risk: Overall risk level.
            ai_assessment: AI assessment.
            
        Returns:
            Complete risk assessment result.
        """
        high_risk_positions = len([
            p for p in position_risks.values()
            if p['risk_level'] in [RISK_LEVEL_HIGH, RISK_LEVEL_CRITICAL]
        ])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'position_risks': position_risks,
            'portfolio_metrics': portfolio_metrics,
            'drawdown_risk': drawdown_risk,
            'correlation_risk': correlation_risk,
            'overall_risk': overall_risk,
            'high_risk_positions': high_risk_positions,
            'recommendations': self._generate_recommendations(
                position_risks, portfolio_metrics, overall_risk
            ),
            'ai_assessment': ai_assessment
        }


# Create singleton instance
risk_monitor = RiskMonitor()