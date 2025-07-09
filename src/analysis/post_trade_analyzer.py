"""Post-Trade Analyzer for analyzing completed trades and stop-loss decisions.

This module provides comprehensive post-trade analysis to learn from
both successful and unsuccessful trades, especially stop-loss executions.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from src.analysis.pattern_learner import PatternLearner
from src.shared.openai_client import OpenAIClient

# Severity thresholds for loss assessment
SEVERITY_MINOR_THRESHOLD = -0.05      # -5% loss
SEVERITY_MODERATE_THRESHOLD = -0.10   # -10% loss
SEVERITY_SEVERE_THRESHOLD = -0.15     # -15% loss
SEVERITY_CRITICAL_THRESHOLD = -0.25   # -25% loss

# Market sentiment thresholds
PANIC_SELLING_PRICE_THRESHOLD = -10  # Price drop threshold for panic
PANIC_SELLING_VOLUME_THRESHOLD = 2   # Volume multiplier for panic
BEARISH_PRICE_THRESHOLD = -5
BULLISH_PRICE_THRESHOLD = 5

# Support breach threshold
SUPPORT_BREACH_THRESHOLD = 0.98  # 2% below support

# Time constants
SECONDS_PER_HOUR = 3600
QUICK_LOSS_HOURS = 24  # Losses within 24 hours are "quick"

# AI analysis temperature
AI_ANALYSIS_TEMPERATURE = 0.2

# Lesson and recommendation limits
MAX_LESSONS_PER_ANALYSIS = 5
MAX_RECOMMENDATIONS = 4
MAX_WARNING_SIGNS_TO_DISPLAY = 3

# Performance thresholds
GOOD_RETURN_THRESHOLD = 5    # 5% return
POOR_RETURN_THRESHOLD = -5   # -5% loss
HIGH_SLIPPAGE_THRESHOLD = 1  # 1% slippage
MODERATE_SLIPPAGE_THRESHOLD = 0.5
DRAWDOWN_THRESHOLD = -0.1    # -10% drawdown

# Execution quality thresholds
RSI_OVERSOLD_THRESHOLD = 40
RSI_OVERBOUGHT_THRESHOLD = 60

# Volume confirmation threshold
VOLUME_CONFIRMATION_THRESHOLD = 1.5

# Trading rating score components
WIN_SCORE = 40
GOOD_RETURN_SCORE = 20
POOR_RETURN_PENALTY = -20
GOOD_EXECUTION_SCORE = 30
AVERAGE_EXECUTION_SCORE = 15
RISK_ADJUSTED_SCORE = 10

# Trading rating thresholds
EXCELLENT_RATING_THRESHOLD = 80
GOOD_RATING_THRESHOLD = 60
AVERAGE_RATING_THRESHOLD = 40
POOR_RATING_THRESHOLD = 20

# Default values
DEFAULT_VOLATILITY = 0.1  # 10% assumed volatility
DEFAULT_RSI = 50

# Severity levels
SEVERITY_CRITICAL = 'critical'
SEVERITY_SEVERE = 'severe'
SEVERITY_MODERATE = 'moderate'
SEVERITY_MINOR = 'minor'
SEVERITY_NEGLIGIBLE = 'negligible'
SEVERITY_UNKNOWN = 'unknown'

# Market sentiments
SENTIMENT_PANIC_SELLING = 'panic_selling'
SENTIMENT_BEARISH = 'bearish'
SENTIMENT_BULLISH = 'bullish'
SENTIMENT_NEUTRAL = 'neutral'

# Execution quality levels
QUALITY_GOOD = 'good'
QUALITY_AVERAGE = 'average'
QUALITY_NEUTRAL = 'neutral'
QUALITY_NORMAL = 'normal'

# Trade ratings
RATING_EXCELLENT = 'excellent'
RATING_GOOD = 'good'
RATING_AVERAGE = 'average'
RATING_POOR = 'poor'
RATING_VERY_POOR = 'very_poor'
RATING_UNKNOWN = 'unknown'

logger = logging.getLogger(__name__)


class PostTradeAnalyzer:
    """Post-trade analysis system for learning and improvement.
    
    Features:
    - Stop-loss decision analysis
    - Trade outcome evaluation
    - Failure severity assessment
    - Lesson extraction
    - Performance metrics calculation
    """
    
    def __init__(self, api_key: str) -> None:
        """Initialize the post-trade analyzer.
        
        Args:
            api_key: OpenAI API key
        """
        self.openai_client = OpenAIClient(api_key=api_key)
        self.pattern_learner = PatternLearner(api_key=api_key)
        self.severity_thresholds = {
            SEVERITY_MINOR: SEVERITY_MINOR_THRESHOLD,
            SEVERITY_MODERATE: SEVERITY_MODERATE_THRESHOLD,
            SEVERITY_SEVERE: SEVERITY_SEVERE_THRESHOLD,
            SEVERITY_CRITICAL: SEVERITY_CRITICAL_THRESHOLD
        }
    
    def analyze_stop_loss_decision(
        self,
        symbol: str,
        stop_loss_action: str,
        original_trade_data: Dict[str, Any],
        current_market_data: Dict[str, Any],
        portfolio_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a stop-loss decision after execution.
        
        Args:
            symbol: Trading symbol.
            stop_loss_action: Action taken (sell_all, partial_sell).
            original_trade_data: Original buy trade information.
            current_market_data: Market data at stop-loss execution.
            portfolio_context: Portfolio state at execution.
            
        Returns:
            Comprehensive analysis of the stop-loss decision.
        """
        try:
            # Calculate trade metrics
            trade_metrics = self._calculate_trade_metrics(
                original_trade_data, current_market_data, stop_loss_action
            )
            
            # Assess failure severity
            severity = self._assess_failure_severity(trade_metrics)
            
            # Analyze market conditions
            market_analysis = self._analyze_market_conditions_at_stop(
                symbol, current_market_data
            )
            
            # Get AI analysis of the decision
            ai_analysis = self._get_ai_stop_loss_analysis(
                symbol, trade_metrics, market_analysis, severity
            )
            
            # Extract lessons learned
            lessons = self._extract_lessons_learned(
                symbol, trade_metrics, market_analysis, ai_analysis
            )
            
            # Record pattern for future learning
            self._record_stop_loss_pattern(
                symbol, stop_loss_action, trade_metrics, market_analysis
            )
            
            return self._build_stop_loss_analysis_result(
                symbol, stop_loss_action, trade_metrics, severity,
                market_analysis, ai_analysis, lessons
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze stop-loss decision: {e}")
            return self._build_error_result(str(e))
    
    def _calculate_trade_metrics(
        self,
        original_trade: Dict[str, Any],
        current_market: Dict[str, Any],
        action: str
    ) -> Dict[str, Any]:
        """Calculate metrics for the trade.
        
        Args:
            original_trade: Original trade data.
            current_market: Current market data.
            action: Action taken.
            
        Returns:
            Dictionary of trade metrics.
        """
        entry_price = float(original_trade.get('price', 0))
        exit_price = float(current_market.get('current_price', 0))
        quantity = float(original_trade.get('quantity', 0))
        
        # Calculate price-based metrics
        price_metrics = self._calculate_price_metrics(
            entry_price, exit_price, quantity
        )
        
        # Calculate time-based metrics
        hold_hours = self._calculate_hold_duration(original_trade)
        
        return {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'loss_percentage': price_metrics['loss_percentage'],
            'loss_amount': price_metrics['loss_amount'],
            'hold_duration_hours': hold_hours,
            'exit_action': action,
            'price_decline_from_entry': price_metrics['loss_percentage'] * 100
        }
    
    def _calculate_price_metrics(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float
    ) -> Dict[str, float]:
        """Calculate price-based metrics.
        
        Args:
            entry_price: Entry price.
            exit_price: Exit price.
            quantity: Trade quantity.
            
        Returns:
            Dictionary with loss percentage and amount.
        """
        if entry_price > 0:
            loss_percentage = (exit_price - entry_price) / entry_price
            loss_amount = (exit_price - entry_price) * quantity
        else:
            loss_percentage = 0
            loss_amount = 0
            
        return {
            'loss_percentage': loss_percentage,
            'loss_amount': loss_amount
        }
    
    def _calculate_hold_duration(self, original_trade: Dict[str, Any]) -> float:
        """Calculate how long the position was held.
        
        Args:
            original_trade: Original trade data.
            
        Returns:
            Duration in hours.
        """
        try:
            entry_time = datetime.fromisoformat(
                original_trade.get('timestamp', datetime.now().isoformat())
            )
            hold_duration = datetime.now() - entry_time
            return hold_duration.total_seconds() / SECONDS_PER_HOUR
        except Exception:
            return 0
    
    def _assess_failure_severity(self, trade_metrics: Dict[str, Any]) -> str:
        """Assess the severity of the trading failure.
        
        Args:
            trade_metrics: Trade metrics dictionary.
            
        Returns:
            Severity level string.
        """
        loss_pct = trade_metrics.get('loss_percentage', 0)
        
        if loss_pct <= self.severity_thresholds[SEVERITY_CRITICAL]:
            return SEVERITY_CRITICAL
        elif loss_pct <= self.severity_thresholds[SEVERITY_SEVERE]:
            return SEVERITY_SEVERE
        elif loss_pct <= self.severity_thresholds[SEVERITY_MODERATE]:
            return SEVERITY_MODERATE
        elif loss_pct <= self.severity_thresholds[SEVERITY_MINOR]:
            return SEVERITY_MINOR
        else:
            return SEVERITY_NEGLIGIBLE
    
    def _analyze_market_conditions_at_stop(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze market conditions at stop-loss execution.
        
        Args:
            symbol: Trading symbol.
            market_data: Market data dictionary.
            
        Returns:
            Market analysis dictionary.
        """
        return {
            'price_trend_24h': market_data.get('price_24h_change', 0),
            'volume_ratio': market_data.get('volume_ratio', 1),
            'rsi': market_data.get('rsi', {}).get('rsi_14', DEFAULT_RSI),
            'volatility': self._calculate_volatility(market_data),
            'market_sentiment': self._assess_market_sentiment(market_data),
            'support_breached': self._check_support_breach(symbol, market_data)
        }
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate current volatility.
        
        Args:
            market_data: Market data dictionary.
            
        Returns:
            Volatility as a decimal.
        """
        high = market_data.get('high_24h', 0)
        low = market_data.get('low_24h', 0)
        current = market_data.get('current_price', 1)
        
        if current > 0:
            return (high - low) / current
        return DEFAULT_VOLATILITY
    
    def _assess_market_sentiment(self, market_data: Dict[str, Any]) -> str:
        """Assess overall market sentiment.
        
        Args:
            market_data: Market data dictionary.
            
        Returns:
            Sentiment string.
        """
        price_change = market_data.get('price_24h_change', 0)
        volume_ratio = market_data.get('volume_ratio', 1)
        
        if (price_change < PANIC_SELLING_PRICE_THRESHOLD and 
            volume_ratio > PANIC_SELLING_VOLUME_THRESHOLD):
            return SENTIMENT_PANIC_SELLING
        elif price_change < BEARISH_PRICE_THRESHOLD:
            return SENTIMENT_BEARISH
        elif price_change > BULLISH_PRICE_THRESHOLD:
            return SENTIMENT_BULLISH
        else:
            return SENTIMENT_NEUTRAL
    
    def _check_support_breach(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """Check if key support levels were breached.
        
        Args:
            symbol: Trading symbol (unused currently).
            market_data: Market data dictionary.
            
        Returns:
            True if support was breached.
        """
        current_price = market_data.get('current_price', 0)
        low_24h = market_data.get('low_24h', current_price)
        
        return current_price < low_24h * SUPPORT_BREACH_THRESHOLD
    
    def _get_ai_stop_loss_analysis(
        self,
        symbol: str,
        trade_metrics: Dict[str, Any],
        market_analysis: Dict[str, Any],
        severity: str
    ) -> Dict[str, Any]:
        """Get AI analysis of the stop-loss decision.
        
        Args:
            symbol: Trading symbol.
            trade_metrics: Trade metrics.
            market_analysis: Market analysis.
            severity: Severity level.
            
        Returns:
            AI analysis results.
        """
        try:
            system_message = self._build_ai_system_message()
            prompt = self._build_ai_analysis_prompt(
                symbol, trade_metrics, market_analysis, severity
            )
            
            result = self.openai_client.analyze_with_prompt(
                prompt=prompt,
                system_message=system_message,
                temperature=AI_ANALYSIS_TEMPERATURE
            )
            
            return result
            
        except Exception as e:
            logger.error(f"AI stop-loss analysis failed: {e}")
            return self._get_default_ai_analysis()
    
    def _build_ai_system_message(self) -> str:
        """Build system message for AI analysis.
        
        Returns:
            System message string.
        """
        return """You are a trading post-mortem analyst. Analyze stop-loss 
        decisions to determine if they were justified and what could be learned. 
        Be objective and focus on actionable insights."""
    
    def _build_ai_analysis_prompt(
        self,
        symbol: str,
        trade_metrics: Dict[str, Any],
        market_analysis: Dict[str, Any],
        severity: str
    ) -> str:
        """Build prompt for AI analysis.
        
        Args:
            symbol: Trading symbol.
            trade_metrics: Trade metrics.
            market_analysis: Market analysis.
            severity: Severity level.
            
        Returns:
            Prompt string.
        """
        return f"""Analyze this stop-loss execution:
        
Symbol: {symbol}
Loss: {trade_metrics['loss_percentage']:.1%}
Hold Duration: {trade_metrics['hold_duration_hours']:.1f} hours
Severity: {severity}

Market Conditions at Stop:
- 24h Price Change: {market_analysis['price_trend_24h']:.1f}%
- Volume Ratio: {market_analysis['volume_ratio']:.1f}x
- RSI: {market_analysis['rsi']}
- Market Sentiment: {market_analysis['market_sentiment']}
- Support Breached: {market_analysis['support_breached']}

Questions to answer:
1. Was the stop-loss decision justified given the conditions?
2. What were the key warning signs that led to this loss?
3. Could this loss have been avoided or minimized?
4. What should be done differently next time?

Provide analysis in JSON format with keys: 
decision_quality, warning_signs, avoidable, key_lesson"""
    
    def _get_default_ai_analysis(self) -> Dict[str, Any]:
        """Get default AI analysis for error cases.
        
        Returns:
            Default analysis dictionary.
        """
        return {
            'decision_quality': SEVERITY_UNKNOWN,
            'warning_signs': [],
            'avoidable': SEVERITY_UNKNOWN,
            'key_lesson': 'Analysis unavailable'
        }
    
    def _extract_lessons_learned(
        self,
        symbol: str,
        trade_metrics: Dict[str, Any],
        market_analysis: Dict[str, Any],
        ai_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract actionable lessons from the analysis.
        
        Args:
            symbol: Trading symbol (unused).
            trade_metrics: Trade metrics.
            market_analysis: Market analysis.
            ai_analysis: AI analysis results.
            
        Returns:
            List of lesson strings.
        """
        lessons = []
        
        # Add severity-based lessons
        lessons.extend(self._get_severity_lessons(trade_metrics))
        
        # Add market condition lessons
        lessons.extend(self._get_market_condition_lessons(market_analysis))
        
        # Add timing lessons
        lessons.extend(self._get_timing_lessons(trade_metrics))
        
        # Add AI-generated lessons
        lessons.extend(self._get_ai_lessons(ai_analysis))
        
        return lessons[:MAX_LESSONS_PER_ANALYSIS]
    
    def _get_severity_lessons(self, trade_metrics: Dict[str, Any]) -> List[str]:
        """Get lessons based on loss severity.
        
        Args:
            trade_metrics: Trade metrics.
            
        Returns:
            List of lesson strings.
        """
        lessons = []
        if trade_metrics['loss_percentage'] <= -0.20:
            lessons.append("Critical losses indicate need for tighter stop-loss levels")
        return lessons
    
    def _get_market_condition_lessons(self, market_analysis: Dict[str, Any]) -> List[str]:
        """Get lessons based on market conditions.
        
        Args:
            market_analysis: Market analysis.
            
        Returns:
            List of lesson strings.
        """
        lessons = []
        
        if market_analysis['market_sentiment'] == SENTIMENT_PANIC_SELLING:
            lessons.append("Avoid entering positions during panic selling events")
        
        if market_analysis['support_breached']:
            lessons.append("Respect support levels - exit when key supports break")
            
        return lessons
    
    def _get_timing_lessons(self, trade_metrics: Dict[str, Any]) -> List[str]:
        """Get lessons based on timing.
        
        Args:
            trade_metrics: Trade metrics.
            
        Returns:
            List of lesson strings.
        """
        lessons = []
        if trade_metrics['hold_duration_hours'] < QUICK_LOSS_HOURS:
            lessons.append("Quick losses suggest poor entry timing - improve entry analysis")
        return lessons
    
    def _get_ai_lessons(self, ai_analysis: Dict[str, Any]) -> List[str]:
        """Get lessons from AI analysis.
        
        Args:
            ai_analysis: AI analysis results.
            
        Returns:
            List of lesson strings.
        """
        lessons = []
        
        # Add main AI lesson
        ai_lesson = ai_analysis.get('key_lesson', '')
        if ai_lesson and isinstance(ai_lesson, str):
            lessons.append(ai_lesson)
        
        # Add warning signs lesson
        warning_signs = ai_analysis.get('warning_signs', [])
        if warning_signs and isinstance(warning_signs, list):
            signs_str = ', '.join(warning_signs[:MAX_WARNING_SIGNS_TO_DISPLAY])
            lessons.append(f"Watch for: {signs_str}")
            
        return lessons
    
    def _record_stop_loss_pattern(
        self,
        symbol: str,
        action: str,
        trade_metrics: Dict[str, Any],
        market_analysis: Dict[str, Any]
    ) -> None:
        """Record stop-loss pattern for pattern learner.
        
        Args:
            symbol: Trading symbol.
            action: Action taken.
            trade_metrics: Trade metrics.
            market_analysis: Market analysis.
        """
        try:
            market_conditions = self._prepare_market_conditions(market_analysis)
            outcome = self._prepare_outcome_data(symbol, trade_metrics)
            
            self.pattern_learner.analyze_trade_patterns(
                symbol, action, market_conditions, outcome
            )
            
        except Exception as e:
            logger.error(f"Failed to record stop-loss pattern: {e}")
    
    def _prepare_market_conditions(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare market conditions for pattern learner.
        
        Args:
            market_analysis: Market analysis data.
            
        Returns:
            Market conditions dictionary.
        """
        return {
            'price_change_24h': market_analysis['price_trend_24h'],
            'volume_ratio': market_analysis['volume_ratio'],
            'rsi': {'rsi_14': market_analysis['rsi']},
            'volatility': market_analysis['volatility'],
            'regime': market_analysis['market_sentiment']
        }
    
    def _prepare_outcome_data(
        self,
        symbol: str,
        trade_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare outcome data for pattern learner.
        
        Args:
            symbol: Trading symbol.
            trade_metrics: Trade metrics.
            
        Returns:
            Outcome dictionary.
        """
        return {
            'trade_id': f"stop_loss_{symbol}_{datetime.now().timestamp()}",
            'profit_loss': trade_metrics['loss_amount'],
            'profit_loss_percentage': trade_metrics['loss_percentage'] * 100
        }
    
    def _generate_future_recommendations(
        self,
        severity: str,
        lessons: List[str]
    ) -> List[str]:
        """Generate recommendations for future trades.
        
        Args:
            severity: Severity level.
            lessons: List of lessons learned.
            
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        # Add severity-based recommendations
        recommendations.extend(self._get_severity_recommendations(severity))
        
        # Add lesson-based recommendations
        recommendations.extend(self._get_lesson_recommendations(lessons))
        
        # Add general recommendation
        recommendations.append("Update risk management rules based on this analysis")
        
        return recommendations[:MAX_RECOMMENDATIONS]
    
    def _get_severity_recommendations(self, severity: str) -> List[str]:
        """Get recommendations based on severity.
        
        Args:
            severity: Severity level.
            
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        if severity in [SEVERITY_CRITICAL, SEVERITY_SEVERE]:
            recommendations.extend([
                "Implement stricter position sizing for high-risk trades",
                "Consider reducing overall portfolio exposure"
            ])
        elif severity == SEVERITY_MODERATE:
            recommendations.append("Review and tighten stop-loss thresholds")
            
        return recommendations
    
    def _get_lesson_recommendations(self, lessons: List[str]) -> List[str]:
        """Get recommendations based on lessons.
        
        Args:
            lessons: List of lessons learned.
            
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        if any('entry timing' in lesson for lesson in lessons):
            recommendations.append("Enhance entry signal confirmation requirements")
        
        if any('support' in lesson for lesson in lessons):
            recommendations.append("Add support/resistance analysis to entry criteria")
        
        if any('panic' in lesson for lesson in lessons):
            recommendations.append("Implement market sentiment filters for entries")
            
        return recommendations
    
    def _calculate_performance_metrics(
        self,
        entry_data: Dict[str, Any],
        exit_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.
        
        Args:
            entry_data: Entry trade data.
            exit_data: Exit trade data.
            
        Returns:
            Performance metrics dictionary.
        """
        entry_price = float(entry_data.get('price', 0))
        exit_price = float(exit_data.get('price', 0))
        quantity = float(entry_data.get('quantity', 0))
        
        # Calculate basic metrics
        return_pct, profit_loss = self._calculate_returns(
            entry_price, exit_price, quantity
        )
        
        # Calculate risk-adjusted return
        risk_adjusted_return = self._calculate_risk_adjusted_return(return_pct)
        
        return {
            'return_percentage': return_pct * 100,
            'profit_loss_amount': profit_loss,
            'risk_adjusted_return': risk_adjusted_return,
            'max_drawdown': min(0, return_pct),
            'win': return_pct > 0
        }
    
    def _calculate_returns(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float
    ) -> tuple[float, float]:
        """Calculate return percentage and profit/loss.
        
        Args:
            entry_price: Entry price.
            exit_price: Exit price.
            quantity: Trade quantity.
            
        Returns:
            Tuple of (return percentage, profit/loss amount).
        """
        if entry_price > 0:
            return_pct = (exit_price - entry_price) / entry_price
            profit_loss = (exit_price - entry_price) * quantity
        else:
            return_pct = 0
            profit_loss = 0
            
        return return_pct, profit_loss
    
    def _calculate_risk_adjusted_return(self, return_pct: float) -> float:
        """Calculate risk-adjusted return.
        
        Args:
            return_pct: Return percentage.
            
        Returns:
            Risk-adjusted return.
        """
        # Simplified Sharpe ratio calculation
        return return_pct / DEFAULT_VOLATILITY
    
    def _analyze_execution_quality(
        self,
        entry_data: Dict[str, Any],
        exit_data: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the quality of trade execution.
        
        Args:
            entry_data: Entry trade data.
            exit_data: Exit trade data.
            market_conditions: Market conditions.
            
        Returns:
            Execution analysis dictionary.
        """
        # Analyze entry and exit timing
        entry_timing = self._analyze_entry_timing(market_conditions)
        exit_timing = self._analyze_exit_timing(market_conditions)
        
        # Calculate slippage
        entry_slippage = self._calculate_entry_slippage(entry_data)
        
        # Determine overall quality
        if entry_timing == QUALITY_GOOD and exit_timing == QUALITY_GOOD:
            overall_quality =  QUALITY_GOOD
        overall_quality =  QUALITY_AVERAGE
        
        return {
            'entry_timing': entry_timing,
            'exit_timing': exit_timing,
            'entry_slippage': entry_slippage * 100,
            'execution_speed': QUALITY_NORMAL,
            'overall_quality': overall_quality
        }
    
    def _analyze_entry_timing(self, market_conditions: Dict[str, Any]) -> str:
        """Analyze entry timing quality.
        
        Args:
            market_conditions: Market conditions.
            
        Returns:
            Timing quality string.
        """
        rsi = market_conditions.get('rsi', {}).get('rsi_14', DEFAULT_RSI)
        return QUALITY_GOOD if rsi < RSI_OVERSOLD_THRESHOLD else QUALITY_NEUTRAL
    
    def _analyze_exit_timing(self, market_conditions: Dict[str, Any]) -> str:
        """Analyze exit timing quality.
        
        Args:
            market_conditions: Market conditions.
            
        Returns:
            Timing quality string.
        """
        rsi = market_conditions.get('rsi', {}).get('rsi_14', DEFAULT_RSI)
        return QUALITY_GOOD if rsi > RSI_OVERBOUGHT_THRESHOLD else QUALITY_NEUTRAL
    
    def _calculate_entry_slippage(self, entry_data: Dict[str, Any]) -> float:
        """Calculate entry slippage.
        
        Args:
            entry_data: Entry trade data.
            
        Returns:
            Slippage as a decimal.
        """
        expected_entry = entry_data.get('expected_price', entry_data.get('price'))
        actual_entry = entry_data.get('price', 0)
        
        if expected_entry > 0:
            return abs(actual_entry - expected_entry) / expected_entry
        return 0
    
    def _identify_success_factors(
        self,
        performance: Dict[str, Any],
        execution: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> List[str]:
        """Identify factors that contributed to success or failure.
        
        Args:
            performance: Performance metrics.
            execution: Execution analysis.
            market_conditions: Market conditions.
            
        Returns:
            List of factor strings.
        """
        factors = []
        
        if performance['win']:
            factors.extend(self._identify_success_factors_for_win(
                execution, market_conditions
            ))
        else:
            factors.extend(self._identify_failure_factors(
                execution, market_conditions
            ))
        
        return factors
    
    def _identify_success_factors_for_win(
        self,
        execution: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> List[str]:
        """Identify success factors for winning trades.
        
        Args:
            execution: Execution analysis.
            market_conditions: Market conditions.
            
        Returns:
            List of success factor strings.
        """
        factors = []
        
        if execution['entry_timing'] == QUALITY_GOOD:
            factors.append("Good entry timing at oversold conditions")
        if execution['exit_timing'] == QUALITY_GOOD:
            factors.append("Good exit timing at overbought conditions")
        if market_conditions.get('volume_ratio', 1) > VOLUME_CONFIRMATION_THRESHOLD:
            factors.append("High volume confirmed the move")
            
        return factors
    
    def _identify_failure_factors(
        self,
        execution: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> List[str]:
        """Identify factors for losing trades.
        
        Args:
            execution: Execution analysis.
            market_conditions: Market conditions.
            
        Returns:
            List of failure factor strings.
        """
        factors = []
        
        if execution['entry_timing'] != QUALITY_GOOD:
            factors.append("Poor entry timing")
        if market_conditions.get('market_sentiment') == SENTIMENT_BEARISH:
            factors.append("Traded against market sentiment")
        if execution['entry_slippage'] > HIGH_SLIPPAGE_THRESHOLD:
            factors.append("High entry slippage affected profitability")
            
        return factors
    
    def _generate_improvement_suggestions(
        self,
        performance: Dict[str, Any],
        execution: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions for improvement.
        
        Args:
            performance: Performance metrics.
            execution: Execution analysis.
            
        Returns:
            List of suggestion strings.
        """
        suggestions = []
        
        if not performance['win']:
            suggestions.append("Review entry criteria to reduce false signals")
        
        if execution['entry_slippage'] > MODERATE_SLIPPAGE_THRESHOLD:
            suggestions.append("Consider using limit orders to reduce slippage")
        
        if performance['max_drawdown'] < DRAWDOWN_THRESHOLD:
            suggestions.append("Implement tighter stop-loss to limit drawdowns")
        
        if execution['overall_quality'] != QUALITY_GOOD:
            suggestions.append("Improve timing indicators for better execution")
        
        return suggestions
    
    def _calculate_trade_rating(
        self,
        performance: Dict[str, Any],
        execution: Dict[str, Any]
    ) -> str:
        """Calculate overall trade rating.
        
        Args:
            performance: Performance metrics.
            execution: Execution analysis.
            
        Returns:
            Rating string.
        """
        score = self._calculate_trade_score(performance, execution)
        return self._convert_score_to_rating(score)
    
    def _calculate_trade_score(
        self,
        performance: Dict[str, Any],
        execution: Dict[str, Any]
    ) -> int:
        """Calculate numerical trade score.
        
        Args:
            performance: Performance metrics.
            execution: Execution analysis.
            
        Returns:
            Score integer.
        """
        score = 0
        
        # Performance scoring
        score += self._score_performance(performance)
        
        # Execution scoring
        score += self._score_execution(execution)
        
        # Risk-adjusted scoring
        score += self._score_risk_adjusted_return(performance)
        
        return score
    
    def _score_performance(self, performance: Dict[str, Any]) -> int:
        """Score based on performance.
        
        Args:
            performance: Performance metrics.
            
        Returns:
            Performance score.
        """
        score = 0
        
        if performance['win']:
            score += WIN_SCORE
            
        if performance['return_percentage'] > GOOD_RETURN_THRESHOLD:
            score += GOOD_RETURN_SCORE
        elif performance['return_percentage'] < POOR_RETURN_THRESHOLD:
            score += POOR_RETURN_PENALTY
            
        return score
    
    def _score_execution(self, execution: Dict[str, Any]) -> int:
        """Score based on execution quality.
        
        Args:
            execution: Execution analysis.
            
        Returns:
            Execution score.
        """
        if execution['overall_quality'] == QUALITY_GOOD:
            return GOOD_EXECUTION_SCORE
        elif execution['overall_quality'] == QUALITY_AVERAGE:
            return AVERAGE_EXECUTION_SCORE
        return 0
    
    def _score_risk_adjusted_return(self, performance: Dict[str, Any]) -> int:
        """Score based on risk-adjusted return.
        
        Args:
            performance: Performance metrics.
            
        Returns:
            Risk-adjusted score.
        """
        if performance['risk_adjusted_return'] > 1:
            return RISK_ADJUSTED_SCORE
        return 0
    
    def _convert_score_to_rating(self, score: int) -> str:
        """Convert numerical score to rating.
        
        Args:
            score: Numerical score.
            
        Returns:
            Rating string.
        """
        if score >= EXCELLENT_RATING_THRESHOLD:
            return RATING_EXCELLENT
        elif score >= GOOD_RATING_THRESHOLD:
            return RATING_GOOD
        elif score >= AVERAGE_RATING_THRESHOLD:
            return RATING_AVERAGE
        elif score >= POOR_RATING_THRESHOLD:
            return RATING_POOR
        else:
            return RATING_VERY_POOR
    
    def _build_stop_loss_analysis_result(
        self,
        symbol: str,
        action: str,
        trade_metrics: Dict[str, Any],
        severity: str,
        market_analysis: Dict[str, Any],
        ai_analysis: Dict[str, Any],
        lessons: List[str]
    ) -> Dict[str, Any]:
        """Build stop-loss analysis result dictionary.
        
        Args:
            symbol: Trading symbol.
            action: Action taken.
            trade_metrics: Trade metrics.
            severity: Severity level.
            market_analysis: Market analysis.
            ai_analysis: AI analysis.
            lessons: Lessons learned.
            
        Returns:
            Complete analysis result.
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'trade_metrics': trade_metrics,
            'failure_severity': severity,
            'market_analysis': market_analysis,
            'ai_analysis': ai_analysis,
            'lessons_learned': lessons,
            'recommendations': self._generate_future_recommendations(
                severity, lessons
            )
        }
 
    def _build_error_result(self, error: str) -> Dict[str, Any]:
        """Build error result dictionary.
        
        Args:
            error: Error message.
            
        Returns:
            Error result dictionary.
        """
        return {
            'error': error,
            'failure_severity': SEVERITY_UNKNOWN,
            'lessons_learned': []
        }


