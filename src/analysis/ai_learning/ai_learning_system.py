import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class AILearningSystem:
    """AI learning system that analyzes past trades to improve future decisions."""
    
    def __init__(self, db_path: str = "trading_data.db") -> None:
        """Initialize the AI learning system.
        
        Args:
            db_path: Path to the trading database.
        """
        self.db_path = db_path
        self.lessons_learned = []
        self.pattern_statistics = defaultdict(lambda: {"success": 0, "total": 0})
    
    def analyze_historical_trades(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Analyze historical trades for learning insights.
        
        Args:
            days_back: Number of days to analyze.
            
        Returns:
            List of learning insights as dictionaries.
        """
        try:
            logger.info(f"Analyzing historical trades for the last {days_back} days")
            
            # Get trades from database
            trades = self._fetch_historical_trades(days_back)
            if not trades:
                logger.info("No historical trades found to analyze")
                return []
            
            # Update pattern statistics for future recommendations
            self._update_pattern_statistics(trades)
            
            # Analyze trade patterns
            insights = []
            
            # Always add success rate analysis
            insights.append(self._analyze_success_rate(trades))
            
            # Always add failed trades analysis
            insights.append(self._analyze_failed_trades(trades))
            
            # Store lessons for future reference
            self.lessons_learned = insights
            
            logger.info(f"Generated {len(insights)} learning insights from {len(trades)} trades")
            return insights
            
        except Exception as e:
            logger.error(f"Historical trade analysis failed: {e}")
            return []
    
    # USED
    def _fetch_historical_trades(self, days_back: int) -> List[Dict[str, Any]]:
        """Fetch historical trades from database.
        
        Args:
            days_back: Number of days to look back.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                    SELECT * FROM trades 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                """
                
                cursor.execute(query, (cutoff_date.isoformat(),))
                trades = [dict(row) for row in cursor.fetchall()]
                
                return trades
                
        except Exception as e:
            logger.error(f"Failed to fetch historical trades: {e}")
            return []
    
    # USED
    def _update_pattern_statistics(self, trades: List[Dict[str, Any]]) -> None:
        """DB의 거래 기록은 그대로 유지되고, 분석용 통계만 재계산
        
        Args:
            trades: List of trade records.
        """
        # Reset statistics
        self.pattern_statistics.clear()
        
        # Count success/total by symbol
        for trade in trades:
            symbol = trade['symbol']
            self.pattern_statistics[symbol]['total'] += 1
            if trade.get('success', False):
                self.pattern_statistics[symbol]['success'] += 1
    
    # USED
    def _analyze_success_rate(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall success rate of trades.
        
        Args:
            trades: List of trade records.
            
        Returns:
            Success rate insight.
        """
        # Only analyze completed trades (sells)
        sell_trades = [t for t in trades if t.get('action') in ['partial_sell', 'sell_all'] and t.get('success') is not None]
        if not sell_trades:
            return {
                "type": "success_rate",
                "title": "Overall Trading Performance",
                "metrics": {
                    "total_trades": 0,
                    "successful_trades": 0,
                    "success_rate": 0.0,
                    "avg_profit": 0.0,
                    "avg_loss": 0.0
                },
                "performance": "no_data",
                "recommendation": "No completed trades to analyze",
                "priority": "low"
            }
        
        successful = [t for t in sell_trades if t.get('success', False)]
        failed = [t for t in sell_trades if not t.get('success', False)]
        
        total = len(sell_trades)
        success_count = len(successful)
        success_rate = (success_count / total) * 100 if total > 0 else 0
        
        # Calculate average profit/loss
        avg_profit = sum(t.get('actual_return', 0) for t in successful) / len(successful) if successful else 0
        avg_loss = sum(t.get('actual_return', 0) for t in failed) / len(failed) if failed else 0
        
        # Determine performance level
        if success_rate >= 70:
            performance = "excellent"
            recommendation = "Continue with current strategy"
        elif success_rate >= 50:
            performance = "moderate"
            recommendation = "Review and refine decision criteria"
        else:
            performance = "poor"
            recommendation = "Major strategy adjustment needed"
        
        return {
            "type": "success_rate",
            "title": "Overall Trading Performance",
            "metrics": {
                "total_trades": total,
                "successful_trades": success_count,
                "success_rate": round(success_rate, 2),
                "avg_profit": round(avg_profit, 2),
                "avg_loss": round(avg_loss, 2)
            },
            "performance": performance,
            "recommendation": recommendation,
            "priority": "high" if performance == "poor" else "medium"
        }
    
    # USED
    def _analyze_failed_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in failed trades.
        
        Args:
            trades: List of trade records.
            
        Returns:
            Failed trades insight.
        """
        # Only analyze completed trades (sells)
        sell_trades = [t for t in trades if t.get('action') in ['partial_sell', 'sell_all'] and t.get('success') is not None]
        if not sell_trades:
            return {
                "type": "failure_analysis",
                "title": "Common Failure Patterns",
                "metrics": {
                    "total_failures": 0,
                    "failure_rate": 0.0
                },
                "common_reasons": {},
                "most_common_reason": "none",
                "recommendation": "No completed trades to analyze",
                "priority": "low"
            }
        
        failed_trades = [t for t in sell_trades if not t.get('success', False)]
        if not failed_trades:
            return {
                "type": "failure_analysis",
                "title": "Common Failure Patterns",
                "metrics": {
                    "total_failures": 0,
                    "failure_rate": 0.0
                },
                "common_reasons": {},
                "most_common_reason": "none",
                "recommendation": "No failures to analyze - excellent performance",
                "priority": "low"
            }
        
        # Group failures by reason and action type
        failure_reasons = defaultdict(int)
        action_failures = defaultdict(int)
        
        for trade in failed_trades:
            reason = trade.get('reason', '').lower()
            action = trade.get('action', '')
            
            # Track action-specific failures
            action_failures[action] += 1
            
            # Extract key phrases from reason
            if 'stop-loss' in reason or 'stop loss' in reason or '손절' in reason:
                failure_reasons['stop_loss'] += 1
            elif '하락' in reason or 'bear' in reason or 'down' in reason or 'decline' in reason:
                failure_reasons['market_downturn'] += 1
            elif 'volatil' in reason or '변동성' in reason:
                failure_reasons['high_volatility'] += 1
            elif 'circuit' in reason or '서킷' in reason:
                failure_reasons['circuit_breaker'] += 1
            else:
                failure_reasons['other'] += 1
        
        # Find most common failure reason
        most_common = max(failure_reasons.items(), key=lambda x: x[1])
        
        # Get recommendation based on failure reason
        recommendations = {
            "stop_loss": "Consider adjusting stop-loss thresholds or improving entry timing",
            "market_downturn": "Implement better market trend detection before entering positions",
            "high_volatility": "Add volatility filters to avoid trading in unstable conditions",
            "circuit_breaker": "Review risk management settings and daily loss limits",
            "other": "Review AI decision criteria and add more specific failure tracking"
        }
        
        return {
            "type": "failure_analysis",
            "title": "Common Failure Patterns",
            "metrics": {
                "total_completed_trades": len(sell_trades),
                "total_failures": len(failed_trades),
                "failure_rate": round((len(failed_trades) / len(sell_trades)) * 100, 2)
            },
            "common_reasons": dict(failure_reasons),
            "action_failures": dict(action_failures),
            "most_common_reason": most_common[0],
            "recommendation": recommendations.get(most_common[0], "Conduct detailed analysis of failure patterns"),
            "priority": "high"
        }
    
    # USED
    def get_symbol_success_rate(self, symbol: str) -> float:
        """Get success rate for a specific symbol.
        
        Args:
            symbol: Cryptocurrency symbol.
            
        Returns:
            Success rate percentage (0-100) or 0 if no history.
        """
        stats = self.pattern_statistics.get(symbol, {"success": 0, "total": 0})
        if stats['total'] == 0:
            return 0.0
        return round((stats['success'] / stats['total']) * 100, 2)
    
    def should_avoid_symbol(self, symbol: str, threshold: float = 40.0) -> bool:
        """Check if a symbol should be avoided based on poor historical performance.
        
        Args:
            symbol: Cryptocurrency symbol.
            threshold: Minimum success rate percentage (default 40%).
            
        Returns:
            True if symbol should be avoided, False otherwise.
        """
        success_rate = self.get_symbol_success_rate(symbol)
        return success_rate > 0 and success_rate < threshold  # Only avoid if we have history AND it's bad
    
    def get_failure_warnings(self) -> List[str]:
        """Get warnings based on recent failure patterns.
        
        Returns:
            List of warning messages.
        """
        warnings = []
        
        for lesson in self.lessons_learned:
            if lesson.get('type') == 'failure_analysis':
                failure_rate = lesson['metrics'].get('failure_rate', 0)
                if failure_rate > 40:  # High failure rate
                    most_common = lesson.get('most_common_reason', '')
                    if most_common == 'stop_loss':
                        warnings.append("⚠️ High stop-loss triggers (adjust thresholds)")
                    elif most_common == 'market_downturn':
                        warnings.append("⚠️ Frequent losses in market downturns")
                    elif most_common == 'high_volatility':
                        warnings.append("⚠️ Volatility causing losses")
                    elif most_common == 'circuit_breaker':
                        warnings.append("⚠️ Circuit breaker triggering often")
        
        return warnings

    # USED
    def get_trading_insights_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive trading insights for a specific symbol.
        
        Args:
            symbol: Cryptocurrency symbol.
            
        Returns:
            Comprehensive insights for the symbol.
        """
        success_rate = self.get_symbol_success_rate(symbol)
        should_avoid = self.should_avoid_symbol(symbol)
        
        # Get recent trades for this symbol
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT action, success, actual_return, reason
                    FROM trades
                    WHERE symbol = ? AND success IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT 10
                """, (symbol,))
                
                recent_trades = cursor.fetchall()
                
                # Analyze patterns
                patterns = {
                    "recent_performance": "improving" if recent_trades and recent_trades[0][1] else "declining",
                    "common_action": "partial_sell" if any(t[0] == 'partial_sell' for t in recent_trades) else "sell_all",
                    "avg_recent_return": sum(t[2] for t in recent_trades if t[2]) / len(recent_trades) if recent_trades else 0
                }
                
                return {
                    "symbol": symbol,
                    "success_rate": success_rate,
                    "should_avoid": should_avoid,
                    "patterns": patterns,
                    "recommendation": self._generate_symbol_recommendation(symbol, patterns)
                }
                
        except Exception as e:
            logger.error(f"Failed to get insights for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def _generate_symbol_recommendation(self, symbol: str, patterns: Dict[str, Any]) -> str:
        """Generate recommendation based on patterns."""
        if patterns["recent_performance"] == "improving":
            return f"Continue trading {symbol} - showing improvement"
        elif patterns["avg_recent_return"] < -5:
            return f"Avoid {symbol} - consistent losses"
        else:
            return f"Monitor {symbol} carefully - mixed results"


# Global instance for trade history analysis
trade_history_analyzer = AILearningSystem()