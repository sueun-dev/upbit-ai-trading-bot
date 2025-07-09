"""Trading orchestrator for coordinating the end-to-end trading process.

This module orchestrates news collection, market analysis, AI decision making,
and trade execution with comprehensive safety systems.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

from src.analysis.ai_analyzer import AIAnalyzer
from src.analysis.enhanced_market_analyzer import enhanced_market_analyzer
from src.analysis.multi_ai_validator import MultiAIValidator
from src.analysis.risk_monitor import RiskMonitor
from src.analysis.adaptive_risk_manager import AdaptiveRiskManager
from src.analysis.pattern_learner import PatternLearner
from src.analysis.post_trade_analyzer import PostTradeAnalyzer
from src.analysis.ai_learning.ai_learning_system import AILearningSystem
from src.data.collectors.enhanced_market_data import enhanced_collector
from src.data.scrapers.multi_source_scraper import multi_source_scraper
from src.core.clients.upbit_trader import UpbitTrader
from src.core.clients.circuit_breaker import CircuitBreaker
from src.shared.utils.helpers import get_formatted_datetime
from src.shared.utils.data_store import DataStore, TradeRecord, AIAnalysisResult, TradeAnalysis


# Default values
DEFAULT_CONFIDENCE = 0.5
DEFAULT_ACTION = 'hold'
DEFAULT_BUY_AMOUNT_KRW = 30_000     # Default buy amount (KRW)
MIN_PORTFOLIO_BALANCE = 0
MIN_VIABLE_TRADE_AMOUNT = 10000
STOP_LOSS_CONFIDENCE = 0.9

# API monitoring
API_STATUS_OK = 200
API_TYPE_NEWS = 'news_apis'
API_SERVICE_SCRAPER = 'multi_source_scraper'
API_METHOD_GET = 'GET'

# News collection
MAX_NEWS_ARTICLES = 15

# Risk levels
RISK_LEVEL_HIGH = 'high'
RISK_LEVEL_CRITICAL = 'critical'
RISK_LEVEL_MEDIUM = 'medium'
RISK_LEVEL_UNKNOWN = 'unknown'

# Trading actions
ACTION_BUY = 'buy'
ACTION_BUY_MORE = 'buy_more'
ACTION_SELL_ALL = 'sell_all'
ACTION_PARTIAL_SELL = 'partial_sell'
ACTION_HOLD = 'hold'

# Health check
HEALTH_CHECK_SYMBOLS = ['BTC', 'ETH']
HEALTH_CHECK_TRADE_LIMIT = 10
HEALTH_CHECK_RECENT_TRADES_LIMIT = 50
HEALTH_CHECK_PATTERN_DAYS = 7
HEALTH_CHECK_TRADE_HOURS = 24
HEALTH_CHECK_TRADE_LIMIT_CIRCUIT = 20


# Log messages
LOG_CYCLE_COMPLETE = "✅ [CYCLE COMPLETED] Trading cycle finished"
LOG_NEWS_COLLECTED = "Collected %d news items from multiple sources."
LOG_SYMBOLS_EXTRACTED = "Target symbols: %s"
LOG_EXECUTING_DECISIONS = "Executing %d decisions on %s %s"
LOG_MISSING_FIELDS = "%s: Missing required fields %s - skipping"
LOG_DECISION_SUMMARY = "[%s] Action=%s | Confidence=%.2f%s"
LOG_AI_ANALYSIS_RECORDED = "📊 AI analysis recorded for {} {}: {}"
LOG_AI_ANALYSIS_FAILED = "Failed to analyze trade action for {}: {}"
LOG_AVERAGING_SUCCESS = "✅ %s: Averaged down (loss: %.2f%%, attempts: %d) on %s"
LOG_NO_NEWS = "No news collected; aborting cycle"
LOG_NO_MARKET_DATA = "No market data; aborting cycle"
LOG_INVALID_PORTFOLIO = "Invalid portfolio; aborting cycle"
LOG_PATTERN_LESSONS = "📚 Applied {} learned lessons to AI analysis"
LOG_HIGH_RISK_DETECTED = "⚠️ High portfolio risk detected: {}"
LOG_AI_RISK_ASSESSMENT = "🤖 AI Risk Assessment: {}"
LOG_STOP_LOSS_TRIGGERED = "⚠️ {} positions triggering stop-loss"
LOG_POST_TRADE_COMPLETE = "📋 Post-trade analysis completed for {} stop-loss"
LOG_FAILURE_SEVERITY = "🎯 Failure severity: {}"
LOG_LESSONS_LEARNED = "📚 Lessons learned: {} items"
LOG_POST_TRADE_FAILED = "Failed post-trade analysis for {}: {}"
LOG_CYCLE_FAILED = "Trading cycle failed: {}"
LOG_HEALTH_CHECK_FAILED = "AI safety systems health check failed: {}"
LOG_SYSTEM_HEALTH_FAILED = "Failed to get system health: {}"

# Required decision fields
REQUIRED_DECISION_FIELDS = ["action", "reason"]

# Action translations
ACTION_KOREAN_MAP = {
    'buy': '매수',
    'sell_all': '전량 매도',
    'partial_sell': '부분 매도',
    'hold': '보유',
    'buy_more': '추가 매수'
}

# AI safety system names
AI_SYSTEM_MULTI_VALIDATOR = 'multi_ai_validator'
AI_SYSTEM_RISK_MONITOR = 'risk_monitor'
AI_SYSTEM_ADAPTIVE_RISK = 'adaptive_risk_manager'
AI_SYSTEM_PATTERN_LEARNER = 'pattern_learner'
AI_SYSTEM_POST_TRADE = 'post_trade_analyzer'
AI_SYSTEM_STATUS_OPERATIONAL = 'operational'
AI_SYSTEM_STATUS_DEGRADED = 'degraded'

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """Orchestrates the end-to-end trading process.
    
    Coordinates news collection, market analysis, AI decision making,
    and trade execution with comprehensive safety systems.
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        openai_api_key: str,
        trade_analyzer: AILearningSystem
    ) -> None:
        """Initialize orchestrator with trader and AI analyzer.
        
        Args:
            access_key: Upbit API access key.
            secret_key: Upbit API secret key.
            openai_api_key: OpenAI API key.
            trade_analyzer: Trade history analyzer instance.
        """
        self.circuit_breaker = CircuitBreaker()
        self.trader = UpbitTrader(access_key=access_key, secret_key=secret_key)
        self.ai_analyzer = AIAnalyzer(api_key=openai_api_key)
        self.data_store = DataStore()
        self.trade_analyzer = trade_analyzer
        
        # AI Safety Systems - Initialize with API key
        self.multi_ai_validator = MultiAIValidator(api_key=openai_api_key)
        self.risk_monitor = RiskMonitor(api_key=openai_api_key)
        self.adaptive_risk_manager = AdaptiveRiskManager()
        self.pattern_learner = PatternLearner(api_key=openai_api_key)
        self.post_trade_analyzer = PostTradeAnalyzer(api_key=openai_api_key)
        
        self._cycle_count = 0
        self._last_cycle_time = None

    # USED
    def collect_news(self) -> List[Dict]:
        """Return latest news items from multiple sources or empty list if none.
        
        Returns:
            List of news articles with title, summary, source, and url.
        """
        news_list = multi_source_scraper.fetch_all_news(max_total_articles=MAX_NEWS_ARTICLES) or []
        logger.info(LOG_NEWS_COLLECTED, len(news_list))
        
        # Store news in data store
        self._store_news_articles(news_list)
        return news_list
    
    # USED
    def _store_news_articles(self, news_list: List[Dict]) -> None:
        """Store news articles in data store, skipping duplicates.
        
        Args:
            news_list: List of news articles to store.
        """
        stored_count = 0
        for article in news_list:
            success = self.data_store.record_news(
                title=article.get('title', ''),
                summary=article.get('summary', ''),
                source=article.get('source', ''),
                url=article.get('url', '')
            )
            if success:
                stored_count += 1
        
        if stored_count < len(news_list):
            logger.info(f"Skipped {len(news_list) - stored_count} duplicate news articles")

    # USED
    def extract_market_symbols(self, news_list: List[Dict]) -> List[str]:
        """Extract symbols from news items.
        
        Args:
            news_list: List of news articles.
            
        Returns:
            List of extracted cryptocurrency symbols.
        """
        symbols = self.ai_analyzer.extract_symbols_from_news(news_list) or []
        logger.info(LOG_SYMBOLS_EXTRACTED, symbols)
        return symbols

    # USED
    def collect_market_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch enhanced market data for symbols.
        
        Args:
            symbols: List of cryptocurrency symbols.
            
        Returns:
            Dictionary mapping symbols to their market data.
        """
        # Get portfolio to check holdings
        portfolio = self.trader.get_portfolio_status()
        held_assets = set(portfolio.get('assets', {}).keys())
        
        data_map: Dict[str, Dict] = {}
        
        for symbol in symbols:
            enhanced_data = enhanced_collector.get_enhanced_market_data(symbol)
            if enhanced_data:
                # Convert dataclass to dict for JSON serialization
                market_dict = enhanced_data.__dict__
                
                # Add is_held field BEFORE analysis
                market_dict['is_held'] = symbol in held_assets
                
                # Perform comprehensive analysis
                comprehensive_analysis = enhanced_market_analyzer.analyze(market_dict)
                
                # Add analysis results to market data
                market_dict['comprehensive_analysis'] = comprehensive_analysis
                data_map[symbol] = market_dict
        
        logger.info("Collected enhanced market data for %d coins.", len(data_map))
        return data_map

    # USED
    def execute_trading_decisions(self, decisions: Dict[str, Dict]) -> None:
        """Execute trading decisions with validation and logging.
        
        Args:
            decisions: Dictionary mapping symbols to trading decisions.
        """
        current_date, current_time = get_formatted_datetime()
        logger.info(LOG_EXECUTING_DECISIONS, len(decisions), current_date, current_time)
        
        for symbol, decision in decisions.items():
            if not self._validate_decision(symbol, decision):
                continue
            self._log_decision_summary(symbol, decision)
            self._perform_trade(symbol, decision, current_date)

    # USED
    def _validate_decision(self, symbol: str, decision: Dict) -> bool:
        """Check that all required fields exist.
        
        Args:
            symbol: Cryptocurrency symbol.
            decision: Trading decision dictionary.
            
        Returns:
            True if decision is valid, False otherwise.
        """
        missing_fields = self._find_missing_fields(decision)
        
        # Handle confidence field specially - it can be None, we'll default it
        if "confidence" not in decision:
            decision["confidence"] = DEFAULT_CONFIDENCE
        elif decision["confidence"] is None:
            decision["confidence"] = DEFAULT_CONFIDENCE
        
        if missing_fields:
            logger.error(LOG_MISSING_FIELDS, symbol, missing_fields)
            return False
        return True
    
    # USED
    def _find_missing_fields(self, decision: Dict) -> List[str]:
        """Find missing required fields in decision.
        
        Args:
            decision: Trading decision dictionary.
            
        Returns:
            List of missing field names.
        """
        missing_fields: List[str] = []
        for field in REQUIRED_DECISION_FIELDS:
            if field not in decision or decision[field] is None:
                missing_fields.append(field)
        return missing_fields
    
    # USED
    def _log_decision_summary(self, symbol: str, decision: Dict) -> None:
        """Emit a concise log line for each decision.
        
        Args:
            symbol: Cryptocurrency symbol.
            decision: Trading decision dictionary.
        """
        action = decision["action"]
        confidence = decision.get("confidence", DEFAULT_CONFIDENCE)
        if confidence is None:
            confidence = DEFAULT_CONFIDENCE
        
        notes = self._build_decision_notes(decision)
        note_str = f" ({'; '.join(notes)})" if notes else ""
        
        logger.info(LOG_DECISION_SUMMARY, symbol, action, confidence, note_str)
    
    # USED
    def _build_decision_notes(self, decision: Dict) -> List[str]:
        """Build notes list for decision logging.
        
        Args:
            decision: Trading decision dictionary.
            
        Returns:
            List of note strings.
        """
        notes: List[str] = []
        if decision.get("averaging_analysis"):
            notes.append("averaging")
        if decision.get("original_action"):
            notes.append(f"from {decision['original_action']}")
        return notes

    # USED
    def _perform_trade(self, symbol: str, decision: Dict, date: str) -> None:
        """Call trader; emit post-trade averaging log if applicable.
        
        Args:
            symbol: Cryptocurrency symbol.
            decision: Trading decision dictionary.
            date: Current date string.
        """
        executed = self.trader.execute_trade(
            symbol, decision["action"], decision["reason"], decision
        )
        if not executed:
            return
        
        # Analyze and record trade
        self._analyze_and_record_trade(symbol, decision)
        
        # Log averaging analysis if applicable
        self._log_averaging_analysis(symbol, decision, date)
    
    # USED
    def _analyze_and_record_trade(self, symbol: str, decision: Dict) -> None:
        """Analyze trade action with AI and record it.
        
        Args:
            symbol: Cryptocurrency symbol.
            decision: Trading decision dictionary.
        """
        try:
            # Get market data for analysis
            market_data = enhanced_collector.get_enhanced_market_data(symbol)
            market_dict = market_data.__dict__ if market_data else {}
            
            # Create and store trade analysis
            trade_analysis = self._create_trade_analysis(symbol, decision, market_dict)
            self.data_store.record_trade_analysis(trade_analysis)
            
            # Record the actual trade with enhanced information
            self._record_enhanced_trade(symbol, decision, market_dict)
            
            logger.info(LOG_AI_ANALYSIS_RECORDED.format(
                symbol, decision['action'], decision["reason"]
            ))
            
        except Exception as e:
            logger.error(LOG_AI_ANALYSIS_FAILED.format(symbol, e))
    
    # USED
    def _create_trade_analysis(self, symbol: str, decision: Dict, market_dict: Dict) -> TradeAnalysis:
        """Create trade analysis record.
        
        Args:
            symbol: Cryptocurrency symbol.
            decision: Trading decision dictionary.
            market_dict: Market data dictionary.
            
        Returns:
            TradeAnalysis object.
        """
        timestamp = datetime.now().isoformat()
        action = decision["action"]
        
        return TradeAnalysis(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            action_korean=ACTION_KOREAN_MAP.get(action, action),
            analysis=f"Executed {action} for {symbol}",
            summary=decision["reason"],
            confidence=decision.get("confidence", DEFAULT_CONFIDENCE),
            market_context=f"Price: {market_dict.get('current_price', 0):,.0f}원"
        )
    
    # USED
    def _record_enhanced_trade(self, symbol: str, decision: Dict, market_dict: Dict) -> None:
        """Record enhanced trade information including profit calculation for sells.
        
        Args:
            symbol: Cryptocurrency symbol.
            decision: Trading decision dictionary.
            market_dict: Market data dictionary.
        """
        action = decision.get('action', '')
        current_price = market_dict.get('current_price', 0)
        
        # Get quantity from decision or estimate
        quantity = decision.get('quantity', 0)
        if not quantity and action in ['buy', 'buy_more']:
            # Estimate quantity from amount
            amount_krw = decision.get('amount_krw', DEFAULT_BUY_AMOUNT_KRW)
            quantity = amount_krw / current_price if current_price > 0 else 0
        
        # Initialize trade record fields
        remaining_quantity = None
        buy_trade_ids = None
        avg_buy_price = None
        success = None
        actual_return = None
        
        # Handle different action types
        if action in ['buy', 'buy_more']:
            remaining_quantity = quantity  # New buy order has full quantity remaining
            
        elif action in ['partial_sell', 'sell_all']:
            # Calculate profit for sell orders
            profit_percent, avg_buy_price, used_buy_ids = self.data_store.calculate_sell_profit(
                symbol, current_price, quantity
            )
            
            buy_trade_ids = ','.join(map(str, used_buy_ids))
            actual_return = profit_percent
            success = profit_percent > 0
            
            # Update trade analyzer with result
            self.trade_analyzer.analyze_historical_trades(days_back=7)
            
            logger.info(f"📊 {symbol} {action}: {profit_percent:.2f}% return (avg buy: {avg_buy_price:,.0f}원)")
        
        # Create and record trade
        trade_record = TradeRecord(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action=action,
            confidence=decision.get('confidence', DEFAULT_CONFIDENCE),
            reason=decision.get('reason', ''),
            price=current_price,
            amount_krw=decision.get('amount_krw'),
            quantity=quantity,
            remaining_quantity=remaining_quantity,
            buy_trade_ids=buy_trade_ids,
            avg_buy_price=avg_buy_price,
            success=success,
            actual_return=actual_return
        )
        
        trade_id = self.data_store.record_trade(trade_record)
        logger.debug(f"Recorded trade #{trade_id} for {symbol} {action}")
    
    # USED
    def _log_averaging_analysis(self, symbol: str, decision: Dict, date: str) -> None:
        """Log averaging analysis if present.
        
        Args:
            symbol: Cryptocurrency symbol.
            decision: Trading decision dictionary.
            date: Current date string.
        """
        averaging_analysis = decision.get("averaging_analysis")
        if averaging_analysis:
            loss_pct = averaging_analysis["current_loss"] * 100
            attempts = averaging_analysis["attempts_used"]
            logger.info(LOG_AVERAGING_SUCCESS, symbol, loss_pct, attempts, date)

    # USED
    def run_trading_cycle(self) -> None:
        """Run one trading cycle: collect data, analyze, and execute decisions."""
        cycle_start_time = time.time()
        logger.info("🚀 [LOOP START] Starting new trading cycle")
        
        # Update cycle tracking
        self._cycle_count += 1
        self._last_cycle_time = datetime.now().isoformat()
        
        # Check circuit breaker
        if not self.circuit_breaker.can_trade():
            logger.warning("Circuit breaker activated - skipping trading cycle")
            return
        
        try:
            # Execute the trading cycle
            self._execute_trading_cycle(cycle_start_time)
            
        except Exception as e:
            logger.error(LOG_CYCLE_FAILED.format(e))
    
    # USED
    def _execute_trading_cycle(self, cycle_start_time: float) -> None:
        """Execute the main trading cycle.
        
        Args:
            cycle_start_time: Timestamp when cycle started.
        """
        # Collect and validate data
        news_list, symbols, market_data, portfolio = self._collect_and_validate_data()
        if not all([news_list, symbols, market_data, portfolio]):
            return
        
        # Get AI decisions with safety systems
        decisions = self._get_ai_decisions_with_safety(news_list, market_data, portfolio)
        
        # Apply risk management and stop-loss
        decisions = self._apply_risk_management(decisions, portfolio, market_data)
        
        # Execute trades
        self._execute_trades_with_analysis(decisions, market_data, portfolio)
        
        # Record results and complete cycle
        self._complete_trading_cycle(
            cycle_start_time, news_list, symbols, market_data, portfolio, decisions
        )
    
    # USED
    def _collect_and_validate_data(self) -> Tuple[List[Dict], List[str], Dict[str, Dict], Dict]:
        """Collect news, symbols, market data, and portfolio.
        
        Returns:
            Tuple of (news_list, symbols, market_data, portfolio).
        """
        # 1. Always get portfolio first - required for all operations
        portfolio = self.trader.get_portfolio_status()
        
        # Validate portfolio
        if not portfolio or portfolio.get("total_balance", 0) <= MIN_PORTFOLIO_BALANCE:
            raise ValueError("Invalid portfolio data")
        
        # 2. Collect news
        news_list = self.collect_news()
        
        if not news_list:
            logger.info(LOG_NO_NEWS)
            news_list = []
        
        # 3. Extract symbols from news or use portfolio holdings
        symbols = self.extract_market_symbols(news_list) if news_list else []
        
        # Add portfolio holdings to symbols
        # 뉴스 심볼: ['BTC', 'ETH']
        # 보유 심볼: ['XRP', 'ETH', 'ADA']
        # 최종 심볼: ['BTC', 'ETH', 'XRP', 'ADA'] (중복 제거됨)
        if portfolio and 'holdings' in portfolio:
            portfolio_symbols = list(portfolio['holdings'].keys())
            symbols = list(set(symbols + portfolio_symbols))  # Remove duplicates
        
        if not symbols:
            logger.info("No symbols extracted; aborting cycle")
            symbols = []
        
        # 4. Collect market data for all symbols (news + portfolio)
        market_data = {}
        if symbols:
            market_data = self.collect_market_data(symbols)
            if not market_data:
                logger.error(LOG_NO_MARKET_DATA)
                market_data = {}
        
        return news_list, symbols, market_data, portfolio
    
    # USED
    def _get_ai_decisions_with_safety(
        self,
        news_list: List[Dict],
        market_data: Dict[str, Dict],
        portfolio: Dict
    ) -> Dict[str, Dict]:
        """Get AI decisions with multi-AI validation and risk management.
        
        Args:
            news_list: List of news articles.
            market_data: Market data dictionary.
            portfolio: Portfolio data.
            
        Returns:
            Final trading decisions.
        """
        # Get pattern learning lessons
        trading_lessons = self._get_pattern_learning_lessons()
        
        # Get initial decisions with trading lessons
        raw_decisions = self.ai_analyzer.analyze_market_data(
            news_list, market_data, portfolio, trading_lessons
        )
        
        # Apply trade history insights
        self._apply_trade_history_insights(raw_decisions)
        
        # Multi-AI validation
        validated_decisions = self.multi_ai_validator.cross_validate_multiple_decisions(
            raw_decisions, market_data, portfolio, news_list
        )
        
        # Apply adaptive risk management
        return self._apply_adaptive_risk_management(validated_decisions, portfolio, market_data)
    
    # USED
    def _apply_trade_history_insights(self, decisions: Dict[str, Dict]) -> None:
        """Apply trade history insights to influence AI decisions.
        
        Args:
            decisions: AI trading decisions.
        """
        for symbol, decision in decisions.items():
            # Get historical success rate
            success_rate = self.trade_analyzer.get_symbol_success_rate(symbol)
            
            # Check if we should avoid this symbol
            if self.trade_analyzer.should_avoid_symbol(symbol):
                logger.warning(f"⚠️ {symbol} has poor history ({success_rate}%), adjusting confidence")
                decision['confidence'] *= 0.5  # Reduce confidence
                decision['reason'] += f" (Historical success: {success_rate}%)"
            
            # Get specific insights for symbol
            insights = self.trade_analyzer.get_trading_insights_for_symbol(symbol)
            if insights.get('patterns', {}).get('recent_performance') == 'declining':
                decision['confidence'] *= 0.8
                logger.info(f"📉 {symbol} showing declining performance")
        
        # Log warnings
        warnings = self.trade_analyzer.get_failure_warnings()
        for warning in warnings:
            logger.warning(f"📊 Historical pattern warning: {warning}")
    
    # USED
    def _get_pattern_learning_lessons(self) -> List[str]:
        """Get learned patterns for AI analysis.
        
        Returns:
            List of trading lessons.
        """
        trading_lessons = self.pattern_learner.get_trading_lessons_for_prompt()
        logger.info(LOG_PATTERN_LESSONS.format(len(trading_lessons)))
        return trading_lessons
    
    def _apply_adaptive_risk_management(
        self,
        validated_decisions: Dict[str, Dict],
        portfolio: Dict,
        market_data: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Apply adaptive risk management to decisions.
        
        Args:
            validated_decisions: Validated trading decisions.
            portfolio: Portfolio data.
            market_data: Market data dictionary.
            
        Returns:
            Decisions with risk management applied.
        """
        final_decisions = {}
        
        for symbol, decision in validated_decisions.items():
            action = decision.get('action')
            
            # Apply position sizing for buy actions
            if action in [ACTION_BUY, ACTION_BUY_MORE]:
                decision = self._apply_position_sizing(decision, portfolio, market_data.get(symbol, {}))
            
            # Apply sell percentage calculation for partial sells
            elif action == ACTION_PARTIAL_SELL:
                decision = self._apply_sell_sizing(decision, portfolio, market_data.get(symbol, {}))
            
            # Add stop-loss recommendations for all positions
            if action in [ACTION_BUY, ACTION_BUY_MORE, ACTION_HOLD]:
                decision = self._add_stop_loss_recommendation(decision, market_data.get(symbol, {}))
            
            final_decisions[symbol] = decision
        
        return final_decisions
    
    def _apply_position_sizing(
        self,
        decision: Dict,
        portfolio: Dict,
        market_data: Dict
    ) -> Dict:
        """Apply position sizing and circuit breaker limits.
        
        Args:
            decision: Trading decision.
            portfolio: Portfolio data.
            market_data: Market data for the symbol.
            
        Returns:
            Decision with position sizing applied.
        """
        # Get validation result for risk management
        # Only use actual validation results, no fake defaults
        validation_result = {
            'approved': decision.get('multi_ai_approved'),
            'risk_level': decision.get('risk_level')
        }
        
        # Calculate optimal position size
        sizing_result = self.adaptive_risk_manager.calculate_optimal_position_size(
            decision, portfolio, market_data, validation_result
        )
        decision['position_sizing'] = sizing_result
        decision['recommended_amount_krw'] = sizing_result.get('recommended_size', MIN_VIABLE_TRADE_AMOUNT)
        
        # Apply circuit breaker limits
        decision = self._apply_circuit_breaker_limits(decision, portfolio)
        
        return decision
    
    def _apply_sell_sizing(
        self,
        decision: Dict,
        portfolio: Dict,
        market_data: Dict
    ) -> Dict:
        """Let AI determine sell percentage for partial sells.
        
        Args:
            decision: Trading decision.
            portfolio: Portfolio data.
            market_data: Market data for the symbol.
            
        Returns:
            Decision with AI-determined sell percentage.
        """
        # If AI already provided sell_percentage, use it
        if 'sell_percentage' in decision:
            return decision
        
        # Otherwise, ask AI to determine optimal sell percentage
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('current_price', 0)
            
            # Get asset info
            assets = portfolio.get('assets', {})
            asset_info = assets.get(symbol, {})
            profit_loss = asset_info.get('profit_loss_percentage', 0)
            
            # Prepare context for AI
            context = {
                'symbol': symbol,
                'current_price': current_price,
                'profit_loss_percentage': profit_loss,
                'risk_level': decision.get('risk_level', 'unknown'),
                'market_data': market_data,
                'original_reason': decision.get('reason', '')
            }
            
            # Ask AI for optimal sell percentage
            sell_percentage = self._get_ai_sell_percentage(context)
            
            decision['sell_percentage'] = sell_percentage
            decision['sell_reason'] = f"AI-optimized partial sell: {sell_percentage:.1%}"
            
        except Exception as e:
            logger.error(f"Failed to get AI sell percentage: {e}")
            # Fallback to conservative 10% if AI fails
            decision['sell_percentage'] = 0.10
            decision['sell_reason'] = "Fallback partial sell: 10%"
        
        return decision
    
    def _get_ai_sell_percentage(self, context: Dict[str, Any]) -> float:
        """Get AI-determined optimal sell percentage.
        
        Args:
            context: Market and portfolio context.
            
        Returns:
            Sell percentage (0.0-1.0).
        """
        prompt = f"""Based on the following trading context, determine the optimal percentage to sell:

Symbol: {context['symbol']}
Current Price: {context['current_price']:,.0f} KRW
Profit/Loss: {context['profit_loss_percentage']:.2f}%
Risk Level: {context['risk_level']}
Original Decision Reason: {context['original_reason']}

Market Indicators:
- RSI: {context['market_data'].get('rsi_14', 'N/A')}
- Volume Ratio: {context['market_data'].get('volume_ratio_24h_7d', 'N/A')}
- Price Change 24h: {context['market_data'].get('price_24h_change', 'N/A')}%

Determine the optimal sell percentage between 5% and 50%.
Consider:
1. If losing money, might need larger sell to cut losses
2. If profitable, might want smaller sell to keep gains running
3. Risk level and market conditions
4. Technical indicators suggesting reversal

Return ONLY a number between 0.05 and 0.50 (e.g., 0.15 for 15%).
"""
        
        try:
            result = self.ai_analyzer.openai_client.analyze_with_prompt(
                prompt=prompt,
                system_message="You are a trading expert. Return only a decimal number representing the sell percentage.",
                temperature=0.3
            )
            
            # Extract number from response
            import re
            match = re.search(r'0\.\d+', str(result))
            if match:
                percentage = float(match.group())
                # Ensure within bounds
                return max(0.05, min(0.50, percentage))
            
        except Exception as e:
            logger.error(f"AI sell percentage determination failed: {e}")
        
        # Default fallback
        return 0.10
    
    def _add_stop_loss_recommendation(
        self,
        decision: Dict,
        market_data: Dict
    ) -> Dict:
        """Let AI determine stop-loss recommendations.
        
        Args:
            decision: Trading decision.
            market_data: Market data for the symbol.
            
        Returns:
            Decision with AI-determined stop-loss recommendation.
        """
        # Skip if already has stop-loss recommendation
        if 'stop_loss_recommendation' in decision:
            return decision
        
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('current_price', 0)
            
            prompt = f"""Determine optimal stop-loss for this position:

Symbol: {symbol}
Current Price: {current_price:,.0f} KRW
Action: {decision.get('action', 'unknown')}
Decision Reason: {decision.get('reason', '')}

Market Data:
- Volatility 7d: {market_data.get('volatility_7d', 'N/A')}
- ATR 14: {market_data.get('atr_14', 'N/A')}
- Support Level 1: {market_data.get('support_1', 'N/A')}
- Support Level 2: {market_data.get('support_2', 'N/A')}
- RSI 14: {market_data.get('rsi_14', 'N/A')}
- Bollinger Lower: {market_data.get('bb_lower', 'N/A')}

Determine the optimal stop-loss percentage (how far below current price).
Consider:
1. Market volatility and ATR
2. Technical support levels
3. Risk tolerance based on action type
4. Market conditions

Return ONLY a decimal number between 0.02 and 0.15 (e.g., 0.05 for 5% stop-loss).
"""
            
            result = self.ai_analyzer.openai_client.analyze_with_prompt(
                prompt=prompt,
                system_message="You are a risk management expert. Return only a decimal number for stop-loss percentage.",
                temperature=0.2
            )
            
            # Extract number from response
            import re
            match = re.search(r'0\.\d+', str(result))
            if match:
                stop_loss_pct = float(match.group())
                stop_loss_pct = max(0.02, min(0.15, stop_loss_pct))  # 2%-15% range
            else:
                stop_loss_pct = 0.05  # Default 5%
            
            stop_loss_price = current_price * (1 - stop_loss_pct)
            
            decision['stop_loss_recommendation'] = {
                'price': stop_loss_price,
                'percentage': stop_loss_pct,
                'reason': f"AI-optimized stop-loss at {stop_loss_pct:.1%} below current price"
            }
            
        except Exception as e:
            logger.error(f"Failed to get AI stop-loss recommendation: {e}")
            # Conservative fallback
            decision['stop_loss_recommendation'] = {
                'price': current_price * 0.95,
                'percentage': 0.05,
                'reason': "Fallback stop-loss at 5% below current price"
            }
        
        return decision
    
    def _apply_circuit_breaker_limits(self, decision: Dict, portfolio: Dict) -> Dict:
        """Apply circuit breaker limits to decision.
        
        Args:
            decision: Trading decision.
            portfolio: Portfolio data.
            
        Returns:
            Decision with circuit breaker limits applied.
        """
        recent_decisions = self.data_store.get_recent_trades(
            limit=HEALTH_CHECK_TRADE_LIMIT_CIRCUIT,
            hours=HEALTH_CHECK_TRADE_HOURS
        )
        
        circuit_result = self.adaptive_risk_manager.apply_circuit_breaker_limits(
            decision.get('recommended_amount_krw', MIN_VIABLE_TRADE_AMOUNT),
            portfolio,
            recent_decisions
        )
        
        if circuit_result.get('circuit_breaker_triggered', False):
            decision['circuit_breaker'] = circuit_result
            decision['recommended_amount_krw'] = circuit_result.get('adjusted_size', 0)
            
            if circuit_result.get('adjusted_size', 0) < MIN_VIABLE_TRADE_AMOUNT:
                decision['action'] = ACTION_HOLD
                decision['reason'] = f"Circuit breaker: {circuit_result.get('reason', 'Trade size too small')}"
        
        return decision
    
    # USED
    def _apply_risk_management(
        self,
        decisions: Dict[str, Dict],
        portfolio: Dict,
        market_data: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Apply risk monitoring and stop-loss checks.
        
        Args:
            decisions: Trading decisions.
            portfolio: Portfolio data.
            market_data: Market data dictionary.
            
        Returns:
            Decisions with risk management applied.
        """
        # Monitor portfolio risks
        self._monitor_portfolio_risks(portfolio, market_data)
        
        # Check and apply stop-loss triggers
        decisions = self._apply_stop_loss_triggers(decisions, portfolio, market_data)
        
        return decisions
    
    # USED
    def _monitor_portfolio_risks(self, portfolio: Dict, market_data: Dict[str, Dict]) -> None:
        """Monitor portfolio risks and log warnings.
        
        Args:
            portfolio: Portfolio data.
            market_data: Market data dictionary.
        """
        recent_trades = self.data_store.get_recent_trades(
            limit=HEALTH_CHECK_RECENT_TRADES_LIMIT,
            hours=HEALTH_CHECK_TRADE_HOURS
        )
        
        risk_assessment = self.risk_monitor.monitor_active_positions(
            portfolio, market_data, recent_trades
        )
        
        if risk_assessment.get('overall_risk') in [RISK_LEVEL_HIGH, RISK_LEVEL_CRITICAL]:
            logger.warning(LOG_HIGH_RISK_DETECTED.format(risk_assessment.get('overall_risk')))
            logger.info(LOG_AI_RISK_ASSESSMENT.format(
                risk_assessment.get('ai_assessment', 'No assessment')
            ))
    
    # USED
    def _apply_stop_loss_triggers(
        self,
        decisions: Dict[str, Dict],
        portfolio: Dict,
        market_data: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Check and apply stop-loss triggers.
        
        Args:
            decisions: Trading decisions.
            portfolio: Portfolio data.
            market_data: Market data dictionary.
            
        Returns:
            Decisions with stop-loss triggers applied.
        """
        stop_loss_triggers = self.risk_monitor.check_stop_loss_triggers(portfolio, market_data)
        
        if stop_loss_triggers:
            logger.warning(LOG_STOP_LOSS_TRIGGERED.format(len(stop_loss_triggers)))
            
            for trigger in stop_loss_triggers:
                symbol = trigger['symbol']
                # Override decision with stop-loss
                decisions[symbol] = {
                    'action': trigger['recommended_action'],
                    'reason': f"Stop-loss triggered: {trigger['reason']}",
                    'confidence': STOP_LOSS_CONFIDENCE,
                    'stop_loss_trigger': trigger
                }
        
        return decisions
    
    # USED
    def _execute_trades_with_analysis(
        self,
        decisions: Dict[str, Dict],
        market_data: Dict[str, Dict],
        portfolio: Dict
    ) -> None:
        """Execute trades and perform post-trade analysis.
        
        Args:
            decisions: Trading decisions.
            market_data: Market data dictionary.
            portfolio: Portfolio data.
        """
        # Execute trades
        self.execute_trading_decisions(decisions)
        
        # Post-trade analysis for stop-loss actions
        self._perform_post_trade_analysis(decisions, market_data, portfolio)
    
    def _perform_post_trade_analysis(
        self,
        decisions: Dict[str, Dict],
        market_data: Dict[str, Dict],
        portfolio: Dict
    ) -> None:
        """Perform post-trade analysis for stop-loss actions.
        
        Args:
            decisions: Trading decisions.
            market_data: Market data dictionary.
            portfolio: Portfolio data.
        """
        for symbol, decision in decisions.items():
            if self._is_stop_loss_decision(decision):
                self._analyze_stop_loss_decision(symbol, decision, market_data, portfolio)
    
    def _is_stop_loss_decision(self, decision: Dict) -> bool:
        """Check if decision is a stop-loss action.
        
        Args:
            decision: Trading decision.
            
        Returns:
            True if stop-loss decision, False otherwise.
        """
        return (
            decision.get('stop_loss_trigger') and
            decision.get('action') in [ACTION_SELL_ALL, ACTION_PARTIAL_SELL]
        )
    
    def _analyze_stop_loss_decision(
        self,
        symbol: str,
        decision: Dict,
        market_data: Dict[str, Dict],
        portfolio: Dict
    ) -> None:
        """Analyze stop-loss decision.
        
        Args:
            symbol: Cryptocurrency symbol.
            decision: Trading decision.
            market_data: Market data dictionary.
            portfolio: Portfolio data.
        """
        try:
            # Get original trade data (simplified - would need better tracking)
            original_trade_data = {
                'action': ACTION_BUY,
                'timestamp': datetime.now().isoformat(),
                'price': market_data.get(symbol, {}).get('current_price', 0)
            }
            
            # Perform post-trade analysis
            analysis_report = self.post_trade_analyzer.analyze_stop_loss_decision(
                symbol=symbol,
                stop_loss_action=decision['action'],
                original_trade_data=original_trade_data,
                current_market_data=market_data.get(symbol, {}),
                portfolio_context=portfolio
            )
            
            # Log analysis results
            logger.info(LOG_POST_TRADE_COMPLETE.format(symbol))
            logger.info(LOG_FAILURE_SEVERITY.format(
                analysis_report.get('failure_severity', RISK_LEVEL_UNKNOWN)
            ))
            logger.info(LOG_LESSONS_LEARNED.format(
                len(analysis_report.get('lessons_learned', []))
            ))
            
        except Exception as e:
            logger.error(LOG_POST_TRADE_FAILED.format(symbol, e))
    
    def _complete_trading_cycle(
        self,
        cycle_start_time: float,
        news_list: List[Dict],
        symbols: List[str],
        market_data: Dict[str, Dict],
        portfolio: Dict,
        decisions: Dict[str, Dict]
    ) -> None:
        """Complete the trading cycle and record results.
        
        Args:
            cycle_start_time: Timestamp when cycle started.
            news_list: List of news articles.
            symbols: List of extracted symbols.
            market_data: Market data dictionary.
            portfolio: Portfolio data.
            decisions: Trading decisions executed.
        """
        analysis_duration = time.time() - cycle_start_time
        
        # Record complete analysis result
        self._record_analysis_result(
            news_list, symbols, market_data, portfolio, decisions, analysis_duration
        )
        
        # Record individual trades
        self._record_individual_trades(decisions, market_data)
        
        # Monitor and log portfolio status
        # Detailed report every 10 cycles, simple report otherwise
        detailed = (self._cycle_count % 10 == 0)
        self._monitor_portfolio_status(detailed=detailed)
        
        # Complete cycle
        logger.info(LOG_CYCLE_COMPLETE)
    
    def _record_analysis_result(
        self,
        news_list: List[Dict],
        symbols: List[str],
        market_data: Dict[str, Dict],
        portfolio: Dict,
        decisions: Dict[str, Dict],
        analysis_duration: float
    ) -> None:
        """Record complete analysis result.
        
        Args:
            news_list: List of news articles.
            symbols: List of extracted symbols.
            market_data: Market data dictionary.
            portfolio: Portfolio data.
            decisions: Trading decisions.
            analysis_duration: Duration of analysis in seconds.
        """
        analysis_result = AIAnalysisResult(
            timestamp=datetime.now().isoformat(),
            news_count=len(news_list),
            extracted_symbols=symbols,
            market_data_symbols=list(market_data.keys()),
            decisions=decisions,
            analysis_duration=analysis_duration,
            portfolio_value=portfolio.get('total_balance', 0),
            circuit_breaker_status=self.circuit_breaker.get_status()
        )
        
        self.data_store.record_ai_analysis(analysis_result)
    
    def _record_individual_trades(self, decisions: Dict[str, Dict], market_data: Dict[str, Dict]) -> None:
        """Record individual trade records.
        
        Args:
            decisions: Trading decisions.
            market_data: Market data dictionary.
        """
        for symbol, decision in decisions.items():
            # Ensure confidence is not None
            confidence = decision.get('confidence', DEFAULT_CONFIDENCE)
            if confidence is None:
                confidence = DEFAULT_CONFIDENCE
            
            trade_record = TradeRecord(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                action=decision.get('action', DEFAULT_ACTION),
                confidence=confidence,
                reason=decision.get('reason', ''),
                price=market_data.get(symbol, {}).get('current_price', 0.0)
            )
            self.data_store.record_trade(trade_record)
    
    def _monitor_portfolio_status(self, detailed: bool = True) -> None:
        """Monitor and log comprehensive portfolio status.
        
        Args:
            detailed: If True, show detailed report. If False, show summary only.
        """
        try:
            portfolio = self.trader.get_portfolio_status()
            
            if not portfolio or not portfolio.get('assets'):
                logger.info("📊 Portfolio is empty or unavailable")
                return
            
            # Simple summary for regular cycles
            if not detailed:
                total_value = portfolio.get('total_krw', 0)
                total_investment = portfolio.get('total_investment', 0)
                total_profit_pct = ((total_value - total_investment) / total_investment * 100) if total_investment > 0 else 0
                
                assets = portfolio.get('assets', {})
                coin_count = sum(1 for k, v in assets.items() if k != 'KRW' and v.get('balance', 0) > 0)
                
                logger.info(
                    f"📊 Portfolio: {total_value:,.0f}원 ({total_profit_pct:+.2f}%) | "
                    f"{coin_count} coins | KRW: {assets.get('KRW', {}).get('balance', 0):,.0f}원"
                )
                return
            
            # Detailed report for every 10th cycle
            logger.info("=" * 60)
            logger.info("📊 PORTFOLIO STATUS REPORT (Detailed)")
            logger.info("=" * 60)
            
            # Total portfolio value
            total_value = portfolio.get('total_krw', 0)
            total_investment = portfolio.get('total_investment', 0)
            total_profit = total_value - total_investment
            total_profit_pct = (total_profit / total_investment * 100) if total_investment > 0 else 0
            
            logger.info(f"💰 Total Value: {total_value:,.0f}원")
            logger.info(f"💵 Total Investment: {total_investment:,.0f}원")
            logger.info(f"📈 Total P&L: {total_profit:+,.0f}원 ({total_profit_pct:+.2f}%)")
            
            # Individual coin status
            assets = portfolio.get('assets', {})
            coin_count = sum(1 for k, v in assets.items() if k != 'KRW' and v.get('balance', 0) > 0)
            
            if coin_count > 0:
                logger.info(f"\n🪙 Holding {coin_count} coins:")
                logger.info("-" * 60)
                
                # Sort by value for better readability
                sorted_assets = sorted(
                    [(k, v) for k, v in assets.items() if k != 'KRW' and v.get('balance', 0) > 0],
                    key=lambda x: x[1].get('krw_value', 0),
                    reverse=True
                )
                
                for symbol, asset in sorted_assets:
                    balance = asset.get('balance', 0)
                    avg_price = asset.get('avg_buy_price', 0)
                    current_price = asset.get('current_price', 0)
                    krw_value = asset.get('krw_value', 0)
                    profit_loss = asset.get('profit_loss', 0)
                    profit_loss_pct = asset.get('profit_loss_percentage', 0)
                    
                    # Emoji based on profit/loss
                    emoji = "🟢" if profit_loss_pct > 0 else "🔴" if profit_loss_pct < 0 else "⚪"
                    
                    logger.info(
                        f"{emoji} {symbol}: {balance:.8f} @ {avg_price:,.0f}원 → {current_price:,.0f}원 "
                        f"| Value: {krw_value:,.0f}원 | P&L: {profit_loss:+,.0f}원 ({profit_loss_pct:+.2f}%)"
                    )
                    
                    # Add trade history insights
                    success_rate = self.trade_analyzer.get_symbol_success_rate(symbol)
                    if success_rate > 0:
                        logger.info(f"   📊 Historical success rate: {success_rate}%")
            
            # Available KRW
            krw_balance = assets.get('KRW', {}).get('balance', 0)
            logger.info(f"\n💵 Available KRW: {krw_balance:,.0f}원")
            
            # Trading performance summary
            recent_trades = self.data_store.get_recent_trades(limit=10, hours=24)
            if recent_trades:
                wins = sum(1 for t in recent_trades if t.get('success'))
                logger.info(f"\n📈 Last 24h: {wins}/{len(recent_trades)} successful trades")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Portfolio monitoring error: {e}")
    
    # USED
