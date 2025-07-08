"""AI Analyzer for cryptocurrency trading decisions.

This module provides AI-powered analysis for trading decisions,
including symbol extraction from news and market data analysis.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.shared.openai_client import OpenAIClient
import pyupbit
# Analysis constants
MAX_NEWS_FOR_ANALYSIS = 10
MAX_NEWS_FOR_SUMMARY = 5
MAX_SYMBOLS_TO_EXTRACT = 10
DEFAULT_CONFIDENCE = 0.5
SYMBOL_EXTRACTION_TEMPERATURE = 0.1
TRADING_DECISION_TEMPERATURE = 0.3


# Valid trading actions
VALID_TRADING_ACTIONS = [
    'buy', 'sell_all', 'partial_sell', 'hold', 'buy_more'
]

# Response parsing keys
RESPONSE_KEYS_TO_TRY = ['response', 'result', 'data', 'symbols']
RESPONSE_SKIP_KEYS = ['response', 'status', 'error']


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AIAnalyzer:
    """Analyzer for cryptocurrency trading using AI.
    
    This class provides AI-powered analysis capabilities for:
    - Extracting cryptocurrency symbols from news
    - Analyzing market data for trading decisions
    - Generating trading recommendations
    """
    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize AI analyzer.
        
        Args:
            api_key: OpenAI API key
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.openai_client = OpenAIClient(api_key=api_key)
        self.valid_symbols = self._load_valid_symbols()
        logger.info(f"AI Analyzer initialized with {len(self.valid_symbols)} valid symbols")
    
    def _load_valid_symbols(self) -> set:
        """Load valid cryptocurrency symbols from Upbit.
        
        Returns:
            Set of valid cryptocurrency symbols.
            
        Raises:
            Exception: If failed to load symbols from Upbit.
        """
        try:
            markets = pyupbit.get_tickers(fiat="KRW")
            # Extract symbol part after 'KRW-'
            symbols = {market.split("-")[1] for market in markets if market.startswith("KRW-")}
            logger.info(f"Loaded {len(symbols)} valid symbols from Upbit")
            return symbols
        except Exception as e:
            logger.error(f"Failed to load Upbit symbols: {e}")
            raise
    
    # USED
    def extract_symbols_from_news(self, news_list: List[Dict[str, Any]]) -> List[str]:
        """Extract cryptocurrency symbols from news articles.
        
        Args:
            news_list: List of news articles with 'title' and 'summary' fields.
            
        Returns:
            List of extracted cryptocurrency symbols (e.g., ['BTC', 'ETH']).
        """
        if not news_list:
            return []
        
        news_content = "\n".join([
            f"Title: {article.get('title', '')}\nSummary: {article.get('summary', '')[:300]}"
            for article in news_list[:MAX_NEWS_FOR_ANALYSIS]
        ])
        
        system_message = self._get_symbol_extraction_system_message()
        prompt = self._create_symbol_extraction_prompt(news_content)
        
        try:
            result = self.openai_client.analyze_with_prompt(
                prompt=prompt,
                system_message=system_message,
                temperature=SYMBOL_EXTRACTION_TEMPERATURE
            )
            
            logger.debug(f"Raw AI response: {result}")
            symbols = self._parse_symbol_extraction_response(result)
            
            if not symbols:
                logger.warning("No symbols extracted from AI response")
            
            valid_symbols = self._validate_and_clean_symbols(symbols)
            
            logger.info(f"Extracted {len(valid_symbols)} symbols from {len(news_list)} news articles")
            return valid_symbols[:MAX_SYMBOLS_TO_EXTRACT]
            
        except Exception as e:
            logger.error(f"Failed to extract symbols from news: {e}")
            return []
    
    # USED
    def analyze_market_data(
        self,
        news_list: List[Dict[str, Any]],
        market_data: Dict[str, Dict[str, Any]],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze market data and generate trading decisions.
        
        Args:
            news_list: List of news articles.
            market_data: Market data for each symbol.
            portfolio: Current portfolio status.
            
        Returns:
            Dictionary of trading decisions for each symbol.
        """
        
        market_summary = self._prepare_market_summary(market_data)
        news_summary = self._prepare_news_summary(news_list)
        portfolio_summary = self._prepare_portfolio_summary(portfolio)
        
        system_message = self._get_trading_analysis_system_message()
        prompt = self._create_trading_analysis_prompt(
            portfolio_summary, market_summary, news_summary
        )
        
        try:
            result = self.openai_client.analyze_with_prompt(
                prompt=prompt,
                system_message=system_message,
                temperature=TRADING_DECISION_TEMPERATURE
            )
            
            logger.debug(f"Raw AI trading decision response: {result}")
            decisions = self._parse_trading_decisions_response(result)
            
            logger.info(f"Generated {len(decisions)} trading decisions")
            
            if not decisions:
                logger.warning(f"No valid trading decisions extracted from AI response: {result}")
            
            return decisions
            
        except Exception as e:
            logger.error(f"Failed to analyze market data: {e}")
            return {}
    
    # USED
    def _prepare_market_summary(self, market_data: Dict[str, Dict[str, Any]]) -> str:
        """Prepare market data summary for AI analysis."""
        lines = []
        for symbol, data in market_data.items():
            price = data.get('current_price', 0)
            change = data.get('price_change_24h', 0)
            volume = data.get('volume_24h', 0)
            
            # Include comprehensive analysis if available
            analysis = data.get('comprehensive_analysis', {})
            trend = analysis.get('trend', 'unknown')
            signal = analysis.get('signal', 'neutral')
            
            # Check if this asset is held in portfolio
            is_held = data.get('is_held', False)
            held_indicator = " [HELD]" if is_held else " [NOT HELD]"
            
            # Format price appropriately for small values
            if price >= 1:
                price_str = f"{price:,.0f}"
            elif price >= 0.01:
                price_str = f"{price:.5f}"
            elif price >= 0.001:
                price_str = f"{price:.6f}"
            else:
                price_str = f"{price:.8f}"
            
            lines.append(
                f"{symbol}{held_indicator}: Price={price_str} KRW, "
                f"24h Change={change:.2f}%, Volume={volume:,.0f} KRW, "
                f"Trend={trend}, Signal={signal}"
            )
        
        return "\n".join(lines)
    
    def _prepare_news_summary(self, news_list: List[Dict[str, Any]]) -> str:
        """Prepare news summary for AI analysis.
        
        Args:
            news_list: List of news articles.
            
        Returns:
            Formatted news summary string.
        """
        if not news_list:
            return "No recent news available."
        
        lines = []
        for article in news_list[:MAX_NEWS_FOR_SUMMARY]:
            title = article.get('title', '')
            summary = article.get('summary', '')
            lines.append(f"- {title}: {summary}")
        
        return "\n".join(lines)
    
    def _prepare_portfolio_summary(self, portfolio: Dict[str, Any]) -> str:
        """Prepare portfolio summary for AI analysis."""
        total_balance = portfolio.get('total_balance', 0)
        available_krw = portfolio.get('available_krw', portfolio.get('krw_balance', 0))
        
        lines = [
            f"Total Balance: {total_balance:,.0f} KRW",
            f"Available KRW: {available_krw:,.0f} KRW",
            "",
            "CURRENTLY HELD ASSETS (You own these):"
        ]
        
        assets = portfolio.get('assets', {})
        held_assets = []
        
        # Separate held assets (excluding KRW)
        for symbol, asset in assets.items():
            if symbol == 'KRW':
                continue
            balance = asset.get('balance', 0)
            if balance > 0:
                value = asset.get('current_value', 0)
                profit_loss = asset.get('profit_loss_percentage', 0)
                held_assets.append(symbol)
                lines.append(
                    f"  - {symbol}: {balance:.6f} units, "
                    f"Value={value:,.0f} KRW, P/L={profit_loss:.2f}%"
                )
        
        if not held_assets:
            lines.append("  - No cryptocurrency holdings")
        
        lines.append("")
        lines.append("AVAILABLE ACTIONS BY ASSET TYPE:")
        lines.append("- For HELD assets: hold, sell_all, partial_sell, buy_more")
        lines.append("- For NOT HELD assets: buy, skip")
        
        return "\n".join(lines)
    
    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate trading decision structure.
        
        Args:
            decision: Trading decision dictionary.
            
        Returns:
            True if decision is valid, False otherwise.
        """
        required_fields = ['action', 'reason']
        
        # Check required fields
        for field in required_fields:
            if field not in decision:
                return False
        
        # Validate action
        if decision['action'] not in VALID_TRADING_ACTIONS:
            return False
        
        # Normalize confidence value
        self._normalize_confidence(decision)
        
        return True
    
    def _normalize_confidence(self, decision: Dict[str, Any]) -> None:
        """Normalize confidence value in decision.
        
        Args:
            decision: Trading decision dictionary (modified in place).
        """
        if 'confidence' in decision:
            try:
                confidence = float(decision['confidence'])
                decision['confidence'] = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                decision['confidence'] = DEFAULT_CONFIDENCE
        else:
            decision['confidence'] = DEFAULT_CONFIDENCE
    
    # USED
    def _get_symbol_extraction_system_message(self) -> str:
        """Get system message for symbol extraction.
        
        Returns:
            System message string.
        """
        return """You are a cryptocurrency market analyst. Extract cryptocurrency symbols from news articles.
        
        CRITICAL: Extract EVERY cryptocurrency mentioned, no matter how briefly!
        
        Rules:
        - Check BOTH titles and summaries carefully
        - Look for cryptocurrency names in ANY form:
          * Full names: Bitcoin, Ethereum, Dogecoin, Sonic
          * Possessive: Bitcoin's, Ethereum's, Sonic's
          * Partial: "BTC price", "ETH growth", "DOGE surge"
          * Context: "token unlock" usually refers to a specific crypto
        
        Common mappings:
        - Bitcoin/BTC, Ethereum/ETH, Dogecoin/DOGE
        - Sonic/S, Ripple/XRP, Cardano/ADA
        - Solana/SOL, Polygon/MATIC, Avalanche/AVAX
        
        Return ALL symbols in uppercase. Be EXHAUSTIVE - missing symbols means missing trading opportunities!
        """
    
    # USED
    def _create_symbol_extraction_prompt(self, news_content: str) -> str:
        """Create prompt for symbol extraction.
        
        Args:
            news_content: Formatted news content.
            
        Returns:
            Prompt string.
        """
        return f"""Extract cryptocurrency symbols from these news articles:

{news_content}

IMPORTANT: Look for ALL cryptocurrencies mentioned, including:
- Full names: Bitcoin → BTC, Ethereum → ETH, Dogecoin → DOGE, Sonic → S
- Possessive forms: Bitcoin's → BTC, Sonic's → S
- Be thorough - extract EVERY cryptocurrency mentioned in titles and summaries

Examples:
- "Bitcoin price rises" → BTC
- "Ethereum's future" → ETH  
- "Dogecoin Social Surge" → DOGE
- "Sonic's $74.59M token unlock" → S

Return a JSON array of ALL symbols found. Example: ["BTC", "ETH", "DOGE", "S"]
If no symbols are found, return an empty array: []"""
    
    # USED 
    def _parse_symbol_extraction_response(
        self, result: Any
    ) -> List[str]:
        """Parse symbol extraction response from AI.
        
        Args:
            result: Raw AI response.
            
        Returns:
            List of extracted symbols.
        """
        symbols = []
        
        if isinstance(result, str):
            symbols = self._parse_symbols_from_string(result)
        elif isinstance(result, dict) and 'symbols' in result:
            symbols = result['symbols']
        elif isinstance(result, list):
            symbols = result
        elif isinstance(result, dict):
            symbols = self._parse_symbols_from_dict(result)
        
        return symbols
    
    def _parse_symbols_from_string(self, result: str) -> List[str]:
        """Parse symbols from string response.
        
        Args:
            result: String response from AI.
            
        Returns:
            List of extracted symbols.
        """
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Extract symbols using regex
            matches = re.findall(r'\b[A-Z]{2,10}\b', result)
            # Filter only valid Upbit symbols
            return [m for m in matches if m in self.valid_symbols]
    
    def _parse_symbols_from_dict(self, result: Dict[str, Any]) -> List[str]:
        """Parse symbols from dictionary response.
        
        Args:
            result: Dictionary response from AI.
            
        Returns:
            List of extracted symbols.
        """
        for key in RESPONSE_KEYS_TO_TRY:
            if key in result:
                value = result[key]
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    parsed = self._try_parse_json_from_value(value)
                    if parsed:
                        return parsed
        return []
    
    def _try_parse_json_from_value(self, value: str) -> List[str]:
        """Try to parse JSON array from string value.
        
        Args:
            value: String value to parse.
            
        Returns:
            List of symbols or empty list.
        """
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            # Look for JSON array pattern
            array_match = re.search(r'\[([^\]]+)\]', value)
            if array_match:
                try:
                    return json.loads(array_match.group(0))
                except json.JSONDecodeError:
                    pass
        return []
    
    # USED
    def _validate_and_clean_symbols(self, symbols: List[Any]) -> List[str]:
        """Validate and clean extracted symbols.
        
        Args:
            symbols: Raw list of symbols.
            
        Returns:
            Cleaned list of valid symbols with duplicates removed.
        """
        valid_symbols = []
        seen_symbols = set()
        for symbol in symbols:
            if isinstance(symbol, str) and symbol.strip():
                clean_symbol = symbol.upper().strip()
                if clean_symbol not in seen_symbols:
                    valid_symbols.append(clean_symbol)
                    seen_symbols.add(clean_symbol)
        return valid_symbols
    
    def _get_trading_analysis_system_message(self) -> str:
        """Get system message for trading analysis.
        
        Returns:
            System message string.
        """
        return """You are an expert cryptocurrency trader. Analyze market data and provide trading decisions.
        
        For each cryptocurrency, provide appropriate action based on holdings:
        
        If CURRENTLY HOLDING the asset:
        - action: "hold", "sell_all", "partial_sell", or "buy_more"
        
        If NOT HOLDING the asset:
        - action: "buy" or "skip"
        
        For ALL decisions provide:
        - reason: Clear explanation for the decision
        - confidence: Float between 0.0 and 1.0
        
        IMPORTANT: Check portfolio holdings first. You cannot "hold" an asset you don't own.
        
        Consider:
        - Technical indicators and market trends
        - News sentiment and impact
        - Current portfolio holdings (CRITICAL)
        - Risk management principles
        - Available capital for new positions
        """
    
    def _create_trading_analysis_prompt(
        self,
        portfolio_summary: str,
        market_summary: str,
        news_summary: str
    ) -> str:
        """Create prompt for trading analysis.
        
        Args:
            portfolio_summary: Portfolio status summary.
            market_summary: Market data summary.
            news_summary: News summary.
            
        Returns:
            Prompt string.
        """
        return f"""Analyze this market data and provide trading decisions:

PORTFOLIO STATUS:
{portfolio_summary}

MARKET DATA:
{market_summary}

RECENT NEWS:
{news_summary}

CRITICAL RULES FOR TRADING DECISIONS:
1. CHECK PORTFOLIO STATUS FIRST - See "CURRENTLY HELD ASSETS" section
2. For assets marked [HELD] or listed in "CURRENTLY HELD ASSETS":
   - Valid actions: "hold", "sell_all", "partial_sell", "buy_more"
   - CANNOT use: "buy" (you already own it!)
3. For assets marked [NOT HELD] or NOT in "CURRENTLY HELD ASSETS":
   - Valid actions: "buy", "skip"  
   - CANNOT use: "hold", "sell_all", "partial_sell" (you don't own it!)
4. NEVER suggest "hold" for an asset you don't currently own

Provide trading decisions in JSON format:
{{
    "SYMBOL": {{
        "action": "appropriate_action_based_on_holdings",
        "reason": "Clear explanation",
        "confidence": 0.0-1.0
    }}
}}"""
    
    def _parse_trading_decisions_response(
        self, result: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Parse trading decisions from AI response.
        
        Args:
            result: Raw AI response.
            
        Returns:
            Dictionary of valid trading decisions.
        """
        decisions = {}
        
        if isinstance(result, dict):
            # Try to extract from response key
            if 'response' in result and isinstance(result['response'], str):
                extracted = self._extract_json_from_response(result['response'])
                if extracted:
                    result = extracted
            
            # Process the result
            for key, value in result.items():
                if key not in RESPONSE_SKIP_KEYS:
                    if isinstance(value, dict) and self._validate_decision(value):
                        decisions[key] = value
        
        return decisions
    
    def _extract_json_from_response(self, response_str: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response string.
        
        Args:
            response_str: Response string potentially containing JSON.
            
        Returns:
            Extracted dictionary or None.
        """
        try:
            # Extract from markdown code block
            if '```json' in response_str and '```' in response_str:
                start = response_str.find('```json') + 7
                end = response_str.rfind('```')
                response_str = response_str[start:end].strip()
            elif '```' in response_str:
                start = response_str.find('```') + 3
                end = response_str.rfind('```')
                response_str = response_str[start:end].strip()
            
            parsed = json.loads(response_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response string: {e}")
        
        return None