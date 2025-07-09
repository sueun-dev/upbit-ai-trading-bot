"""Multi-AI Validator for cross-validating trading decisions.

This module provides multiple AI perspectives to validate trading decisions,
reducing single-point-of-failure risks and improving decision quality.
"""

import logging
from typing import Any, Dict, List

import numpy as np

from src.shared.openai_client import OpenAIClient

# Validation constants
DEFAULT_CONFIDENCE = 0.5
DEFAULT_APPROVAL = True
MAJORITY_REJECTION_THRESHOLD = 0.5
STRONG_REJECTION_THRESHOLD = 0.25

# Temperature settings for different validators
CONSERVATIVE_TEMPERATURE = 0.2
AGGRESSIVE_TEMPERATURE = 0.5
TECHNICAL_TEMPERATURE = 0.1
SENTIMENT_TEMPERATURE = 0.3

# Action types
BUY_ACTIONS = ['buy', 'buy_more']
SELL_ACTIONS = ['sell_all', 'partial_sell']
HOLD_ACTION = 'hold'

# Market thresholds
HIGH_VOLUME_THRESHOLD = 1.5
NEWS_LIMIT = 5
NEWS_SUMMARY_LENGTH = 100

logger = logging.getLogger(__name__)


class MultiAIValidator:
    """Multi-AI validation system for trading decisions.
    
    Uses multiple AI perspectives to validate and improve trading decisions:
    - Conservative Risk Assessor
    - Aggressive Growth Seeker
    - Technical Analysis Expert
    - Market Sentiment Analyzer
    - Final Arbitrator
    """
    
    # USED
    def __init__(self, api_key: str) -> None:
        """Initialize the multi-AI validator.
        
        Args:
            api_key: OpenAI API key
        """
        self.openai_client = OpenAIClient(api_key=api_key)
        self.validators = {
            'conservative': self._validate_conservative,
            'aggressive': self._validate_aggressive,
            'technical': self._validate_technical,
            'sentiment': self._validate_sentiment
        }
    
    # USED
    def cross_validate_multiple_decisions(
        self,
        decisions: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Dict[str, Any]],
        portfolio: Dict[str, Any],
        news_list: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Cross-validate multiple trading decisions using different AI perspectives.
        
        Args:
            decisions: Original trading decisions
            market_data: Market data for each symbol
            portfolio: Current portfolio status
            news_list: Recent news articles
            
        Returns:
            Validated and potentially modified decisions
        """
        validated_decisions = {}
        
        for symbol, decision in decisions.items():
            try:
                # Collect validations from different perspectives
                validations = self._collect_validations(
                    symbol, decision, market_data.get(symbol, {}), portfolio, news_list
                )
                
                # Aggregate validations
                final_decision = self._aggregate_validations(decision, validations, symbol, portfolio)
                
                # Add validation metadata
                final_decision['validation_results'] = validations
                final_decision['validators_agreed'] = self._check_consensus(validations)
                
                validated_decisions[symbol] = final_decision
            except Exception as e:
                logger.error(f"Failed to validate decision for {symbol}: {e}")
                # Keep original decision if validation fails
                validated_decisions[symbol] = decision
        
        return validated_decisions
    
    def _collect_validations(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any],
        portfolio: Dict[str, Any],
        news_list: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect validations from all validators.
        
        Args:
            symbol: Trading symbol.
            decision: Original trading decision.
            market_data: Market data for the symbol.
            portfolio: Current portfolio status.
            news_list: Recent news articles.
            
        Returns:
            Dictionary of validation results.
        """
        validations = {}
        for validator_name, validator_func in self.validators.items():
            try:
                validation = validator_func(symbol, decision, market_data, portfolio, news_list)
                validations[validator_name] = validation
            except Exception as e:
                logger.warning(f"{validator_name} validation failed for {symbol}: {e}")
                validations[validator_name] = self._get_default_validation()
        
        return validations
    
    def _get_default_validation(self) -> Dict[str, Any]:
        """Get default validation result for error cases.
        
        Returns:
            Default validation dictionary.
        """
        return {
            'approved': DEFAULT_APPROVAL,
            'confidence': DEFAULT_CONFIDENCE,
            'reason': 'Validation error'
        }
    
    # USED
    def _validate_conservative(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any],
        portfolio: Dict[str, Any],
        news_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Conservative risk assessor perspective."""
        system_message = """You are a conservative risk manager for a cryptocurrency trading system.
        Your role is to protect capital and minimize losses. You are skeptical of aggressive trades
        and prefer steady, low-risk opportunities. Review the proposed trading decision and assess
        whether it aligns with conservative risk management principles."""
        
        prompt = f"""Review this trading decision for {symbol}:
        
            Decision: {decision.get('action')}
            Reason: {decision.get('reason')}
            Confidence: {decision.get('confidence', DEFAULT_CONFIDENCE)}
            
            Current Price: {market_data.get('current_price', 0):,.0f} KRW
            24h Change: {market_data.get('price_24h_change', 0):.2f}%
            RSI: {market_data.get('rsi', {}).get('rsi_14', 50)}
            Portfolio Value: {portfolio.get('total_balance', 0):,.0f} KRW
            
            Should this trade be approved from a conservative perspective?
            Respond with JSON: {{"approved": true/false, "confidence": 0.0-1.0, "reason": "explanation"}}"""
        
        return self.openai_client.analyze_with_prompt(
            prompt=prompt,
            system_message=system_message,
            temperature=CONSERVATIVE_TEMPERATURE
        )
    
    def _validate_aggressive(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any],
        portfolio: Dict[str, Any],  # pylint: disable=unused-argument
        news_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggressive growth seeker perspective."""
        system_message = """You are an aggressive growth trader for a cryptocurrency trading system.
        Your role is to identify high-potential opportunities and maximize returns. You favor bold
        moves when market conditions show promise. Review the proposed trading decision and assess
        whether it's aggressive enough to capture potential gains."""
        
        prompt = self._create_aggressive_prompt(symbol, decision, market_data, news_list)
        
        return self.openai_client.analyze_with_prompt(
            prompt=prompt,
            system_message=system_message,
            temperature=AGGRESSIVE_TEMPERATURE
        )
    
    def _validate_technical(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any],
        portfolio: Dict[str, Any],  # pylint: disable=unused-argument
        news_list: List[Dict[str, Any]]  # pylint: disable=unused-argument
    ) -> Dict[str, Any]:
        """Technical analysis expert perspective."""
        system_message = """You are a technical analysis expert for cryptocurrency trading.
        Your role is to evaluate trades based purely on technical indicators, chart patterns,
        and quantitative metrics. Ignore news and sentiment - focus only on the numbers."""
        
        prompt = self._create_technical_prompt(symbol, decision, market_data)
        
        return self.openai_client.analyze_with_prompt(
            prompt=prompt,
            system_message=system_message,
            temperature=TECHNICAL_TEMPERATURE
        )
    
    def _validate_sentiment(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any],
        portfolio: Dict[str, Any],  # pylint: disable=unused-argument
        news_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Market sentiment analyzer perspective."""
        system_message = """You are a market sentiment analyst for cryptocurrency trading.
        Your role is to evaluate trades based on news sentiment, market psychology, and
        social indicators. Focus on the narrative and emotional aspects of the market."""
        
        prompt = self._create_sentiment_prompt(symbol, decision, market_data, news_list)
        
        return self.openai_client.analyze_with_prompt(
            prompt=prompt,
            system_message=system_message,
            temperature=SENTIMENT_TEMPERATURE
        )
    
    def _aggregate_validations(
        self,
        original_decision: Dict[str, Any],
        validations: Dict[str, Dict[str, Any]],
        symbol: str,
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate multiple validation results into final decision.
        
        Args:
            original_decision: Original trading decision.
            validations: Dictionary of validation results.
            
        Returns:
            Aggregated final decision.
        """
        # Calculate approval metrics
        approval_rate = self._calculate_approval_rate(validations)
        avg_confidence = self._calculate_average_confidence(validations)
        consensus_factor = self._calculate_consensus_factor(validations)
        
        # Create final decision
        final_decision = original_decision.copy()
        
        # Apply validation rules
        self._apply_validation_rules(final_decision, original_decision, approval_rate, symbol, portfolio)
        
        # Adjust confidence based on consensus
        final_decision['confidence'] = avg_confidence * consensus_factor
        
        # Add validation metadata
        self._add_validation_metadata(final_decision, approval_rate, consensus_factor)
        
        return final_decision
    
    def _calculate_approval_rate(self, validations: Dict[str, Dict[str, Any]]) -> float:
        """Calculate approval rate from validations.
        
        Args:
            validations: Dictionary of validation results.
            
        Returns:
            Approval rate between 0 and 1.
        """
        approvals = sum(1 for v in validations.values() if v.get('approved', False))
        total_validators = len(validations)
        return approvals / total_validators if total_validators > 0 else 0
    
    def _calculate_average_confidence(self, validations: Dict[str, Dict[str, Any]]) -> float:
        """Calculate average confidence from validations.
        
        Args:
            validations: Dictionary of validation results.
            
        Returns:
            Average confidence score.
        """
        confidences = [v.get('confidence', DEFAULT_CONFIDENCE) for v in validations.values()]
        return np.mean(confidences) if confidences else DEFAULT_CONFIDENCE
    
    def _calculate_consensus_factor(self, validations: Dict[str, Dict[str, Any]]) -> float:
        """Calculate consensus factor based on confidence variance.
        
        Args:
            validations: Dictionary of validation results.
            
        Returns:
            Consensus factor between 0 and 1.
        """
        confidences = [v.get('confidence', DEFAULT_CONFIDENCE) for v in validations.values()]
        return 1 - np.std(confidences) if confidences else 1.0
    
    def _apply_validation_rules(
        self,
        final_decision: Dict[str, Any],
        original_decision: Dict[str, Any],
        approval_rate: float,
        symbol: str,
        portfolio: Dict[str, Any]
    ) -> None:
        """Apply validation rules to modify decision.
        
        Args:
            final_decision: Decision to modify (modified in place).
            original_decision: Original decision for reference.
            approval_rate: Calculated approval rate.
            symbol: The symbol being validated.
            portfolio: Current portfolio status.
        """
        if approval_rate < MAJORITY_REJECTION_THRESHOLD:
            # Check if asset is held
            held_assets = set(portfolio.get('assets', {}).keys())
            is_held = symbol in held_assets
            
            if original_decision['action'] in BUY_ACTIONS:
                # If rejected buy action, change to skip (not hold)
                final_decision['action'] = 'skip'
                final_decision['reason'] = f"Multi-AI validation rejected buy: {approval_rate:.0%} approval"
            elif original_decision['action'] in SELL_ACTIONS:
                # For sells, be more cautious about overriding
                if approval_rate < STRONG_REJECTION_THRESHOLD:
                    final_decision['action'] = HOLD_ACTION
                    final_decision['reason'] = f"Multi-AI strongly rejected sell: {approval_rate:.0%} approval"
    
    def _add_validation_metadata(
        self,
        decision: Dict[str, Any],
        approval_rate: float,
        consensus_factor: float
    ) -> None:
        """Add validation metadata to decision.
        
        Args:
            decision: Decision to add metadata to (modified in place).
            approval_rate: Calculated approval rate.
            consensus_factor: Calculated consensus factor.
        """
        decision['multi_ai_approved'] = approval_rate >= MAJORITY_REJECTION_THRESHOLD
        decision['approval_rate'] = approval_rate
        decision['consensus_level'] = consensus_factor
    
    def _check_consensus(self, validations: Dict[str, Dict[str, Any]]) -> bool:
        """Check if validators reached consensus."""
        approvals = [v.get('approved', False) for v in validations.values()]
        return all(approvals) or not any(approvals)  # All agree (either way)
    
    # Prompt creation helper methods
    
    def _create_aggressive_prompt(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any],
        news_list: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for aggressive validator.
        
        Args:
            symbol: Trading symbol.
            decision: Trading decision.
            market_data: Market data.
            news_list: Recent news.
            
        Returns:
            Formatted prompt string.
        """
        news_sentiment = self._analyze_news_sentiment(news_list)
        
        return f"""Review this trading decision for {symbol}:
        
            Decision: {decision.get('action')}
            Reason: {decision.get('reason')}
            Confidence: {decision.get('confidence', DEFAULT_CONFIDENCE)}

            Current Price: {market_data.get('current_price', 0):,.0f} KRW
            24h Change: {market_data.get('price_24h_change', 0):.2f}%
            Volume Ratio: {market_data.get('volume_ratio', 1.0):.2f}

            Recent News Sentiment: {news_sentiment}

            Should this trade be approved from an aggressive growth perspective?
            Respond with JSON: {{"approved": true/false, "confidence": 0.0-1.0, "reason": "explanation"}}"""
    
    def _create_technical_prompt(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> str:
        """Create prompt for technical validator.
        
        Args:
            symbol: Trading symbol.
            decision: Trading decision.
            market_data: Market data with indicators.
            
        Returns:
            Formatted prompt string.
        """
        # Extract technical indicators
        rsi = market_data.get('rsi', {})
        macd = market_data.get('macd', {})
        bollinger = market_data.get('bollinger_bands', {})
        
        return f"""Review this trading decision for {symbol} from a technical perspective:
        
            Decision: {decision.get('action')}
            Confidence: {decision.get('confidence', DEFAULT_CONFIDENCE)}

            Technical Indicators:
            - Price: {market_data.get('current_price', 0):,.0f} KRW
            - RSI(14): {rsi.get('rsi_14', 50)}
            - RSI(30): {rsi.get('rsi_30', 50)}
            - MACD: {macd.get('macd', 0):.2f}
            - Signal: {macd.get('signal', 0):.2f}
            - Histogram: {macd.get('histogram', 0):.2f}
            - Upper BB: {bollinger.get('upper', 0):,.0f}
            - Lower BB: {bollinger.get('lower', 0):,.0f}
            - 24h High: {market_data.get('high_24h', 0):,.0f}
            - 24h Low: {market_data.get('low_24h', 0):,.0f}

            Based purely on technical analysis, should this trade be approved?
            Respond with JSON: {{"approved": true/false, "confidence": 0.0-1.0, "reason": "explanation"}}"""
    
    def _create_sentiment_prompt(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any],
        news_list: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for sentiment validator.
        
        Args:
            symbol: Trading symbol.
            decision: Trading decision.
            market_data: Market data.
            news_list: Recent news.
            
        Returns:
            Formatted prompt string.
        """
        news_summary = self._prepare_news_summary(news_list)
        volume_trend = self._determine_volume_trend(market_data)
        
        return f"""Review this trading decision for {symbol} from a sentiment perspective:
        
            Decision: {decision.get('action')}
            Reason: {decision.get('reason')}

            Market Context:
            - 24h Price Change: {market_data.get('price_24h_change', 0):.2f}%
            - 7d Price Change: {market_data.get('price_change_7d', 0):.2f}%
            - Volume Trend: {volume_trend}

            Recent News:
            {news_summary}

            Based on market sentiment and news, should this trade be approved?
            Respond with JSON: {{"approved": true/false, "confidence": 0.0-1.0, "reason": "explanation"}}"""
    
    def _analyze_news_sentiment(self, news_list: List[Dict[str, Any]]) -> str:
        """Analyze overall news sentiment.
        
        Args:
            news_list: List of news articles.
            
        Returns:
            Sentiment description.
        """
        if any('positive' in str(n).lower() for n in news_list[:3]):
            return "Positive"
        return "Mixed"
    
    def _prepare_news_summary(self, news_list: List[Dict[str, Any]]) -> str:
        """Prepare formatted news summary.
        
        Args:
            news_list: List of news articles.
            
        Returns:
            Formatted news summary.
        """
        return "\n".join([
            f"- {news.get('title', '')}: {news.get('summary', '')[:NEWS_SUMMARY_LENGTH]}..."
            for news in news_list[:NEWS_LIMIT]
        ])
    
    def _determine_volume_trend(self, market_data: Dict[str, Any]) -> str:
        """Determine volume trend description.
        
        Args:
            market_data: Market data dictionary.
            
        Returns:
            Volume trend description.
        """
        volume_ratio = market_data.get('volume_ratio', 1)
        return "High" if volume_ratio > HIGH_VOLUME_THRESHOLD else "Normal"


