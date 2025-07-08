"""Upbit exchange trader for executing cryptocurrency trades.

This module provides functionality for executing trades on the Upbit exchange,
including market buy/sell orders and various trading strategies.
"""

from typing import Any, Dict, Optional
import logging

import pyupbit

from src.analysis.portfolio.holding_manager import record_purchase
from src.analysis.portfolio.portfolio_manager import get_portfolio_status
from src.infrastructure.config.settings import (
    DEFAULT_BUY_AMOUNT_KRW,
    MIN_ORDER_KRW,
    PARTIAL_SELL_RATIO
)
from src.shared.utils.helpers import get_formatted_datetime

# Constants for trading actions
ACTION_HOLD = "hold"
ACTION_BUY = "buy"
ACTION_BUY_MORE = "buy_more"
ACTION_SELL_ALL = "sell_all"
ACTION_PARTIAL_SELL = "partial_sell"

# Valid trading actions
VALID_ACTIONS = {ACTION_HOLD, ACTION_BUY, ACTION_BUY_MORE, ACTION_SELL_ALL, ACTION_PARTIAL_SELL}
BUY_ACTIONS = {ACTION_BUY, ACTION_BUY_MORE}

# Confidence calculation constants
DEFAULT_CONFIDENCE = 0.5
CONFIDENCE_OFFSET = 0.5
CONFIDENCE_MULTIPLIER = 2
MIN_AVERAGING_CONFIDENCE = 0.0
MAX_AVERAGING_CONFIDENCE = 1.0

# Amount calculation constants
MIN_AMOUNT_THRESHOLD = 0
INVALID_COIN_AMOUNT = 0
AVERAGING_SCALE_BASE = 1

# Log message formats
LOG_BUY_ERROR_NOT_LISTED = "[BUY ERROR] {}: Not listed. Skipping."
LOG_BUY_ERROR_MIN_ORDER = "[BUY ERROR] {}: Amount {:.0f} KRW below {:.0f} KRW. Skipping."
LOG_BUY_SUCCESS = "[BUY] {}: Bought for {:.0f} KRW at {:.0f} KRW."
LOG_SELL_ERROR_NOT_LISTED = "[SELL ERROR] {}: Not listed. Skipping."
LOG_SELL_ERROR_INVALID_AMOUNT = "[SELL ERROR] {}: Invalid amount {:.6f}. Skipping."
LOG_SELL_SUCCESS = "[SELL] {}: Sold {:.6f} coins."
LOG_CIRCUIT_BREAKER_PREVENTED = "Circuit breaker prevented trade: {} {}"
LOG_TRADE_EXECUTION = "Trade {} for {} at {} {}. Reason: {}"
LOG_NO_AVERAGING_ANALYSIS = "No averaging analysis for {} - converting to regular buy"
LOG_HOLDING = "Holding {}."
LOG_UNKNOWN_ACTION = "Unknown action '{}' for {}."
LOG_REGULAR_BUY = "Regular buy {}: {:.0f} KRW."
LOG_NO_POSITION_TO_AVERAGE = "No position to average down {}."
LOG_AVERAGING_AMOUNT_TOO_SMALL = "Averaging amount too small for {}."
LOG_AVERAGING_DOWN = "Averaging down {}: {:.0f} KRW at loss {:.2%}."
LOG_PARTIAL_SELL = "Partial sell {}: {:.6f} coins."
LOG_SELL_ALL = "Sell all {}: {:.6f} coins."

logger = logging.getLogger(__name__)


class UpbitTrader:
    """Upbit exchange trader for executing cryptocurrency trades.
    
    Handles market buy/sell orders and implements various trading strategies
    including averaging down and partial selling.
    """
    def __init__(self, access_key: str, secret_key: str):
        """Initialize Upbit client and load available KRW market symbols.
        
        Args:
            access_key: Upbit API access key.
            secret_key: Upbit API secret key.
        """
        try:
            self.upbit = pyupbit.Upbit(access=access_key, secret=secret_key)
        except Exception as e:
            logger.error(f"Failed to initialize Upbit client: {e}")
            raise
        
        markets = pyupbit.get_tickers(fiat="KRW")
        self.available_symbols = {m.split("-")[1] for m in markets}
        
        logger.info("Upbit trader initialized successfully")
    

    def buy_market_order(self, symbol: str, krw_amount: float) -> bool:
        """Place a market buy order for a given KRW amount.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC").
            krw_amount: Amount in KRW to buy.
            
        Returns:
            True if order was successful, False otherwise.
        """
        # Validate symbol and amount
        if not self._validate_buy_order(symbol, krw_amount):
            return False
        
        try:
            market_code = self._get_market_code(symbol)
            
            # Execute buy order
            self._execute_buy_order(market_code, krw_amount)
            
            # Record purchase and log success
            price = self._get_current_price(market_code)
            self._record_and_log_purchase(symbol, krw_amount, price)
            
            return True
        except Exception as e:
            logger.error(e)
            return False
    
    def _validate_buy_order(self, symbol: str, krw_amount: float) -> bool:
        """Validate buy order parameters.
        
        Args:
            symbol: Cryptocurrency symbol.
            krw_amount: Amount in KRW.
            
        Returns:
            True if valid, False otherwise.
        """
        if symbol not in self.available_symbols:
            logger.error(LOG_BUY_ERROR_NOT_LISTED.format(symbol))
            return False
        
        if krw_amount < MIN_ORDER_KRW:
            logger.error(LOG_BUY_ERROR_MIN_ORDER.format(symbol, krw_amount, MIN_ORDER_KRW))
            return False
        
        return True
    
    def _get_market_code(self, symbol: str) -> str:
        """Get market code for a symbol.
        
        Args:
            symbol: Cryptocurrency symbol.
            
        Returns:
            Market code (e.g., "KRW-BTC").
        """
        return f"{"KRW-"}{symbol}"
    
    def _execute_buy_order(self, market_code: str, krw_amount: float) -> None:
        """Execute the buy order on Upbit.
        
        Args:
            market_code: Market code (e.g., "KRW-BTC").
            krw_amount: Amount in KRW.
        """
        self.upbit.buy_market_order(market_code, krw_amount)
    
    def _get_current_price(self, market_code: str) -> float:
        """Get current price for a market.
        
        Args:
            market_code: Market code (e.g., "KRW-BTC").
            
        Returns:
            Current price in KRW.
        """
        return float(pyupbit.get_current_price(market_code))
    
    def _record_and_log_purchase(self, symbol: str, krw_amount: float, price: float) -> None:
        """Record purchase and log success message.
        
        Args:
            symbol: Cryptocurrency symbol.
            krw_amount: Amount in KRW.
            price: Purchase price.
        """
        record_purchase(symbol, price)
        logger.info(LOG_BUY_SUCCESS.format(symbol, krw_amount, price))

    def sell_market_order(self, symbol: str, coin_amount: float) -> bool:
        """Place a market sell order for a given coin amount.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC").
            coin_amount: Amount of coins to sell.
            
        Returns:
            True if order was successful, False otherwise.
        """
        # Validate symbol and amount
        if not self._validate_sell_order(symbol, coin_amount):
            return False
        
        try:
            market_code = self._get_market_code(symbol)
            
            # Execute sell order
            self._execute_sell_order(market_code, coin_amount)
            
            # Log success
            logger.info(LOG_SELL_SUCCESS.format(symbol, coin_amount))
            
            return True
        except Exception as e:
            logger.error(e)
            return False
    
    def _validate_sell_order(self, symbol: str, coin_amount: float) -> bool:
        """Validate sell order parameters.
        
        Args:
            symbol: Cryptocurrency symbol.
            coin_amount: Amount of coins.
            
        Returns:
            True if valid, False otherwise.
        """
        if symbol not in self.available_symbols:
            logger.error(LOG_SELL_ERROR_NOT_LISTED.format(symbol))
            return False
        
        if coin_amount <= INVALID_COIN_AMOUNT:
            logger.error(LOG_SELL_ERROR_INVALID_AMOUNT.format(symbol, coin_amount))
            return False
        
        return True
    
    def _execute_sell_order(self, market_code: str, coin_amount: float) -> None:
        """Execute the sell order on Upbit.
        
        Args:
            market_code: Market code (e.g., "KRW-BTC").
            coin_amount: Amount of coins to sell.
        """
        self.upbit.sell_market_order(market_code, coin_amount)

    # USED
    def execute_trade(
        self,
        symbol: str,
        action: str,
        reason: str,
        decision_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Dispatch trading actions: hold, buy, buy_more, sell_all, partial_sell.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC").
            action: Trading action to execute.
            reason: Reason for the trade.
            decision_data: Additional decision data including confidence and analysis.
            
        Returns:
            True if trade was successful, False otherwise.
        """
        symbol = symbol.upper()
        
        # Log trade intention
        current_date, current_time = get_formatted_datetime()
        logger.info(LOG_TRADE_EXECUTION.format(action, symbol, current_date, current_time, reason))
        
        
        # Handle buy_more validation
        action = self._validate_buy_more_action(symbol, action, decision_data)
        
        # Execute the trade
        trade_result = self._dispatch_trade_action(symbol, action, decision_data)
        
        
        return trade_result
    
    
    def _validate_buy_more_action(
        self,
        symbol: str,
        action: str,
        decision_data: Optional[Dict[str, Any]]
    ) -> str:
        """Validate buy_more action and convert to regular buy if needed.
        
        Args:
            symbol: Cryptocurrency symbol.
            action: Trading action.
            decision_data: Decision data containing averaging analysis.
            
        Returns:
            Validated action (may be converted from buy_more to buy).
        """
        if action == ACTION_BUY_MORE:
            if not self._has_averaging_analysis(decision_data):
                logger.warning(LOG_NO_AVERAGING_ANALYSIS.format(symbol))
                return ACTION_BUY
        return action
    
    def _has_averaging_analysis(self, decision_data: Optional[Dict[str, Any]]) -> bool:
        """Check if decision data contains averaging analysis.
        
        Args:
            decision_data: Decision data dictionary.
            
        Returns:
            True if averaging analysis exists, False otherwise.
        """
        return bool(decision_data and decision_data.get("averaging_analysis"))
    
    def _dispatch_trade_action(
        self,
        symbol: str,
        action: str,
        decision_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Dispatch trade to appropriate handler based on action.
        
        Args:
            symbol: Cryptocurrency symbol.
            action: Trading action.
            decision_data: Decision data for the trade.
            
        Returns:
            True if trade was successful, False otherwise.
        """
        if action == ACTION_HOLD:
            return self._handle_hold_action(symbol)
        
        if action == ACTION_BUY:
            return self._execute_regular_buy(symbol)
        
        if action == ACTION_BUY_MORE:
            return self._buy_average_down(symbol, decision_data["averaging_analysis"])
        
        if action == ACTION_SELL_ALL:
            return self._execute_sell_all(symbol)
        
        if action == ACTION_PARTIAL_SELL:
            sell_percentage = decision_data.get('sell_percentage') if decision_data else None
            return self._execute_partial_sell(symbol, sell_percentage)
        
        # Unknown action
        logger.error(LOG_UNKNOWN_ACTION.format(action, symbol))
        return False
    
    def _handle_hold_action(self, symbol: str) -> bool:
        """Handle hold action (no trade).
        
        Args:
            symbol: Cryptocurrency symbol.
            
        Returns:
            Always True for hold actions.
        """
        logger.info(LOG_HOLDING.format(symbol))
        return True
    
    def _get_trade_amount(self, action: str) -> float:
        """Get trade amount based on action.
        
        Args:
            action: Trading action.
            
        Returns:
            Trade amount in KRW.
        """
        if action in BUY_ACTIONS:
            return DEFAULT_BUY_AMOUNT_KRW
        return 0

    def _execute_regular_buy(self, symbol: str) -> bool:
        """Execute a fixed KRW amount buy.
        
        Args:
            symbol: Cryptocurrency symbol.
            
        Returns:
            True if buy was successful, False otherwise.
        """
        logger.info(LOG_REGULAR_BUY.format(symbol, DEFAULT_BUY_AMOUNT_KRW))
        return self.buy_market_order(symbol, DEFAULT_BUY_AMOUNT_KRW)

    def _buy_average_down(self, symbol: str, avg_analysis: Dict[str, Any]) -> bool:
        """Execute averaging down buy based on analysis.
        
        Args:
            symbol: Cryptocurrency symbol.
            avg_analysis: Averaging analysis data.
            
        Returns:
            True if buy was successful, False otherwise.
        """
        # Check if position exists
        if not self._has_position(symbol):
            logger.error(LOG_NO_POSITION_TO_AVERAGE.format(symbol))
            return False
        
        # Calculate averaging amount
        amount = self._calculate_averaging_amount(avg_analysis)
        if not self._validate_averaging_amount(symbol, amount):
            return False
        
        # Check if we have sufficient KRW balance
        try:
            from src.analysis.portfolio.portfolio_manager import PortfolioManager
            pm = PortfolioManager(self.client)
            available_krw = pm.get_krw_balance()
            
            if available_krw < amount:
                logger.error(f"Insufficient KRW balance for averaging down. Required: {amount:,.0f} KRW, Available: {available_krw:,.0f} KRW")
                return False
        except Exception as e:
            logger.error(f"Failed to check KRW balance: {e}")
            # Don't proceed with averaging down if we can't verify balance
            return False
        
        # Log and execute averaging down
        current_loss = avg_analysis.get('current_loss', 0)
        logger.info(LOG_AVERAGING_DOWN.format(symbol, amount, current_loss))
        
        return self.buy_market_order(symbol, amount)
    
    def _has_position(self, symbol: str) -> bool:
        """Check if we have a position in the symbol.
        
        Args:
            symbol: Cryptocurrency symbol.
            
        Returns:
            True if position exists, False otherwise.
        """
        portfolio = self.get_portfolio_status()
        return symbol in portfolio["assets"]
    
    def _calculate_averaging_amount(self, avg_analysis: Dict[str, Any]) -> float:
        """Calculate amount for averaging down based on confidence.
        
        Args:
            avg_analysis: Averaging analysis containing confidence.
            
        Returns:
            Amount in KRW for averaging down.
        """
        confidence = avg_analysis.get("confidence", DEFAULT_CONFIDENCE)
        confidence = max(MIN_AVERAGING_CONFIDENCE, min(confidence, MAX_AVERAGING_CONFIDENCE))
        
        base_amount = DEFAULT_BUY_AMOUNT_KRW
        return base_amount * (AVERAGING_SCALE_BASE + confidence)
    
    def _validate_averaging_amount(self, symbol: str, amount: float) -> bool:
        """Validate if averaging amount is sufficient.
        
        Args:
            symbol: Cryptocurrency symbol.
            amount: Amount in KRW.
            
        Returns:
            True if amount is valid, False otherwise.
        """
        if amount <= MIN_AMOUNT_THRESHOLD:
            logger.error(LOG_AVERAGING_AMOUNT_TOO_SMALL.format(symbol))
            return False
        return True

    def _execute_partial_sell(self, symbol: str, sell_percentage: Optional[float] = None) -> bool:
        """Sell a percentage of current holdings.
        
        Args:
            symbol: Cryptocurrency symbol.
            sell_percentage: Optional percentage to sell (0.0-1.0). 
                           Uses PARTIAL_SELL_RATIO if not provided.
            
        Returns:
            True if sell was successful, False otherwise.
        """
        balance = self._get_symbol_balance(symbol)
        if balance is None:
            return False
        
        # Use provided percentage or default
        percentage = sell_percentage if sell_percentage is not None else PARTIAL_SELL_RATIO
        # Ensure percentage is within reasonable bounds
        percentage = max(0.05, min(0.5, percentage))  # Between 5% and 50%
        
        amount = balance * percentage
        logger.info(f"💰 Executing partial sell for {symbol}: {amount:.6f} ({percentage:.1%} of holdings)")
        
        return self.sell_market_order(symbol, amount)

    def _execute_sell_all(self, symbol: str) -> bool:
        """Sell entire position.
        
        Args:
            symbol: Cryptocurrency symbol.
            
        Returns:
            True if sell was successful, False otherwise.
        """
        balance = self._get_symbol_balance(symbol)
        if balance is None:
            return False
        
        logger.info(LOG_SELL_ALL.format(symbol, balance))
        return self.sell_market_order(symbol, balance)
    
    def _get_symbol_balance(self, symbol: str) -> Optional[float]:
        """Get balance for a specific symbol.
        
        Args:
            symbol: Cryptocurrency symbol.
            
        Returns:
            Balance amount or None if not found.
        """
        portfolio = self.get_portfolio_status()
        
        if symbol not in portfolio["assets"]:
            logger.error(f"No {symbol} position found in portfolio")
            return None
        
        return portfolio["assets"][symbol]["balance"]
    
    def get_portfolio_status(self, api_key: str = None) -> Dict[str, Any]:
        """Get portfolio status through portfolio manager.
        
        Args:
            api_key: Optional OpenAI API key for AI analysis.
            
        Returns:
            Portfolio status dictionary.
        """
        return get_portfolio_status(self.upbit, api_key=api_key)