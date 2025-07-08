"""Circuit breaker for preventing consecutive losses."""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker to prevent cascading losses."""
    

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        max_daily_loss_percent: float = 10.0,
        max_trades_per_hour: int = 10,
        cooldown_hours: int = 2,
        data_file: str = "circuit_breaker_data.json"
    ):
        """Initialize circuit breaker.
        
        Args:
            max_consecutive_losses: Maximum allowed consecutive losses
            max_daily_loss_percent: Maximum daily loss percentage
            max_trades_per_hour: Maximum trades allowed per hour
            cooldown_hours: Hours to wait after circuit break
            data_file: File to store circuit breaker data
        """
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_trades_per_hour = max_trades_per_hour
        self.cooldown_hours = cooldown_hours
        self.data_file = os.path.join(os.path.dirname(__file__), data_file)
        
        self.trade_history: List[Dict] = []
        self.circuit_open = False
        self.circuit_open_time: Optional[datetime] = None
        self.consecutive_losses = 0
        
        self._load_data()
    
    # USED
    def _load_data(self) -> None:
        """Load circuit breaker data from file."""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            self.trade_history = data.get('trade_history', [])
            self.circuit_open = data.get('circuit_open', False)
            self.circuit_open_time = datetime.fromisoformat(data['circuit_open_time']) if data.get('circuit_open_time') else None
            self.consecutive_losses = data.get('consecutive_losses', 0)
            logger.info(f"Loaded circuit breaker data: {len(self.trade_history)} trades, circuit_open={self.circuit_open}")
        except:
            self._initialize_empty_data()
    

    def _initialize_empty_data(self) -> None:
        """Initialize with empty data."""
        self.trade_history = []
        self.circuit_open = False
        self.circuit_open_time = None
        self.consecutive_losses = 0
    
    def _save_data(self) -> None:
        """Save circuit breaker data to file."""
        try:
            data = {
                'trade_history': self.trade_history[-100:],  # Keep last 100
                'circuit_open': self.circuit_open,
                'circuit_open_time': (
                    self.circuit_open_time.isoformat()
                    if self.circuit_open_time
                    else None
                ),
                'consecutive_losses': self.consecutive_losses
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save circuit breaker data: %s", e)
    
    # USED
    def can_trade(self) -> bool:
        """Check if trading is allowed.
        
        Returns:
            bool: True if trading is allowed, False otherwise
        """
        # Check if circuit is open and cooldown has passed
        if self.circuit_open and self.circuit_open_time:
            cooldown_end = self.circuit_open_time + timedelta(hours=self.cooldown_hours)
            if datetime.now() >= cooldown_end:
                logger.info("Circuit breaker cooldown period ended, resetting")
                self.reset_circuit()
            else:
                remaining = (cooldown_end - datetime.now()).total_seconds() / 60
                logger.warning("Circuit breaker is OPEN. Trading disabled for %.1f more minutes",
                             remaining)
                return False
        
        # Check hourly trade limit
        if self._check_hourly_trade_limit():
            logger.warning("Hourly trade limit reached (%d trades)", self.max_trades_per_hour)
            return False
        
        # Check daily loss limit
        if self._check_daily_loss_limit():
            logger.warning("Daily loss limit reached (%.1f%%)", self.max_daily_loss_percent)
            self._open_circuit("Daily loss limit exceeded")
            return False
        
        return True
    
    def record_trade(
        self,
        symbol: str,
        action: str,
        profit_loss_percent: float,
        amount_krw: float
    ) -> None:
        """Record a completed trade.
        
        Args:
            symbol: Trading symbol
            action: Trade action (buy, sell, etc.)
            profit_loss_percent: Profit/loss percentage
            amount_krw: Trade amount in KRW
        """
        # Determine trade result
        if profit_loss_percent > 0.5:
            result = "profit"
            self.consecutive_losses = 0  # Reset consecutive losses
        elif profit_loss_percent < -0.5:
            result = "loss"
            self.consecutive_losses += 1
            
            # Check consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                self._open_circuit(f"Max consecutive losses ({self.max_consecutive_losses}) reached")
        else:
            result = "neutral"
        
        # Create trade record
        record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'result': result,
            'profit_loss_percent': profit_loss_percent,
            'amount_krw': amount_krw
        }
        
        self.trade_history.append(record)
        self._save_data()
        
        logger.info("Recorded trade: %s %s %.2f%% (%s), consecutive_losses=%d",
                   symbol, action, profit_loss_percent, result, self.consecutive_losses)
    
    # USED
    def _check_hourly_trade_limit(self) -> bool:
        """Check if hourly trade limit is exceeded.
        
        Returns:
            bool: True if limit exceeded, False otherwise
        """
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_trades = [
            t for t in self.trade_history
            if datetime.fromisoformat(t['timestamp']) > one_hour_ago
        ]
        return len(recent_trades) >= self.max_trades_per_hour
    
    # USED
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded.
        
        Returns:
            bool: True if limit exceeded, False otherwise
        """
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = [
            t for t in self.trade_history
            if datetime.fromisoformat(t['timestamp']) >= today_start
        ]
        
        if not today_trades:
            return False
        
        # Calculate total daily profit/loss
        total_loss_percent = sum(
            t['profit_loss_percent'] for t in today_trades
            if t['profit_loss_percent'] < 0
        )
        
        return abs(total_loss_percent) >= self.max_daily_loss_percent
    
    def _open_circuit(self, reason: str) -> None:
        """Open the circuit breaker.
        
        Args:
            reason: Reason for opening circuit
        """
        self.circuit_open = True
        self.circuit_open_time = datetime.now()
        self._save_data()
        
        logger.error("CIRCUIT BREAKER OPENED: %s. Trading disabled for %d hours",
                    reason, self.cooldown_hours)
    
    def reset_circuit(self) -> None:
        """Reset the circuit breaker."""
        self.circuit_open = False
        self.circuit_open_time = None
        self.consecutive_losses = 0
        self._save_data()
        
        logger.info("Circuit breaker reset. Trading enabled.")
    
    def get_status(self) -> Dict[str, any]:
        """Get current circuit breaker status.
        
        Returns:
            Dict containing status information
        """
        status = {
            'circuit_open': self.circuit_open,
            'consecutive_losses': self.consecutive_losses,
            'trades_last_hour': len([
                t for t in self.trade_history
                if datetime.fromisoformat(t['timestamp']) > datetime.now() - timedelta(hours=1)
            ]),
            'daily_loss_percent': 0.0
        }
        
        # Calculate daily loss
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = [
            t for t in self.trade_history
            if datetime.fromisoformat(t['timestamp']) >= today_start
        ]
        
        if today_trades:
            status['daily_loss_percent'] = abs(sum(
                t['profit_loss_percent'] for t in today_trades
                if t['profit_loss_percent'] < 0
            ))
        
        if self.circuit_open and self.circuit_open_time:
            cooldown_end = self.circuit_open_time + timedelta(hours=self.cooldown_hours)
            status['cooldown_remaining_minutes'] = max(
                0,
                (cooldown_end - datetime.now()).total_seconds() / 60
            )
        
        return status