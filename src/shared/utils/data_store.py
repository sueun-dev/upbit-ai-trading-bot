"""
Data store for AI trading system - manages trading history, AI analysis results, and system state
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
import threading

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Trading record with AI analysis details."""
    timestamp: str
    symbol: str
    action: str  # buy, buy_more, partial_sell, sell_all, hold
    confidence: float
    reason: str
    price: float
    amount_krw: Optional[float] = None
    quantity: Optional[float] = None
    remaining_quantity: Optional[float] = None  # For buy orders
    buy_trade_ids: Optional[str] = None  # For sell orders (comma-separated IDs)
    avg_buy_price: Optional[float] = None  # For sell orders
    success: Optional[bool] = None
    actual_return: Optional[float] = None

@dataclass
class AIAnalysisResult:
    """Complete AI analysis result."""
    timestamp: str
    news_count: int
    extracted_symbols: List[str]
    market_data_symbols: List[str]
    decisions: Dict[str, Dict[str, Any]]
    analysis_duration: float
    portfolio_value: float
    circuit_breaker_status: Dict[str, Any]

@dataclass 
class TradeAnalysis:
    """AI analysis of trade actions."""
    timestamp: str
    symbol: str
    action: str
    action_korean: str
    analysis: str
    summary: str
    confidence: str
    market_context: Optional[str] = None


class DataStore:
    """Manages all trading data, AI analysis results, and system metrics."""
    
    # USED
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
        
    # USED
    def _initialize_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Trading records table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        reason TEXT NOT NULL,
                        price REAL NOT NULL,
                        amount_krw REAL,
                        quantity REAL,
                        remaining_quantity REAL,
                        buy_trade_ids TEXT,
                        avg_buy_price REAL,
                        success BOOLEAN,
                        actual_return REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # AI analysis results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ai_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        news_count INTEGER NOT NULL,
                        extracted_symbols TEXT NOT NULL,
                        market_data_symbols TEXT NOT NULL,
                        decisions TEXT NOT NULL,
                        analysis_duration REAL NOT NULL,
                        portfolio_value REAL NOT NULL,
                        circuit_breaker_status TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Trade analysis table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        action_korean TEXT NOT NULL,
                        analysis TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        confidence TEXT NOT NULL,
                        market_context TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # System metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_trades INTEGER NOT NULL,
                        successful_trades INTEGER NOT NULL,
                        total_return REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        sharpe_ratio REAL,
                        win_rate REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # News records table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS news_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        title TEXT NOT NULL,
                        summary TEXT,
                        source TEXT NOT NULL,
                        url TEXT,
                        extracted_symbols TEXT,
                        sentiment_score REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index on URL for faster duplicate checks
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_news_url 
                    ON news_records(url)
                """)
                
                # Create index on created_at for time-based queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_news_created_at 
                    ON news_records(created_at)
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def record_trade(self, trade: TradeRecord) -> int:
        """Record a trading decision and execution.
        
        Returns:
            Trade ID if successful, -1 if failed.
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO trades (
                            timestamp, symbol, action, confidence, reason, 
                            price, amount_krw, quantity, remaining_quantity,
                            buy_trade_ids, avg_buy_price, success, actual_return
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade.timestamp, trade.symbol, trade.action, 
                        trade.confidence, trade.reason, trade.price,
                        trade.amount_krw, trade.quantity, trade.remaining_quantity,
                        trade.buy_trade_ids, trade.avg_buy_price, trade.success, 
                        trade.actual_return
                    ))
                    conn.commit()
                    trade_id = cursor.lastrowid
                    
            logger.info(f"Recorded trade #{trade_id}: {trade.symbol} {trade.action}")
            return trade_id
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            return -1
    
    def record_ai_analysis(self, analysis: AIAnalysisResult) -> bool:
        """Record complete AI analysis results."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO ai_analysis (
                            timestamp, news_count, extracted_symbols, 
                            market_data_symbols, decisions, analysis_duration,
                            portfolio_value, circuit_breaker_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        analysis.timestamp,
                        analysis.news_count,
                        json.dumps(analysis.extracted_symbols),
                        json.dumps(analysis.market_data_symbols),
                        json.dumps(analysis.decisions),
                        analysis.analysis_duration,
                        analysis.portfolio_value,
                        json.dumps(analysis.circuit_breaker_status)
                    ))
                    conn.commit()
                    
            logger.info(f"Recorded AI analysis with {len(analysis.decisions)} decisions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record AI analysis: {e}")
            return False
    
    def record_news(self, title: str, summary: str, source: str, url: str = None, 
                   extracted_symbols: List[str] = None, sentiment_score: float = None) -> bool:
        """Record news item with analysis, checking for duplicates."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Check for duplicate by URL (if provided) or title
                    if url:
                        cursor.execute("""
                            SELECT id FROM news_records 
                            WHERE url = ? 
                            AND datetime(created_at) > datetime('now', '-72 hours')
                        """, (url,))
                    else:
                        # If no URL, check by title similarity (exact match for now)
                        cursor.execute("""
                            SELECT id FROM news_records 
                            WHERE title = ? 
                            AND datetime(created_at) > datetime('now', '-72 hours')
                        """, (title,))
                    
                    if cursor.fetchone():
                        logger.debug(f"News already exists: {title[:50]}...")
                        return False
                    
                    # Insert new news
                    cursor.execute("""
                        INSERT INTO news_records (
                            timestamp, title, summary, source, url, 
                            extracted_symbols, sentiment_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        title, summary, source, url,
                        json.dumps(extracted_symbols or []),
                        sentiment_score
                    ))
                    conn.commit()
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to record news: {e}")
            return False
    
    # USED
    def get_recent_trades(self, limit: int = 50, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent trading records."""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (cutoff_time, limit))
                
                columns = [desc[0] for desc in cursor.description]
                trades = []
                
                for row in cursor.fetchall():
                    trade = dict(zip(columns, row))
                    trades.append(trade)
                    
                return trades
                
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []
    
    def get_last_cycle_time(self) -> Optional[str]:
        """Get timestamp of last trading cycle."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp FROM ai_analysis 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                return row[0] if row else None
                
        except Exception as e:
            logger.error(f"Failed to get last cycle time: {e}")
            return None
    
    def record_trade_analysis(self, analysis: TradeAnalysis):
        """Record AI analysis of a trade action."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO trade_analysis 
                        (timestamp, symbol, action, action_korean, analysis, summary, confidence, market_context)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        analysis.timestamp,
                        analysis.symbol,
                        analysis.action,
                        analysis.action_korean,
                        analysis.analysis,
                        analysis.summary,
                        analysis.confidence,
                        analysis.market_context
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to record trade analysis: {e}")
    

    def get_open_buy_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Get open buy trades for a symbol (trades with remaining quantity).
        
        Args:
            symbol: Cryptocurrency symbol.
            
        Returns:
            List of open buy trades ordered by timestamp (FIFO).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, price, quantity, remaining_quantity
                    FROM trades
                    WHERE symbol = ? 
                    AND action IN ('buy', 'buy_more')
                    AND remaining_quantity > 0
                    ORDER BY timestamp ASC
                """, (symbol,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get open buy trades: {e}")
            return []
    
    def calculate_sell_profit(self, symbol: str, sell_price: float, sell_quantity: float) -> Tuple[float, float, List[int]]:
        """Calculate profit for a sell order using FIFO.
        
        Args:
            symbol: Cryptocurrency symbol.
            sell_price: Selling price per unit.
            sell_quantity: Quantity to sell.
            
        Returns:
            Tuple of (profit_percent, avg_buy_price, list_of_used_buy_trade_ids)
        """
        try:
            open_buys = self.get_open_buy_trades(symbol)
            if not open_buys:
                logger.error(f"No open buy trades found for {symbol}")
                return 0.0, 0.0, []
            
            total_cost = 0.0
            total_quantity = 0.0
            used_buy_ids = []
            
            for buy in open_buys:
                if total_quantity >= sell_quantity:
                    break
                    
                available = buy['remaining_quantity']
                use_quantity = min(available, sell_quantity - total_quantity)
                
                total_cost += buy['price'] * use_quantity
                total_quantity += use_quantity
                used_buy_ids.append(buy['id'])
                
                # Update remaining quantity
                new_remaining = available - use_quantity
                self._update_remaining_quantity(buy['id'], new_remaining)
            
            if total_quantity == 0:
                return 0.0, 0.0, []
                
            avg_buy_price = total_cost / total_quantity
            profit_percent = ((sell_price - avg_buy_price) / avg_buy_price) * 100
            
            return profit_percent, avg_buy_price, used_buy_ids
            
        except Exception as e:
            logger.error(f"Failed to calculate sell profit: {e}")
            return 0.0, 0.0, []
    
    def _update_remaining_quantity(self, trade_id: int, new_quantity: float) -> None:
        """Update remaining quantity for a trade."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE trades 
                    SET remaining_quantity = ?
                    WHERE id = ?
                """, (new_quantity, trade_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update remaining quantity: {e}")


# Global instance
data_store = DataStore()