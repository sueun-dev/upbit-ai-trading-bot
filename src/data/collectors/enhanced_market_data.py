"""Enhanced market data collection with comprehensive quantitative analysis."""

import logging
import pandas as pd
import numpy as np
import pyupbit
from datetime import datetime, UTC
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMarketData:
    """Comprehensive market data structure."""
    # Basic Info
    symbol: str
    current_price: float
    timestamp: str
    
    # Price Analysis
    price_1h_change: float
    price_24h_change: float
    price_7d_change: float
    price_30d_change: float
    price_1h_high: float
    price_1h_low: float
    price_24h_high: float
    price_24h_low: float
    price_7d_high: float
    price_7d_low: float
    price_30d_high: float
    price_30d_low: float
    
    # Volume Analysis
    volume_1h: float
    volume_24h: float
    volume_7d_avg: float
    volume_30d_avg: float
    volume_ratio_1h_24h: float
    volume_ratio_24h_7d: float
    vwap_24h: float
    vwap_7d: float
    
    # Technical Indicators
    rsi_14: float
    rsi_30: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    stoch_k: float
    stoch_d: float
    
    # Additional Indicators
    adx: float
    adx_di_plus: float
    adx_di_minus: float
    atr_14: float
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: float
    ema_cross_signal: str
    ichimoku_tenkan: float
    ichimoku_kijun: float
    ichimoku_senkou_a: float
    ichimoku_senkou_b: float
    ichimoku_chikou: float
    obv: float
    obv_ema: float
    mfi_14: float
    cci_20: float
    williams_r: float
    roc_10: float
    pivot_point: float
    pivot_r1: float
    pivot_r2: float
    pivot_s1: float
    pivot_s2: float
    
    # Support/Resistance
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    
    # Volatility
    volatility_1d: float
    volatility_7d: float
    volatility_30d: float
    
    # Correlation (vs BTC)
    correlation_btc_7d: float
    correlation_btc_30d: float


class EnhancedMarketDataCollector:
    """Collects comprehensive quantitative market data for AI analysis."""
    
    def __init__(self):
        pass  # All data comes from Upbit API
        
    # USED
    def get_enhanced_market_data(self, symbol: str) -> Optional[EnhancedMarketData]:
        """Get comprehensive market data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            
        Returns:
            EnhancedMarketData object or None if failed
        """
        symbol = symbol.upper()
        market_code = f"KRW-{symbol}"
        
        try:
            logger.info("Collecting enhanced market data for %s", symbol)
            
            # Get current price
            current_price = pyupbit.get_current_price(market_code)
            
            # Get price and volume data
            price_data = self._get_price_data(market_code, current_price)
            volume_data = self._get_volume_data(market_code)
            
            # Get technical indicators
            technical_data = self._get_technical_indicators(market_code)
            
            # Get support/resistance levels
            support_resistance = self._get_support_resistance_levels(market_code)
            
            # Get volatility metrics
            volatility_data = self._get_volatility_metrics(market_code)
            
            # Get correlation data
            correlation_data = self._get_correlation_data(symbol)
            
            # Combine all data
            enhanced_data = EnhancedMarketData(
                # Basic Info
                symbol=symbol,
                current_price=current_price,
                timestamp=datetime.now(UTC).isoformat(),
                
                # Price Analysis
                **price_data,
                
                # Volume Analysis
                **volume_data,
                
                # Technical Indicators
                **technical_data,
                
                # Support/Resistance
                **support_resistance,
                
                # Volatility
                **volatility_data,
                
                # Correlation
                **correlation_data
            )
            
            logger.info("Successfully collected enhanced data for %s", symbol)
            return enhanced_data
            
        except Exception as e:
            logger.error("Failed to collect enhanced market data for %s: %s", symbol, e)
            return None
    
    # USED
    def _get_price_data(self, market_code: str, current_price: float) -> Dict[str, float]:
        """Get comprehensive price analysis data."""
        try:
            # Get different timeframe data
            df_1h = pyupbit.get_ohlcv(market_code, interval="minute60", count=24)  # 24 hours
            df_1d = pyupbit.get_ohlcv(market_code, interval="day", count=30)      # 30 days
            
            if df_1h is None or df_1d is None or len(df_1h) < 2 or len(df_1d) < 2:
                raise ValueError("Insufficient price data")
            
            # 1 hour analysis
            if len(df_1h) >= 2:
                price_1h_ago = df_1h.iloc[-2]['close']
                price_1h_change = ((current_price - price_1h_ago) / price_1h_ago) * 100
                price_1h_high = df_1h.iloc[-1]['high']
                price_1h_low = df_1h.iloc[-1]['low']
            else:
                price_1h_change = price_1h_high = price_1h_low = 0.0
            
            # 24 hour analysis
            if len(df_1d) >= 2:
                price_24h_ago = df_1d.iloc[-2]['close']
                price_24h_change = ((current_price - price_24h_ago) / price_24h_ago) * 100
                price_24h_high = df_1d.iloc[-1]['high']
                price_24h_low = df_1d.iloc[-1]['low']
            else:
                price_24h_change = price_24h_high = price_24h_low = 0.0
            
            # 7 day analysis
            if len(df_1d) >= 8:
                price_7d_ago = df_1d.iloc[-8]['close']
                price_7d_change = ((current_price - price_7d_ago) / price_7d_ago) * 100
                price_7d_high = df_1d.iloc[-7:]['high'].max()
                price_7d_low = df_1d.iloc[-7:]['low'].min()
            else:
                price_7d_change = price_7d_high = price_7d_low = 0.0
            
            # 30 day analysis
            if len(df_1d) >= 31:
                price_30d_ago = df_1d.iloc[-31]['close']
                price_30d_change = ((current_price - price_30d_ago) / price_30d_ago) * 100
                price_30d_high = df_1d.iloc[-30:]['high'].max()
                price_30d_low = df_1d.iloc[-30:]['low'].min()
            else:
                price_30d_change = price_30d_high = price_30d_low = 0.0
            
            return {
                'price_1h_change': round(price_1h_change, 2),
                'price_24h_change': round(price_24h_change, 2),
                'price_7d_change': round(price_7d_change, 2),
                'price_30d_change': round(price_30d_change, 2),
                'price_1h_high': round(price_1h_high, 2),
                'price_1h_low': round(price_1h_low, 2),
                'price_24h_high': round(price_24h_high, 2),
                'price_24h_low': round(price_24h_low, 2),
                'price_7d_high': round(price_7d_high, 2),
                'price_7d_low': round(price_7d_low, 2),
                'price_30d_high': round(price_30d_high, 2),
                'price_30d_low': round(price_30d_low, 2),
            }
            
        except Exception as e:
            logger.error("Failed to get price data: %s", e)
            return {k: 0.0 for k in [
                'price_1h_change', 'price_24h_change', 'price_7d_change', 'price_30d_change',
                'price_1h_high', 'price_1h_low', 'price_24h_high', 'price_24h_low',
                'price_7d_high', 'price_7d_low', 'price_30d_high', 'price_30d_low'
            ]}
    
    def _get_volume_data(self, market_code: str) -> Dict[str, float]:
        """Get comprehensive volume analysis data."""
        try:
            df_1h = pyupbit.get_ohlcv(market_code, interval="minute60", count=24)
            df_1d = pyupbit.get_ohlcv(market_code, interval="day", count=30)
            
            if df_1h is None or df_1d is None or len(df_1h) == 0 or len(df_1d) == 0:
                raise ValueError("No volume data available")
            
            # Current hour volume
            volume_1h = df_1h.iloc[-1]['volume'] if len(df_1h) > 0 else 0.0
            
            # 24h volume
            volume_24h = df_1d.iloc[-1]['volume'] if len(df_1d) > 0 else 0.0
            
            # 7d and 30d average volumes
            volume_7d_avg = df_1d.iloc[-7:]['volume'].mean() if len(df_1d) >= 7 else volume_24h
            volume_30d_avg = df_1d.iloc[-30:]['volume'].mean() if len(df_1d) >= 30 else volume_24h
            
            # Volume ratios
            volume_ratio_1h_24h = volume_1h / (volume_24h / 24) if volume_24h > 0 else 0
            volume_ratio_24h_7d = volume_24h / volume_7d_avg if volume_7d_avg > 0 else 0
            
            # VWAP calculation
            vwap_24h = self._calculate_vwap(df_1d.iloc[-1:])
            vwap_7d = self._calculate_vwap(df_1d.iloc[-7:]) if len(df_1d) >= 7 else vwap_24h
            
            return {
                'volume_1h': round(volume_1h, 2),
                'volume_24h': round(volume_24h, 2),
                'volume_7d_avg': round(volume_7d_avg, 2),
                'volume_30d_avg': round(volume_30d_avg, 2),
                'volume_ratio_1h_24h': round(volume_ratio_1h_24h, 2),
                'volume_ratio_24h_7d': round(volume_ratio_24h_7d, 2),
                'vwap_24h': round(vwap_24h, 2),
                'vwap_7d': round(vwap_7d, 2),
            }
            
        except Exception as e:
            logger.error("Failed to get volume data: %s", e)
            return {k: 0.0 for k in [
                'volume_1h', 'volume_24h', 'volume_7d_avg', 'volume_30d_avg',
                'volume_ratio_1h_24h', 'volume_ratio_24h_7d', 'vwap_24h', 'vwap_7d'
            ]}
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price."""
        if df.empty:
            return 0.0
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return float(vwap)
    
    def _get_technical_indicators(self, market_code: str) -> Dict[str, Any]:
        """Calculate technical indicators."""
        try:
            df = pyupbit.get_ohlcv(market_code, interval="day", count=200)  # Need more data for EMA 200
            if df is None or len(df) < 30:
                logger.warning(f"Insufficient data for technical indicators: {market_code}, got {len(df) if df is not None else 0} rows")
                raise ValueError("Insufficient data for technical indicators")
            
            closes = df['close']
            highs = df['high']
            lows = df['low']
            volumes = df['volume']
            
            # RSI calculation
            rsi_14 = self._calculate_rsi(closes, 14)
            rsi_30 = self._calculate_rsi(closes, 30)
            
            # MACD calculation
            macd_line, macd_signal, macd_histogram = self._calculate_macd(closes)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bollinger_bands(closes)
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes)
            
            # ADX calculation
            adx, di_plus, di_minus = self._calculate_adx(highs, lows, closes)
            
            # ATR calculation
            atr_14 = self._calculate_atr(highs, lows, closes, 14)
            
            # EMA calculations with safety checks
            ema_9 = float(closes.ewm(span=9).mean().iloc[-1]) if len(closes) >= 9 else 0.0
            ema_21 = float(closes.ewm(span=21).mean().iloc[-1]) if len(closes) >= 21 else 0.0
            ema_50 = float(closes.ewm(span=50).mean().iloc[-1]) if len(closes) >= 50 else 0.0
            ema_200 = float(closes.ewm(span=200).mean().iloc[-1]) if len(closes) >= 200 else 0.0
            
            # EMA crossover signal
            ema_cross_signal = self._get_ema_cross_signal(ema_9, ema_21, ema_50)
            
            # Ichimoku Cloud
            ichimoku = self._calculate_ichimoku(highs, lows, closes)
            
            # OBV (On Balance Volume)
            obv, obv_ema = self._calculate_obv(closes, volumes)
            
            # MFI (Money Flow Index)
            mfi_14 = self._calculate_mfi(highs, lows, closes, volumes, 14)
            
            # CCI (Commodity Channel Index)
            cci_20 = self._calculate_cci(highs, lows, closes, 20)
            
            # Williams %R
            williams_r = self._calculate_williams_r(highs, lows, closes)
            
            # ROC (Rate of Change)
            roc_10 = self._calculate_roc(closes, 10)
            
            # Pivot Points with safety check
            if len(highs) > 0 and len(lows) > 0 and len(closes) > 0:
                pivot_points = self._calculate_pivot_points(highs.iloc[-1], lows.iloc[-1], closes.iloc[-1])
            else:
                pivot_points = {
                    'pivot_point': 0.0, 'pivot_r1': 0.0, 'pivot_r2': 0.0,
                    'pivot_s1': 0.0, 'pivot_s2': 0.0
                }
            
            # Clean all values before returning
            def safe_round(value, decimals=2):
                """Safely round a value, replacing NaN/inf with 0."""
                import math
                if isinstance(value, (int, float)) and (math.isnan(value) or math.isinf(value)):
                    return 0.0
                return round(float(value), decimals) if value is not None else 0.0
            
            result = {
                'rsi_14': safe_round(rsi_14),
                'rsi_30': safe_round(rsi_30),
                'macd_line': safe_round(macd_line, 4),
                'macd_signal': safe_round(macd_signal, 4),
                'macd_histogram': safe_round(macd_histogram, 4),
                'bb_upper': safe_round(bb_upper),
                'bb_middle': safe_round(bb_middle),
                'bb_lower': safe_round(bb_lower),
                'bb_width': safe_round(bb_width),
                'stoch_k': safe_round(stoch_k),
                'stoch_d': safe_round(stoch_d),
                'adx': safe_round(adx),
                'adx_di_plus': safe_round(di_plus),
                'adx_di_minus': safe_round(di_minus),
                'atr_14': safe_round(atr_14),
                'ema_9': safe_round(ema_9),
                'ema_21': safe_round(ema_21),
                'ema_50': safe_round(ema_50),
                'ema_200': safe_round(ema_200),
                'ema_cross_signal': ema_cross_signal or 'neutral',
            }
            
            # Clean ichimoku values
            for key, value in ichimoku.items():
                result[key] = safe_round(value)
            
            # Add other cleaned values
            result.update({
                'obv': safe_round(obv, 0),
                'obv_ema': safe_round(obv_ema, 0),
                'mfi_14': safe_round(mfi_14),
                'cci_20': safe_round(cci_20),
                'williams_r': safe_round(williams_r),
                'roc_10': safe_round(roc_10),
            })
            
            # Clean pivot points
            for key, value in pivot_points.items():
                result[key] = safe_round(value)
            
            return result
            
        except Exception as e:
            logger.error("Failed to calculate technical indicators: %s", e)
            default_values = {k: 0.0 for k in [
                'rsi_14', 'rsi_30', 'macd_line', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'stoch_k', 'stoch_d',
                'adx', 'adx_di_plus', 'adx_di_minus', 'atr_14', 'ema_9', 'ema_21',
                'ema_50', 'ema_200', 'ichimoku_tenkan', 'ichimoku_kijun', 
                'ichimoku_senkou_a', 'ichimoku_senkou_b', 'ichimoku_chikou',
                'obv', 'obv_ema', 'mfi_14', 'cci_20', 'williams_r', 'roc_10',
                'pivot_point', 'pivot_r1', 'pivot_r2', 'pivot_s1', 'pivot_s2'
            ]}
            default_values['ema_cross_signal'] = 'neutral'
            return default_values
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Ensure no NaN values
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        return rsi_val
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal
        
        # Ensure no NaN values
        macd_line_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
        macd_signal_val = float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0
        macd_histogram_val = float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0.0
        
        return (macd_line_val, macd_signal_val, macd_histogram_val)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[float, float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            current_price = float(prices.iloc[-1])
            return current_price, current_price, current_price, 0.0
        
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        middle_band = rolling_mean
        
        bb_width = ((upper_band - lower_band) / middle_band) * 100
        
        # Ensure no NaN values
        upper_val = float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else 0.0
        middle_val = float(middle_band.iloc[-1]) if not pd.isna(middle_band.iloc[-1]) else 0.0
        lower_val = float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else 0.0
        width_val = float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else 0.0
        
        return (upper_val, middle_val, lower_val, width_val)
    
    def _calculate_stochastic(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic oscillator."""
        if len(closes) < k_period:
            return 50.0, 50.0
        
        highest_high = highs.rolling(window=k_period).max()
        lowest_low = lows.rolling(window=k_period).min()
        
        k_percent = 100 * ((closes - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        # Ensure no NaN values
        k_val = float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else 50.0
        d_val = float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else 50.0
        
        return k_val, d_val
    
    
    def _get_support_resistance_levels(self, market_code: str) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        try:
            df = pyupbit.get_ohlcv(market_code, interval="day", count=30)
            if df is None or len(df) < 10:
                raise ValueError("Insufficient data for support/resistance")
            
            highs = df['high']
            lows = df['low']
            
            # Simple support/resistance based on recent highs and lows
            resistance_1 = float(highs.iloc[-10:].max())
            resistance_2 = float(highs.iloc[-20:].max())
            support_1 = float(lows.iloc[-10:].min())
            support_2 = float(lows.iloc[-20:].min())
            
            return {
                'support_1': support_1,
                'support_2': support_2,
                'resistance_1': resistance_1,
                'resistance_2': resistance_2,
            }
            
        except Exception as e:
            logger.error("Failed to calculate support/resistance: %s", e)
            return {
                'support_1': 0.0,
                'support_2': 0.0,
                'resistance_1': 0.0,
                'resistance_2': 0.0,
            }
    
    def _get_volatility_metrics(self, market_code: str) -> Dict[str, float]:
        """Calculate volatility metrics."""
        try:
            df = pyupbit.get_ohlcv(market_code, interval="day", count=30)
            if df is None or len(df) < 2:
                raise ValueError("Insufficient data for volatility")
            
            returns = df['close'].pct_change().dropna()
            
            # For 1d volatility, use absolute return instead of std on single value
            volatility_1d = float(abs(returns.iloc[-1]) * 100) if len(returns) >= 1 else 0.0
            volatility_7d = float(returns.iloc[-7:].std() * 100) if len(returns) >= 7 else 0.0
            volatility_30d = float(returns.std() * 100) if len(returns) >= 30 else 0.0
            
            # Handle NaN values
            if pd.isna(volatility_1d):
                volatility_1d = 0.0
            if pd.isna(volatility_7d):
                volatility_7d = 0.0
            if pd.isna(volatility_30d):
                volatility_30d = 0.0
            
            return {
                'volatility_1d': round(volatility_1d, 2),
                'volatility_7d': round(volatility_7d, 2),
                'volatility_30d': round(volatility_30d, 2),
            }
            
        except Exception as e:
            logger.error("Failed to calculate volatility: %s", e)
            return {
                'volatility_1d': 0.0,
                'volatility_7d': 0.0,
                'volatility_30d': 0.0,
            }
    
    def _get_correlation_data(self, symbol: str) -> Dict[str, float]:
        """Calculate correlation with Bitcoin."""
        try:
            if symbol == 'BTC':
                return {
                    'correlation_btc_7d': 1.0,
                    'correlation_btc_30d': 1.0,
                }
            
            # Get both symbol and BTC data
            symbol_df = pyupbit.get_ohlcv(f"KRW-{symbol}", interval="day", count=30)
            btc_df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
            
            if symbol_df is None or btc_df is None or len(symbol_df) < 7:
                raise ValueError("Insufficient data for correlation")
            
            symbol_returns = symbol_df['close'].pct_change().dropna()
            btc_returns = btc_df['close'].pct_change().dropna()
            
            # Align the data
            min_len = min(len(symbol_returns), len(btc_returns))
            symbol_returns = symbol_returns.iloc[-min_len:]
            btc_returns = btc_returns.iloc[-min_len:]
            
            correlation_7d = float(symbol_returns.iloc[-7:].corr(btc_returns.iloc[-7:])) if min_len >= 7 else 0.0
            correlation_30d = float(symbol_returns.corr(btc_returns)) if min_len >= 30 else 0.0
            
            return {
                'correlation_btc_7d': round(correlation_7d, 3),
                'correlation_btc_30d': round(correlation_30d, 3),
            }
            
        except Exception as e:
            logger.error("Failed to calculate correlation: %s", e)
            return {
                'correlation_btc_7d': 0.0,
                'correlation_btc_30d': 0.0,
            }
    
    def _calculate_adx(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> Tuple[float, float, float]:
        """Calculate Average Directional Index and DI+/DI-."""
        if len(closes) < period + 1:
            return 0.0, 0.0, 0.0
        
        # Calculate True Range
        high_low = highs - lows
        high_close = abs(highs - closes.shift(1))
        low_close = abs(lows - closes.shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate directional movements
        up_move = highs - highs.shift(1)
        down_move = lows.shift(1) - lows
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return float(adx.iloc[-1]), float(pos_di.iloc[-1]), float(neg_di.iloc[-1])
    
    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(closes) < period:
            return 0.0
        
        high_low = highs - lows
        high_close = abs(highs - closes.shift(1))
        low_close = abs(lows - closes.shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return float(atr.iloc[-1])
    
    def _get_ema_cross_signal(self, ema_9: float, ema_21: float, ema_50: float) -> str:
        """Determine EMA crossover signal."""
        if ema_9 > ema_21 > ema_50:
            return "bullish"
        elif ema_9 < ema_21 < ema_50:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_ichimoku(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> Dict[str, float]:
        """Calculate Ichimoku Cloud indicators."""
        if len(closes) < 52:
            return {
                'ichimoku_tenkan': 0.0,
                'ichimoku_kijun': 0.0,
                'ichimoku_senkou_a': 0.0,
                'ichimoku_senkou_b': 0.0,
                'ichimoku_chikou': 0.0
            }
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = highs.rolling(window=9).max()
        nine_period_low = lows.rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = highs.rolling(window=26).max()
        period26_low = lows.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = highs.rolling(window=52).max()
        period52_low = lows.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close plotted 26 days in the past
        chikou_span = closes.shift(-26)
        
        return {
            'ichimoku_tenkan': round(float(tenkan_sen.iloc[-1]), 2),
            'ichimoku_kijun': round(float(kijun_sen.iloc[-1]), 2),
            'ichimoku_senkou_a': round(float(senkou_span_a.iloc[-1]) if not pd.isna(senkou_span_a.iloc[-1]) else 0.0, 2),
            'ichimoku_senkou_b': round(float(senkou_span_b.iloc[-1]) if not pd.isna(senkou_span_b.iloc[-1]) else 0.0, 2),
            'ichimoku_chikou': round(float(chikou_span.iloc[-27]) if len(chikou_span) > 26 else 0.0, 2)
        }
    
    def _calculate_obv(self, closes: pd.Series, volumes: pd.Series) -> Tuple[float, float]:
        """Calculate On Balance Volume and its EMA."""
        if len(closes) < 2:
            return 0.0, 0.0
        
        price_diff = closes.diff()
        volume_direction = volumes.copy()
        volume_direction[price_diff < 0] *= -1
        volume_direction[price_diff == 0] = 0
        
        obv = volume_direction.cumsum()
        obv_ema = obv.ewm(span=20).mean()
        
        return float(obv.iloc[-1]), float(obv_ema.iloc[-1])
    
    def _calculate_mfi(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, volumes: pd.Series, period: int = 14) -> float:
        """Calculate Money Flow Index."""
        if len(closes) < period + 1:
            return 50.0
        
        typical_price = (highs + lows + closes) / 3
        money_flow = typical_price * volumes
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return float(mfi.iloc[-1])
    
    def _calculate_cci(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 20) -> float:
        """Calculate Commodity Channel Index."""
        if len(closes) < period:
            return 0.0
        
        typical_price = (highs + lows + closes) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        
        cci = (typical_price - sma) / (0.015 * mad)
        
        return float(cci.iloc[-1])
    
    def _calculate_williams_r(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        """Calculate Williams %R."""
        if len(closes) < period:
            return -50.0
        
        highest_high = highs.rolling(window=period).max()
        lowest_low = lows.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - closes) / (highest_high - lowest_low))
        
        return float(williams_r.iloc[-1])
    
    def _calculate_roc(self, closes: pd.Series, period: int = 10) -> float:
        """Calculate Rate of Change."""
        if len(closes) < period + 1:
            return 0.0
        
        roc = ((closes - closes.shift(period)) / closes.shift(period)) * 100
        
        return float(roc.iloc[-1])
    
    def _calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate pivot points for support and resistance."""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        
        return {
            'pivot_point': round(pivot, 2),
            'pivot_r1': round(r1, 2),
            'pivot_r2': round(r2, 2),
            'pivot_s1': round(s1, 2),
            'pivot_s2': round(s2, 2)
        }


# Global instance
enhanced_collector = EnhancedMarketDataCollector()