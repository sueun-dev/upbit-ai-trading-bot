"""Tests for EnhancedMarketDataCollector."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, UTC
from src.data.collectors.enhanced_market_data import (
    EnhancedMarketData,
    EnhancedMarketDataCollector
)


class TestEnhancedMarketDataCollector(unittest.TestCase):
    """Test cases for EnhancedMarketDataCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = EnhancedMarketDataCollector()
        
        # Sample OHLCV data for testing
        self.sample_1h_data = pd.DataFrame({
            'open': [50000] * 24,
            'high': [51000] * 24,
            'low': [49000] * 24,
            'close': [50500] * 24,
            'volume': [100] * 24
        }, index=pd.date_range(end='2025-01-01 23:00:00', periods=24, freq='h'))
        
        self.sample_1d_data = pd.DataFrame({
            'open': [45000 + i*100 for i in range(35)],
            'high': [46000 + i*100 for i in range(35)],
            'low': [44000 + i*100 for i in range(35)],
            'close': [45500 + i*100 for i in range(35)],
            'volume': [1000 + i*10 for i in range(35)]
        }, index=pd.date_range(end='2025-01-01', periods=35, freq='D'))
        
        self.sample_200d_data = pd.DataFrame({
            'open': [40000 + i*50 for i in range(200)],
            'high': [41000 + i*50 for i in range(200)],
            'low': [39000 + i*50 for i in range(200)],
            'close': [40500 + i*50 for i in range(200)],
            'volume': [5000 + i*5 for i in range(200)]
        }, index=pd.date_range(end='2025-01-01', periods=200, freq='D'))
    
    @patch('src.data.collectors.enhanced_market_data.pyupbit.get_current_price')
    @patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv')
    def test_get_enhanced_market_data_success(self, mock_get_ohlcv, mock_get_current_price):
        """Test successful enhanced market data collection."""
        # Setup mocks
        mock_get_current_price.return_value = 50000.0
        
        def ohlcv_side_effect(market_code, interval, count):
            if interval == "minute60":
                return self.sample_1h_data
            elif interval == "day" and count == 30:
                return self.sample_1d_data
            elif interval == "day" and count == 200:
                return self.sample_200d_data
            return None
        
        mock_get_ohlcv.side_effect = ohlcv_side_effect
        
        # Test
        result = self.collector.get_enhanced_market_data("BTC")
        
        # Assertions
        self.assertIsInstance(result, EnhancedMarketData)
        self.assertEqual(result.symbol, "BTC")
        self.assertEqual(result.current_price, 50000.0)
        self.assertIsInstance(result.timestamp, str)
        
        # Verify all required fields are present
        self.assertIsNotNone(result.price_1h_change)
        self.assertIsNotNone(result.volume_24h)
        self.assertIsNotNone(result.rsi_14)
        self.assertIsNotNone(result.macd_line)
        self.assertIsNotNone(result.bb_upper)
        self.assertIsNotNone(result.support_1)
        self.assertIsNotNone(result.volatility_7d)
        self.assertIsNotNone(result.correlation_btc_7d)
    
    @patch('src.data.collectors.enhanced_market_data.pyupbit.get_current_price')
    def test_get_enhanced_market_data_failure(self, mock_get_current_price):
        """Test enhanced market data collection failure."""
        mock_get_current_price.side_effect = Exception("API Error")
        
        result = self.collector.get_enhanced_market_data("BTC")
        
        self.assertIsNone(result)
    
    def test_get_price_data_comprehensive(self):
        """Test comprehensive price data calculation."""
        with patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv') as mock_get_ohlcv:
            mock_get_ohlcv.side_effect = lambda market_code, interval, count: (
                self.sample_1h_data if interval == "minute60" else self.sample_1d_data
            )
            
            result = self.collector._get_price_data("KRW-BTC", 50000.0)
            
            # Check all price fields
            expected_fields = [
                'price_1h_change', 'price_24h_change', 'price_7d_change', 'price_30d_change',
                'price_1h_high', 'price_1h_low', 'price_24h_high', 'price_24h_low',
                'price_7d_high', 'price_7d_low', 'price_30d_high', 'price_30d_low'
            ]
            
            for field in expected_fields:
                self.assertIn(field, result)
                self.assertIsInstance(result[field], (int, float, np.integer, np.floating))
    
    def test_get_price_data_insufficient_data(self):
        """Test price data calculation with insufficient data."""
        with patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv') as mock_get_ohlcv:
            # Return very limited data
            limited_data = pd.DataFrame({
                'open': [50000],
                'high': [51000],
                'low': [49000],
                'close': [50500],
                'volume': [100]
            })
            mock_get_ohlcv.return_value = limited_data
            
            result = self.collector._get_price_data("KRW-BTC", 50000.0)
            
            # Should return default values
            self.assertEqual(result['price_1h_change'], 0.0)
            self.assertEqual(result['price_24h_change'], 0.0)
    
    def test_get_volume_data(self):
        """Test volume data calculation."""
        with patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv') as mock_get_ohlcv:
            mock_get_ohlcv.side_effect = lambda market_code, interval, count: (
                self.sample_1h_data if interval == "minute60" else self.sample_1d_data
            )
            
            result = self.collector._get_volume_data("KRW-BTC")
            
            # Check volume fields
            self.assertIn('volume_1h', result)
            self.assertIn('volume_24h', result)
            self.assertIn('volume_7d_avg', result)
            self.assertIn('volume_30d_avg', result)
            self.assertIn('volume_ratio_1h_24h', result)
            self.assertIn('volume_ratio_24h_7d', result)
            self.assertIn('vwap_24h', result)
            self.assertIn('vwap_7d', result)
            
            # Check calculations
            self.assertGreater(result['volume_24h'], 0)
            self.assertGreater(result['vwap_24h'], 0)
    
    def test_calculate_vwap(self):
        """Test VWAP calculation."""
        # Create sample data
        df = pd.DataFrame({
            'high': [51000, 52000, 53000],
            'low': [49000, 50000, 51000],
            'close': [50000, 51000, 52000],
            'volume': [100, 200, 300]
        })
        
        vwap = self.collector._calculate_vwap(df)
        
        # VWAP should be weighted average of typical prices
        expected_vwap = ((50000*100 + 51000*200 + 52000*300) / 600)
        self.assertAlmostEqual(vwap, expected_vwap, delta=1)
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        # Create price series with known pattern
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 
                           111, 110, 112, 114, 113, 115, 117, 116, 118, 120])
        
        rsi = self.collector._calculate_rsi(prices, 14)
        
        # RSI should be between 0 and 100
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        
        # With mostly upward movement, RSI should be > 50
        self.assertGreater(rsi, 50)
    
    def test_calculate_macd(self):
        """Test MACD calculation."""
        # Create price series
        prices = pd.Series([100 + i for i in range(30)])
        
        macd_line, macd_signal, macd_histogram = self.collector._calculate_macd(prices)
        
        # All values should be numeric
        self.assertIsInstance(macd_line, float)
        self.assertIsInstance(macd_signal, float)
        self.assertIsInstance(macd_histogram, float)
        
        # Histogram should be difference between line and signal
        self.assertAlmostEqual(macd_histogram, macd_line - macd_signal, places=4)
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        prices = pd.Series([100 + np.random.normal(0, 2) for _ in range(30)])
        
        upper, middle, lower, width = self.collector._calculate_bollinger_bands(prices)
        
        # Bands should be in correct order
        self.assertGreater(upper, middle)
        self.assertGreater(middle, lower)
        self.assertGreater(width, 0)
        
        # Middle band should be close to mean
        self.assertAlmostEqual(middle, prices.iloc[-20:].mean(), delta=0.1)
    
    def test_calculate_stochastic(self):
        """Test Stochastic oscillator calculation."""
        highs = pd.Series([100 + i for i in range(20)])
        lows = pd.Series([90 + i for i in range(20)])
        closes = pd.Series([95 + i for i in range(20)])
        
        k, d = self.collector._calculate_stochastic(highs, lows, closes)
        
        # Values should be between 0 and 100
        self.assertGreaterEqual(k, 0)
        self.assertLessEqual(k, 100)
        self.assertGreaterEqual(d, 0)
        self.assertLessEqual(d, 100)
    
    def test_get_technical_indicators_comprehensive(self):
        """Test comprehensive technical indicators calculation."""
        with patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv') as mock_get_ohlcv:
            mock_get_ohlcv.return_value = self.sample_200d_data
            
            result = self.collector._get_technical_indicators("KRW-BTC")
            
            # Check all technical indicator fields
            expected_fields = [
                'rsi_14', 'rsi_30', 'macd_line', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'stoch_k', 'stoch_d', 'adx', 'adx_di_plus', 'adx_di_minus',
                'atr_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
                'ema_cross_signal', 'ichimoku_tenkan', 'ichimoku_kijun',
                'ichimoku_senkou_a', 'ichimoku_senkou_b', 'ichimoku_chikou',
                'obv', 'obv_ema', 'mfi_14', 'cci_20', 'williams_r', 'roc_10',
                'pivot_point', 'pivot_r1', 'pivot_r2', 'pivot_s1', 'pivot_s2'
            ]
            
            for field in expected_fields:
                self.assertIn(field, result)
                if field != 'ema_cross_signal':
                    self.assertIsInstance(result[field], (int, float))
                else:
                    self.assertIn(result[field], ['bullish', 'bearish', 'neutral'])
    
    def test_get_technical_indicators_insufficient_data(self):
        """Test technical indicators with insufficient data."""
        with patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv') as mock_get_ohlcv:
            # Return only 10 rows
            limited_data = self.sample_200d_data.iloc[:10]
            mock_get_ohlcv.return_value = limited_data
            
            result = self.collector._get_technical_indicators("KRW-BTC")
            
            # Should return default values
            self.assertEqual(result['rsi_14'], 0.0)
            self.assertEqual(result['ema_cross_signal'], 'neutral')
    
    def test_get_support_resistance_levels(self):
        """Test support and resistance calculation."""
        with patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv') as mock_get_ohlcv:
            mock_get_ohlcv.return_value = self.sample_1d_data
            
            result = self.collector._get_support_resistance_levels("KRW-BTC")
            
            # Check fields
            self.assertIn('support_1', result)
            self.assertIn('support_2', result)
            self.assertIn('resistance_1', result)
            self.assertIn('resistance_2', result)
            
            # Resistance should be higher than support
            self.assertGreater(result['resistance_1'], result['support_1'])
            self.assertGreater(result['resistance_2'], result['support_2'])
    
    def test_get_volatility_metrics(self):
        """Test volatility calculation."""
        with patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv') as mock_get_ohlcv:
            # Create data with some volatility
            volatile_data = pd.DataFrame({
                'open': [45000 + i*100 + np.random.normal(0, 500) for i in range(35)],
                'high': [46000 + i*100 + np.random.normal(0, 500) for i in range(35)],
                'low': [44000 + i*100 + np.random.normal(0, 500) for i in range(35)],
                'close': [45500 + i*100 + np.random.normal(0, 500) for i in range(35)],
                'volume': [1000 + i*10 for i in range(35)]
            }, index=pd.date_range(end='2025-01-01', periods=35, freq='D'))
            
            mock_get_ohlcv.return_value = volatile_data
            
            result = self.collector._get_volatility_metrics("KRW-BTC")
            
            # Check fields
            self.assertIn('volatility_1d', result)
            self.assertIn('volatility_7d', result)
            self.assertIn('volatility_30d', result)
            
            # All should be non-negative (or 0 for single day std)
            self.assertGreaterEqual(result['volatility_1d'], 0)
            self.assertGreaterEqual(result['volatility_7d'], 0)
            self.assertGreaterEqual(result['volatility_30d'], 0)
    
    def test_get_correlation_data_btc(self):
        """Test correlation calculation for BTC."""
        result = self.collector._get_correlation_data("BTC")
        
        # BTC correlation with itself should be 1.0
        self.assertEqual(result['correlation_btc_7d'], 1.0)
        self.assertEqual(result['correlation_btc_30d'], 1.0)
    
    def test_get_correlation_data_other_symbol(self):
        """Test correlation calculation for non-BTC symbol."""
        with patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv') as mock_get_ohlcv:
            # Create correlated data
            eth_data = self.sample_1d_data.copy()
            eth_data['close'] = eth_data['close'] * 0.1  # Scale down
            
            def ohlcv_side_effect(market_code, interval, count):
                if "ETH" in market_code:
                    return eth_data
                else:  # BTC
                    return self.sample_1d_data
            
            mock_get_ohlcv.side_effect = ohlcv_side_effect
            
            result = self.collector._get_correlation_data("ETH")
            
            # Check fields
            self.assertIn('correlation_btc_7d', result)
            self.assertIn('correlation_btc_30d', result)
            
            # Correlation should be between -1 and 1
            self.assertGreaterEqual(result['correlation_btc_7d'], -1)
            self.assertLessEqual(result['correlation_btc_7d'], 1)
    
    def test_calculate_adx(self):
        """Test ADX calculation."""
        highs = pd.Series([100 + i + np.random.normal(0, 1) for i in range(30)])
        lows = pd.Series([90 + i + np.random.normal(0, 1) for i in range(30)])
        closes = pd.Series([95 + i + np.random.normal(0, 1) for i in range(30)])
        
        adx, di_plus, di_minus = self.collector._calculate_adx(highs, lows, closes)
        
        # All values should be non-negative
        self.assertGreaterEqual(adx, 0)
        self.assertGreaterEqual(di_plus, 0)
        self.assertGreaterEqual(di_minus, 0)
    
    def test_calculate_atr(self):
        """Test ATR calculation."""
        highs = pd.Series([100 + i for i in range(20)])
        lows = pd.Series([90 + i for i in range(20)])
        closes = pd.Series([95 + i for i in range(20)])
        
        atr = self.collector._calculate_atr(highs, lows, closes)
        
        # ATR should be positive
        self.assertGreater(atr, 0)
    
    def test_get_ema_cross_signal(self):
        """Test EMA crossover signal determination."""
        # Bullish pattern
        signal = self.collector._get_ema_cross_signal(100, 95, 90)
        self.assertEqual(signal, "bullish")
        
        # Bearish pattern
        signal = self.collector._get_ema_cross_signal(90, 95, 100)
        self.assertEqual(signal, "bearish")
        
        # Neutral pattern
        signal = self.collector._get_ema_cross_signal(95, 100, 90)
        self.assertEqual(signal, "neutral")
    
    def test_calculate_ichimoku(self):
        """Test Ichimoku Cloud calculation."""
        highs = pd.Series([100 + i for i in range(60)])
        lows = pd.Series([90 + i for i in range(60)])
        closes = pd.Series([95 + i for i in range(60)])
        
        result = self.collector._calculate_ichimoku(highs, lows, closes)
        
        # Check all Ichimoku components
        expected_fields = [
            'ichimoku_tenkan', 'ichimoku_kijun', 
            'ichimoku_senkou_a', 'ichimoku_senkou_b', 'ichimoku_chikou'
        ]
        
        for field in expected_fields:
            self.assertIn(field, result)
            self.assertIsInstance(result[field], (int, float))
    
    def test_calculate_obv(self):
        """Test OBV calculation."""
        closes = pd.Series([100, 102, 101, 103, 105])
        volumes = pd.Series([1000, 1200, 800, 1500, 2000])
        
        obv, obv_ema = self.collector._calculate_obv(closes, volumes)
        
        # OBV should reflect volume direction
        self.assertNotEqual(obv, 0)
        self.assertIsInstance(obv_ema, float)
    
    def test_calculate_mfi(self):
        """Test MFI calculation."""
        highs = pd.Series([100 + i for i in range(20)])
        lows = pd.Series([90 + i for i in range(20)])
        closes = pd.Series([95 + i for i in range(20)])
        volumes = pd.Series([1000 + i*10 for i in range(20)])
        
        mfi = self.collector._calculate_mfi(highs, lows, closes, volumes)
        
        # MFI should be between 0 and 100
        self.assertGreaterEqual(mfi, 0)
        self.assertLessEqual(mfi, 100)
    
    def test_calculate_cci(self):
        """Test CCI calculation."""
        highs = pd.Series([100 + i for i in range(25)])
        lows = pd.Series([90 + i for i in range(25)])
        closes = pd.Series([95 + i for i in range(25)])
        
        cci = self.collector._calculate_cci(highs, lows, closes)
        
        # CCI is typically between -200 and +200
        self.assertIsInstance(cci, float)
    
    def test_calculate_williams_r(self):
        """Test Williams %R calculation."""
        highs = pd.Series([100 + i for i in range(20)])
        lows = pd.Series([90 + i for i in range(20)])
        closes = pd.Series([95 + i for i in range(20)])
        
        williams_r = self.collector._calculate_williams_r(highs, lows, closes)
        
        # Williams %R should be between -100 and 0
        self.assertGreaterEqual(williams_r, -100)
        self.assertLessEqual(williams_r, 0)
    
    def test_calculate_roc(self):
        """Test ROC calculation."""
        closes = pd.Series([100 + i*2 for i in range(15)])
        
        roc = self.collector._calculate_roc(closes)
        
        # With increasing prices, ROC should be positive
        self.assertGreater(roc, 0)
    
    def test_calculate_pivot_points(self):
        """Test pivot points calculation."""
        high, low, close = 110.0, 90.0, 100.0
        
        result = self.collector._calculate_pivot_points(high, low, close)
        
        # Check all pivot fields
        self.assertIn('pivot_point', result)
        self.assertIn('pivot_r1', result)
        self.assertIn('pivot_r2', result)
        self.assertIn('pivot_s1', result)
        self.assertIn('pivot_s2', result)
        
        # Check order: R2 > R1 > Pivot > S1 > S2
        self.assertGreater(result['pivot_r2'], result['pivot_r1'])
        self.assertGreater(result['pivot_r1'], result['pivot_point'])
        self.assertGreater(result['pivot_point'], result['pivot_s1'])
        self.assertGreater(result['pivot_s1'], result['pivot_s2'])
    
    def test_edge_case_empty_dataframe(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()
        
        vwap = self.collector._calculate_vwap(empty_df)
        self.assertEqual(vwap, 0.0)
    
    def test_edge_case_nan_values(self):
        """Test handling of NaN values in calculations."""
        prices = pd.Series([100, np.nan, 102, 103, np.nan, 105])
        
        # RSI should handle NaN values gracefully
        rsi = self.collector._calculate_rsi(prices, 14)
        self.assertIsInstance(rsi, float)
        self.assertFalse(np.isnan(rsi))
    
    @patch('src.data.collectors.enhanced_market_data.pyupbit.get_ohlcv')
    def test_api_error_handling(self, mock_get_ohlcv):
        """Test handling of API errors."""
        mock_get_ohlcv.side_effect = Exception("API Error")
        
        # Price data should return defaults on error
        result = self.collector._get_price_data("KRW-BTC", 50000.0)
        self.assertEqual(result['price_1h_change'], 0.0)
        
        # Volume data should return defaults on error
        result = self.collector._get_volume_data("KRW-BTC")
        self.assertEqual(result['volume_1h'], 0.0)
        
        # Technical indicators should return defaults on error
        result = self.collector._get_technical_indicators("KRW-BTC")
        self.assertEqual(result['rsi_14'], 0.0)
    
    def test_symbol_case_handling(self):
        """Test handling of different symbol cases."""
        with patch('src.data.collectors.enhanced_market_data.pyupbit.get_current_price') as mock_price:
            mock_price.return_value = 50000.0
            
            # Test lowercase symbol
            result = self.collector.get_enhanced_market_data("btc")
            # Should be converted to uppercase
            if result:
                self.assertEqual(result.symbol, "BTC")


if __name__ == '__main__':
    unittest.main()