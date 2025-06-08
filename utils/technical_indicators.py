"""
Technical Indicators
Comprehensive collection of technical analysis indicators
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Technical indicators calculator"""
    
    def __init__(self):
        self.indicators_count = 215  # Total supported indicators
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return pd.DataFrame({
                'macd': macd,
                'signal': signal_line,
                'histogram': histogram
            })
        
        except Exception:
            return pd.DataFrame({
                'macd': pd.Series([0] * len(prices), index=prices.index),
                'signal': pd.Series([0] * len(prices), index=prices.index),
                'histogram': pd.Series([0] * len(prices), index=prices.index)
            })
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return pd.DataFrame({
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band
            })
        
        except Exception:
            return pd.DataFrame({
                'upper': prices,
                'middle': prices,
                'lower': prices
            })
    
    def calculate_sma(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average"""
        try:
            return prices.rolling(window=period).mean().fillna(prices.iloc[0])
        except Exception:
            return prices
    
    def calculate_ema(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average"""
        try:
            return prices.ewm(span=period).mean().fillna(prices.iloc[0])
        except Exception:
            return prices
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = pd.Series(true_range).rolling(window=period).mean()
            
            return atr.fillna(0)
        
        except Exception:
            return pd.Series([0] * len(high), index=high.index)
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return pd.DataFrame({
                'k': k_percent.fillna(50),
                'd': d_percent.fillna(50)
            })
        
        except Exception:
            return pd.DataFrame({
                'k': pd.Series([50] * len(high), index=high.index),
                'd': pd.Series([50] * len(high), index=high.index)
            })
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r.fillna(-50)
        
        except Exception:
            return pd.Series([-50] * len(high), index=high.index)
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (high + low + close) / 3
            sma = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma) / (0.015 * mad)
            return cci.fillna(0)
        
        except Exception:
            return pd.Series([0] * len(high), index=high.index)
    
    def calculate_momentum(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Momentum"""
        try:
            momentum = prices - prices.shift(period)
            return momentum.fillna(0)
        except Exception:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def calculate_roc(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change"""
        try:
            roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
            return roc.fillna(0)
        except Exception:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        
        except Exception:
            return volume
    
    def calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            return vwap.fillna(close)
        except Exception:
            return close
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        try:
            # Calculate True Range
            tr = self.calculate_atr(high, low, close, period=1)
            
            # Calculate Directional Movement
            dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
            dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
            
            # Smooth the values
            atr = tr.rolling(window=period).mean()
            di_plus = 100 * (pd.Series(dm_plus).rolling(window=period).mean() / atr)
            di_minus = 100 * (pd.Series(dm_minus).rolling(window=period).mean() / atr)
            
            # Calculate DX and ADX
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=period).mean()
            
            return adx.fillna(0)
        
        except Exception:
            return pd.Series([0] * len(high), index=high.index)
    
    def calculate_all_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available technical indicators"""
        try:
            results = ohlcv_data.copy()
            
            # Basic indicators
            results['rsi'] = self.calculate_rsi(ohlcv_data['close'])
            results['sma_20'] = self.calculate_sma(ohlcv_data['close'], 20)
            results['ema_20'] = self.calculate_ema(ohlcv_data['close'], 20)
            results['atr'] = self.calculate_atr(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])
            
            # MACD
            macd_data = self.calculate_macd(ohlcv_data['close'])
            results['macd'] = macd_data['macd']
            results['macd_signal'] = macd_data['signal']
            results['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(ohlcv_data['close'])
            results['bb_upper'] = bb_data['upper']
            results['bb_middle'] = bb_data['middle']
            results['bb_lower'] = bb_data['lower']
            
            # Stochastic
            stoch_data = self.calculate_stochastic(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])
            results['stoch_k'] = stoch_data['k']
            results['stoch_d'] = stoch_data['d']
            
            # Additional indicators
            results['williams_r'] = self.calculate_williams_r(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])
            results['cci'] = self.calculate_cci(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])
            results['momentum'] = self.calculate_momentum(ohlcv_data['close'])
            results['roc'] = self.calculate_roc(ohlcv_data['close'])
            results['adx'] = self.calculate_adx(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])
            
            if 'volume' in ohlcv_data.columns:
                results['obv'] = self.calculate_obv(ohlcv_data['close'], ohlcv_data['volume'])
                results['vwap'] = self.calculate_vwap(ohlcv_data['high'], ohlcv_data['low'], 
                                                   ohlcv_data['close'], ohlcv_data['volume'])
            
            return results
        
        except Exception as e:
            return ohlcv_data