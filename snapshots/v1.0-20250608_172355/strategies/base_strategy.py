import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import pandas_ta as ta

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.signals_history = []
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal based on market data"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                return False
            
            if len(data) < 50:  # Minimum data points required
                return False
            
            if data[required_columns].isnull().any().any():
                return False
            
            return True
        except:
            return False
    
    def add_signal_to_history(self, signal: Dict[str, Any]):
        """Add signal to history"""
        signal['timestamp'] = datetime.now()
        signal['strategy'] = self.name
        self.signals_history.append(signal)
        
        # Keep only last 1000 signals
        if len(self.signals_history) > 1000:
            self.signals_history = self.signals_history[-1000:]
    
    def get_recent_signals(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals"""
        return self.signals_history[-count:]
    
    def calculate_performance_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate strategy performance metrics"""
        try:
            if not returns:
                return {}
            
            returns_array = np.array(returns)
            
            metrics = {
                'total_return': np.sum(returns_array),
                'avg_return': np.mean(returns_array),
                'volatility': np.std(returns_array),
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
            
            # Sharpe ratio (assuming risk-free rate = 0)
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = metrics['avg_return'] / metrics['volatility']
            
            # Max drawdown
            cumulative_returns = np.cumsum(returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            metrics['max_drawdown'] = np.max(drawdown)
            
            # Win rate
            winning_trades = np.sum(returns_array > 0)
            total_trades = len(returns_array)
            metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = np.sum(returns_array[returns_array > 0])
            gross_loss = abs(np.sum(returns_array[returns_array < 0]))
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            self.performance_metrics = metrics
            return metrics
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {}

class TechnicalStrategy(BaseStrategy):
    """Base class for technical analysis strategies"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.indicators = {}
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI using pandas_ta"""
        try:
            # Clean data before RSI calculation
            clean_data = data.copy()
            if clean_data.index.duplicated().any():
                clean_data = clean_data[~clean_data.index.duplicated(keep='last')]
            
            rsi_result = ta.rsi(clean_data['close'], length=period)
            
            # Ensure proper indexing
            if rsi_result is not None and len(rsi_result) > 0:
                return rsi_result
            else:
                return pd.Series([50.0] * len(data), index=data.index)
        except Exception as e:
            print(f"RSI calculation error: {e}")
            return pd.Series([50.0] * len(data), index=data.index)
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD using pandas_ta"""
        try:
            macd_data = ta.macd(data['close'], fast=fast, slow=slow, signal=signal)
            if macd_data is not None and not macd_data.empty:
                return macd_data
            else:
                # Return neutral MACD if calculation fails
                return pd.DataFrame({
                    f'MACD_{fast}_{slow}_{signal}': [0] * len(data),
                    f'MACDh_{fast}_{slow}_{signal}': [0] * len(data),
                    f'MACDs_{fast}_{slow}_{signal}': [0] * len(data)
                })
        except:
            return pd.DataFrame({
                f'MACD_{12}_{26}_{9}': [0] * len(data),
                f'MACDh_{12}_{26}_{9}': [0] * len(data),
                f'MACDs_{12}_{26}_{9}': [0] * len(data)
            })
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands using pandas_ta"""
        try:
            bb = ta.bbands(data['close'], length=period, std=std)
            if bb is not None and not bb.empty:
                return bb
            else:
                # Return price as all bands if calculation fails
                close_price = data['close']
                return pd.DataFrame({
                    f'BBL_{period}_{std}': close_price,
                    f'BBM_{period}_{std}': close_price,
                    f'BBU_{period}_{std}': close_price,
                    f'BBB_{period}_{std}': [0.5] * len(data),
                    f'BBP_{period}_{std}': [0.5] * len(data)
                })
        except:
            close_price = data['close']
            return pd.DataFrame({
                f'BBL_{20}_{2}': close_price,
                f'BBM_{20}_{2}': close_price,
                f'BBU_{20}_{2}': close_price,
                f'BBB_{20}_{2}': [0.5] * len(data),
                f'BBP_{20}_{2}': [0.5] * len(data)
            })
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        try:
            ma_data = pd.DataFrame()
            
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                if len(data) >= period:
                    ma_data[f'SMA_{period}'] = ta.sma(data['close'], length=period)
                else:
                    ma_data[f'SMA_{period}'] = data['close']
            
            # Exponential Moving Averages
            for period in [5, 10, 20, 50]:
                if len(data) >= period:
                    ma_data[f'EMA_{period}'] = ta.ema(data['close'], length=period)
                else:
                    ma_data[f'EMA_{period}'] = data['close']
            
            return ma_data
            
        except:
            # Return basic moving averages if calculation fails
            return pd.DataFrame({
                'SMA_20': data['close'].rolling(20, min_periods=1).mean(),
                'SMA_50': data['close'].rolling(50, min_periods=1).mean(),
                'EMA_20': data['close'].ewm(span=20).mean()
            })
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        try:
            vol_data = pd.DataFrame()
            
            # Volume moving average
            vol_data['Volume_SMA_20'] = ta.sma(data['volume'], length=20)
            
            # On Balance Volume
            vol_data['OBV'] = ta.obv(data['close'], data['volume'])
            
            # Volume Rate of Change
            vol_data['Volume_ROC'] = ta.roc(data['volume'], length=10)
            
            return vol_data.fillna(0)
            
        except:
            return pd.DataFrame({
                'Volume_SMA_20': data['volume'].rolling(20, min_periods=1).mean(),
                'OBV': [0] * len(data),
                'Volume_ROC': [0] * len(data)
            })
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        try:
            momentum_data = pd.DataFrame()
            
            # Rate of Change
            momentum_data['ROC_10'] = ta.roc(data['close'], length=10)
            momentum_data['ROC_20'] = ta.roc(data['close'], length=20)
            
            # Williams %R
            momentum_data['WILLR'] = ta.willr(data['high'], data['low'], data['close'], length=14)
            
            # Commodity Channel Index
            momentum_data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=20)
            
            return momentum_data.fillna(0)
            
        except:
            return pd.DataFrame({
                'ROC_10': data['close'].pct_change(10),
                'ROC_20': data['close'].pct_change(20),
                'WILLR': [-50] * len(data),
                'CCI': [0] * len(data)
            }).fillna(0)
    
    def identify_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """Identify support and resistance levels"""
        try:
            if len(data) < window * 2:
                current_price = data['close'].iloc[-1]
                return {
                    'support': current_price * 0.98,
                    'resistance': current_price * 1.02
                }
            
            # Find local minima and maxima
            highs = data['high'].rolling(window=window, center=True).max()
            lows = data['low'].rolling(window=window, center=True).min()
            
            # Current price levels
            recent_highs = data['high'].tail(window).max()
            recent_lows = data['low'].tail(window).min()
            
            return {
                'support': recent_lows,
                'resistance': recent_highs,
                'pivot': (recent_highs + recent_lows) / 2
            }
            
        except:
            current_price = data['close'].iloc[-1] if not data.empty else 0
            return {
                'support': current_price * 0.98,
                'resistance': current_price * 1.02,
                'pivot': current_price
            }
