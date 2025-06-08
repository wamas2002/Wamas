"""
Advanced Strategy Engine
Implements DCA, Grid, Breakout, Ping-Pong, and Trailing ATR strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        self.symbol = symbol
        self.config = config
        self.active_orders = []
        self.position_size = 0
        self.entry_price = 0
        self.last_signal_time = None
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate trading signals based on strategy logic"""
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information and status"""
        pass

class DCAStrategy(BaseStrategy):
    """Dollar Cost Averaging Strategy - Buy more after every X% drop"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.drop_threshold = config.get('drop_threshold', 0.05)  # 5% default
        self.max_orders = config.get('max_orders', 5)
        self.order_amount = config.get('order_amount', 100)
        self.last_buy_price = None
        self.dca_levels = []
        
    def generate_signals(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate DCA buy signals"""
        if data.empty:
            return {'action': 'hold', 'reason': 'No data available'}
        
        # Calculate price drop from last buy or recent high
        reference_price = self.last_buy_price or data['high'].tail(20).max()
        price_drop = (reference_price - current_price) / reference_price
        
        signal = {'action': 'hold', 'amount': 0, 'price': current_price}
        
        # Buy signal if price dropped enough and we haven't reached max orders
        if price_drop >= self.drop_threshold and len(self.dca_levels) < self.max_orders:
            signal = {
                'action': 'buy',
                'amount': self.order_amount,
                'price': current_price,
                'reason': f'DCA buy: {price_drop:.2%} drop from ${reference_price:.2f}',
                'stop_loss': current_price * 0.95,  # 5% SL
                'take_profit': current_price * 1.10  # 10% TP
            }
            self.dca_levels.append(current_price)
            self.last_buy_price = current_price
        
        # Sell signal if price recovered significantly
        elif self.position_size > 0:
            avg_buy_price = np.mean(self.dca_levels) if self.dca_levels else self.entry_price
            profit_pct = (current_price - avg_buy_price) / avg_buy_price
            
            if profit_pct >= 0.15:  # 15% profit target
                signal = {
                    'action': 'sell',
                    'amount': self.position_size,
                    'price': current_price,
                    'reason': f'DCA profit taking: {profit_pct:.2%} gain'
                }
                self.dca_levels = []  # Reset for next cycle
        
        return signal
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'DCA Strategy',
            'drop_threshold': f"{self.drop_threshold:.1%}",
            'active_levels': len(self.dca_levels),
            'max_orders': self.max_orders,
            'avg_buy_price': np.mean(self.dca_levels) if self.dca_levels else None
        }

class GridStrategy(BaseStrategy):
    """Grid Trading Strategy - Set buy/sell orders in grid levels"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.grid_size = config.get('grid_size', 0.02)  # 2% between levels
        self.num_levels = config.get('num_levels', 10)
        self.order_amount = config.get('order_amount', 50)
        self.grid_center = None
        self.buy_levels = []
        self.sell_levels = []
        
    def setup_grid(self, current_price: float):
        """Setup grid levels around current price"""
        self.grid_center = current_price
        
        # Create buy levels below current price
        self.buy_levels = []
        for i in range(1, self.num_levels // 2 + 1):
            level = current_price * (1 - self.grid_size * i)
            self.buy_levels.append(level)
        
        # Create sell levels above current price
        self.sell_levels = []
        for i in range(1, self.num_levels // 2 + 1):
            level = current_price * (1 + self.grid_size * i)
            self.sell_levels.append(level)
    
    def generate_signals(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate grid trading signals"""
        if self.grid_center is None:
            self.setup_grid(current_price)
        
        signal = {'action': 'hold', 'amount': 0, 'price': current_price}
        
        # Check for buy signals (price hits buy level)
        for level in self.buy_levels:
            if abs(current_price - level) <= level * 0.001:  # Within 0.1%
                signal = {
                    'action': 'buy',
                    'amount': self.order_amount,
                    'price': level,
                    'reason': f'Grid buy at ${level:.2f}',
                    'take_profit': level * (1 + self.grid_size)
                }
                break
        
        # Check for sell signals (price hits sell level)
        for level in self.sell_levels:
            if abs(current_price - level) <= level * 0.001:  # Within 0.1%
                signal = {
                    'action': 'sell',
                    'amount': self.order_amount,
                    'price': level,
                    'reason': f'Grid sell at ${level:.2f}',
                    'stop_loss': level * (1 - self.grid_size)
                }
                break
        
        return signal
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Grid Strategy',
            'grid_size': f"{self.grid_size:.1%}",
            'num_levels': self.num_levels,
            'grid_center': self.grid_center,
            'buy_levels': len(self.buy_levels),
            'sell_levels': len(self.sell_levels)
        }

class BreakoutStrategy(BaseStrategy):
    """Breakout Strategy - Trade based on volume/volatility breakouts"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.volatility_threshold = config.get('volatility_threshold', 2.0)  # 2x average
        self.volume_threshold = config.get('volume_threshold', 1.5)  # 1.5x average
        self.lookback_period = config.get('lookback_period', 20)
        
    def calculate_breakout_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility and volume breakout indicators"""
        if len(data) < self.lookback_period:
            return {'volatility_ratio': 1.0, 'volume_ratio': 1.0}
        
        # Calculate volatility (ATR)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        current_volatility = atr.tail(1).iloc[0]
        avg_volatility = atr.tail(self.lookback_period).mean()
        volatility_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        
        # Calculate volume ratio
        current_volume = data['volume'].tail(1).iloc[0]
        avg_volume = data['volume'].tail(self.lookback_period).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return {
            'volatility_ratio': volatility_ratio,
            'volume_ratio': volume_ratio
        }
    
    def generate_signals(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate breakout trading signals"""
        if len(data) < self.lookback_period:
            return {'action': 'hold', 'reason': 'Insufficient data'}
        
        breakout_data = self.calculate_breakout_signals(data)
        volatility_ratio = breakout_data['volatility_ratio']
        volume_ratio = breakout_data['volume_ratio']
        
        # Determine price direction
        recent_prices = data['close'].tail(5)
        price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        signal = {'action': 'hold', 'amount': 0, 'price': current_price}
        
        # Breakout conditions
        is_volatility_breakout = volatility_ratio >= self.volatility_threshold
        is_volume_breakout = volume_ratio >= self.volume_threshold
        
        if is_volatility_breakout and is_volume_breakout:
            if price_momentum > 0.02:  # Upward breakout
                signal = {
                    'action': 'buy',
                    'amount': 100,
                    'price': current_price,
                    'reason': f'Upward breakout: Vol {volatility_ratio:.1f}x, Volume {volume_ratio:.1f}x',
                    'stop_loss': current_price * 0.95,
                    'take_profit': current_price * 1.08
                }
            elif price_momentum < -0.02:  # Downward breakout
                signal = {
                    'action': 'sell',
                    'amount': 100,
                    'price': current_price,
                    'reason': f'Downward breakout: Vol {volatility_ratio:.1f}x, Volume {volume_ratio:.1f}x',
                    'stop_loss': current_price * 1.05,
                    'take_profit': current_price * 0.92
                }
        
        return signal
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Breakout Strategy',
            'volatility_threshold': f"{self.volatility_threshold}x",
            'volume_threshold': f"{self.volume_threshold}x",
            'lookback_period': self.lookback_period
        }

class PingPongStrategy(BaseStrategy):
    """Ping-Pong Strategy - Bounce between upper and lower price bands"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.band_width = config.get('band_width', 0.04)  # 4% bands
        self.lookback_period = config.get('lookback_period', 20)
        self.upper_band = None
        self.lower_band = None
        self.center_price = None
        
    def update_bands(self, data: pd.DataFrame):
        """Update price bands based on recent data"""
        if len(data) < self.lookback_period:
            return
        
        recent_prices = data['close'].tail(self.lookback_period)
        self.center_price = recent_prices.mean()
        self.upper_band = self.center_price * (1 + self.band_width)
        self.lower_band = self.center_price * (1 - self.band_width)
    
    def generate_signals(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate ping-pong trading signals"""
        self.update_bands(data)
        
        if self.upper_band is None or self.lower_band is None:
            return {'action': 'hold', 'reason': 'Bands not established'}
        
        signal = {'action': 'hold', 'amount': 0, 'price': current_price}
        
        # Buy at lower band
        if current_price <= self.lower_band:
            signal = {
                'action': 'buy',
                'amount': 100,
                'price': current_price,
                'reason': f'Ping-pong buy at lower band ${self.lower_band:.2f}',
                'take_profit': self.upper_band,
                'stop_loss': self.lower_band * 0.98
            }
        
        # Sell at upper band
        elif current_price >= self.upper_band:
            signal = {
                'action': 'sell',
                'amount': 100,
                'price': current_price,
                'reason': f'Ping-pong sell at upper band ${self.upper_band:.2f}',
                'take_profit': self.lower_band,
                'stop_loss': self.upper_band * 1.02
            }
        
        return signal
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Ping-Pong Strategy',
            'band_width': f"{self.band_width:.1%}",
            'upper_band': self.upper_band,
            'lower_band': self.lower_band,
            'center_price': self.center_price
        }

class TrailingATRStrategy(BaseStrategy):
    """Trailing ATR Strategy - Use ATR for dynamic trailing SL/TP"""
    
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.atr_period = config.get('atr_period', 14)
        self.trailing_sl = None
        self.trailing_tp = None
        self.trend_direction = None
        
    def calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        if len(data) < self.atr_period:
            return 0
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.tail(self.atr_period).mean()
    
    def update_trailing_levels(self, current_price: float, atr: float):
        """Update trailing stop-loss and take-profit levels"""
        atr_distance = atr * self.atr_multiplier
        
        if self.trend_direction == 'up':
            new_sl = current_price - atr_distance
            if self.trailing_sl is None or new_sl > self.trailing_sl:
                self.trailing_sl = new_sl
            
            self.trailing_tp = current_price + atr_distance
            
        elif self.trend_direction == 'down':
            new_sl = current_price + atr_distance
            if self.trailing_sl is None or new_sl < self.trailing_sl:
                self.trailing_sl = new_sl
            
            self.trailing_tp = current_price - atr_distance
    
    def generate_signals(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate trailing ATR signals"""
        if len(data) < self.atr_period + 5:
            return {'action': 'hold', 'reason': 'Insufficient data for ATR'}
        
        atr = self.calculate_atr(data)
        
        # Determine trend direction using EMA
        ema_short = data['close'].ewm(span=9).mean().tail(1).iloc[0]
        ema_long = data['close'].ewm(span=21).mean().tail(1).iloc[0]
        
        new_trend = 'up' if ema_short > ema_long else 'down'
        
        signal = {'action': 'hold', 'amount': 0, 'price': current_price}
        
        # Trend change - enter position
        if new_trend != self.trend_direction:
            self.trend_direction = new_trend
            self.trailing_sl = None
            
            if new_trend == 'up':
                signal = {
                    'action': 'buy',
                    'amount': 100,
                    'price': current_price,
                    'reason': f'ATR trend change to UP, ATR: {atr:.2f}',
                    'stop_loss': current_price - (atr * self.atr_multiplier),
                    'take_profit': current_price + (atr * self.atr_multiplier)
                }
            else:
                signal = {
                    'action': 'sell',
                    'amount': 100,
                    'price': current_price,
                    'reason': f'ATR trend change to DOWN, ATR: {atr:.2f}',
                    'stop_loss': current_price + (atr * self.atr_multiplier),
                    'take_profit': current_price - (atr * self.atr_multiplier)
                }
        
        # Update trailing levels for existing position
        elif self.position_size != 0:
            self.update_trailing_levels(current_price, atr)
            
            # Check for trailing stop hit
            if self.trend_direction == 'up' and current_price <= self.trailing_sl:
                signal = {
                    'action': 'sell',
                    'amount': self.position_size,
                    'price': current_price,
                    'reason': f'Trailing stop hit at ${self.trailing_sl:.2f}'
                }
            elif self.trend_direction == 'down' and current_price >= self.trailing_sl:
                signal = {
                    'action': 'buy',
                    'amount': abs(self.position_size),
                    'price': current_price,
                    'reason': f'Trailing stop hit at ${self.trailing_sl:.2f}'
                }
        
        return signal
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Trailing ATR Strategy',
            'atr_multiplier': self.atr_multiplier,
            'atr_period': self.atr_period,
            'trend_direction': self.trend_direction,
            'trailing_sl': self.trailing_sl,
            'trailing_tp': self.trailing_tp
        }

class StrategyEngine:
    """Main strategy engine that manages all trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.strategy_classes = {
            'dca': DCAStrategy,
            'grid': GridStrategy,
            'breakout': BreakoutStrategy,
            'ping_pong': PingPongStrategy,
            'trailing_atr': TrailingATRStrategy
        }
        self.default_configs = {
            'dca': {
                'drop_threshold': 0.05,
                'max_orders': 5,
                'order_amount': 100
            },
            'grid': {
                'grid_size': 0.02,
                'num_levels': 10,
                'order_amount': 50
            },
            'breakout': {
                'volatility_threshold': 2.0,
                'volume_threshold': 1.5,
                'lookback_period': 20
            },
            'ping_pong': {
                'band_width': 0.04,
                'lookback_period': 20
            },
            'trailing_atr': {
                'atr_multiplier': 2.0,
                'atr_period': 14
            }
        }
    
    def create_strategy(self, symbol: str, strategy_type: str, config: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """Create a new strategy instance"""
        if strategy_type not in self.strategy_classes:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Use default config if none provided
        strategy_config = config or self.default_configs.get(strategy_type, {})
        
        strategy_class = self.strategy_classes[strategy_type]
        strategy = strategy_class(symbol, strategy_config)
        
        key = f"{symbol}_{strategy_type}"
        self.strategies[key] = strategy
        
        return strategy
    
    def get_strategy(self, symbol: str, strategy_type: str) -> Optional[BaseStrategy]:
        """Get existing strategy instance"""
        key = f"{symbol}_{strategy_type}"
        return self.strategies.get(key)
    
    def generate_signals(self, symbol: str, strategy_type: str, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate trading signals for specified strategy"""
        strategy = self.get_strategy(symbol, strategy_type)
        if not strategy:
            strategy = self.create_strategy(symbol, strategy_type)
        
        return strategy.generate_signals(data, current_price)
    
    def get_all_strategies_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about all strategies for a symbol"""
        info = {}
        for strategy_type in self.strategy_classes.keys():
            strategy = self.get_strategy(symbol, strategy_type)
            if strategy:
                info[strategy_type] = strategy.get_strategy_info()
        return info
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy types"""
        return list(self.strategy_classes.keys())