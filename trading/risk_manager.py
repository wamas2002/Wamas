import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math
from config import Config

class RiskManager:
    """Risk management system for position sizing and risk control"""
    
    def __init__(self):
        self.portfolio_value = 10000.0
        self.max_portfolio_risk = Config.RISK_PARAMS['max_drawdown']
        self.max_position_size = Config.RISK_PARAMS['max_position_size']
        self.stop_loss_pct = Config.RISK_PARAMS['stop_loss_pct']
        self.take_profit_pct = Config.RISK_PARAMS['take_profit_pct']
        
        # Risk tracking
        self.position_risks = {}
        self.portfolio_risk = 0.0
        self.drawdown_history = []
        self.risk_metrics = {}
        
        # Volatility-based sizing
        self.volatility_lookback = 20
        self.volatility_target = 0.02  # 2% daily volatility target
        
    def initialize(self, initial_portfolio_value: float):
        """Initialize risk manager with portfolio value"""
        self.portfolio_value = initial_portfolio_value
        print(f"Risk manager initialized with portfolio value: ${initial_portfolio_value:,.2f}")
    
    def calculate_position_size(self, portfolio_value: float, asset_price: float, 
                              signal_strength: float, volatility: float = None) -> float:
        """Calculate optimal position size based on risk parameters"""
        try:
            if portfolio_value <= 0 or asset_price <= 0:
                return 0.0
            
            # Base position size as percentage of portfolio
            base_size_pct = self.max_position_size * signal_strength
            base_size_value = portfolio_value * base_size_pct
            
            # Volatility adjustment
            if volatility is not None and volatility > 0:
                # Inverse volatility sizing - reduce size for higher volatility
                volatility_adj = min(2.0, self.volatility_target / volatility)
                base_size_value *= volatility_adj
            
            # Convert to quantity
            max_quantity = base_size_value / asset_price
            
            # Apply additional constraints
            max_quantity = self._apply_risk_constraints(max_quantity, asset_price, portfolio_value)
            
            return max(0.0, max_quantity)
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.0
    
    def _apply_risk_constraints(self, quantity: float, price: float, portfolio_value: float) -> float:
        """Apply additional risk constraints to position size"""
        try:
            position_value = quantity * price
            
            # Maximum position value constraint
            max_position_value = portfolio_value * self.max_position_size
            if position_value > max_position_value:
                quantity = max_position_value / price
            
            # Minimum position size (avoid dust trades)
            min_position_value = 10.0  # $10 minimum
            if position_value < min_position_value:
                return 0.0
            
            return quantity
            
        except Exception as e:
            print(f"Error applying risk constraints: {e}")
            return 0.0
    
    def calculate_volatility_adjusted_size(self, data: pd.DataFrame, base_size: float) -> float:
        """Calculate volatility-adjusted position size"""
        try:
            if len(data) < self.volatility_lookback:
                return base_size
            
            # Calculate recent volatility
            returns = data['close'].pct_change().dropna()
            recent_volatility = returns.tail(self.volatility_lookback).std()
            
            if recent_volatility <= 0:
                return base_size
            
            # Volatility scaling factor
            volatility_factor = self.volatility_target / recent_volatility
            volatility_factor = max(0.1, min(2.0, volatility_factor))  # Clamp between 0.1 and 2.0
            
            adjusted_size = base_size * volatility_factor
            
            return adjusted_size
            
        except Exception as e:
            print(f"Error calculating volatility-adjusted size: {e}")
            return base_size
    
    def calculate_stop_loss_price(self, entry_price: float, position_type: str, 
                                 atr: float = None) -> float:
        """Calculate stop loss price"""
        try:
            if atr is not None:
                # ATR-based stop loss
                stop_distance = atr * 2.0  # 2x ATR
            else:
                # Percentage-based stop loss
                stop_distance = entry_price * self.stop_loss_pct
            
            if position_type.upper() == 'LONG':
                stop_price = entry_price - stop_distance
            else:  # SHORT
                stop_price = entry_price + stop_distance
            
            return max(0.0, stop_price)
            
        except Exception as e:
            print(f"Error calculating stop loss price: {e}")
            return entry_price * (1 - self.stop_loss_pct) if position_type.upper() == 'LONG' else entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit_price(self, entry_price: float, position_type: str, 
                                   risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit price"""
        try:
            profit_distance = entry_price * self.take_profit_pct
            
            # Apply risk-reward ratio
            profit_distance *= risk_reward_ratio
            
            if position_type.upper() == 'LONG':
                tp_price = entry_price + profit_distance
            else:  # SHORT
                tp_price = entry_price - profit_distance
            
            return max(0.0, tp_price)
            
        except Exception as e:
            print(f"Error calculating take profit price: {e}")
            return entry_price * (1 + self.take_profit_pct) if position_type.upper() == 'LONG' else entry_price * (1 - self.take_profit_pct)
    
    def calculate_position_risk(self, entry_price: float, stop_loss_price: float, 
                               quantity: float) -> float:
        """Calculate risk amount for a position"""
        try:
            if stop_loss_price <= 0 or quantity <= 0:
                return 0.0
            
            risk_per_share = abs(entry_price - stop_loss_price)
            total_risk = risk_per_share * quantity
            
            return total_risk
            
        except Exception as e:
            print(f"Error calculating position risk: {e}")
            return 0.0
    
    def check_portfolio_risk(self, new_position_risk: float) -> bool:
        """Check if adding new position exceeds portfolio risk limits"""
        try:
            total_risk = self.portfolio_risk + new_position_risk
            risk_percentage = total_risk / self.portfolio_value
            
            return risk_percentage <= self.max_portfolio_risk
            
        except Exception as e:
            print(f"Error checking portfolio risk: {e}")
            return False
    
    def update_portfolio_risk(self, positions: Dict[str, Dict[str, Any]]):
        """Update total portfolio risk based on current positions"""
        try:
            total_risk = 0.0
            
            for symbol, position in positions.items():
                if position.get('quantity', 0) > 0:
                    entry_price = position.get('entry_price', 0)
                    stop_loss = position.get('stop_loss', 0)
                    quantity = position.get('quantity', 0)
                    
                    position_risk = self.calculate_position_risk(entry_price, stop_loss, quantity)
                    total_risk += position_risk
                    
                    self.position_risks[symbol] = position_risk
            
            self.portfolio_risk = total_risk
            
        except Exception as e:
            print(f"Error updating portfolio risk: {e}")
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        try:
            if len(portfolio_values) < 2:
                return 0.0
            
            values = np.array(portfolio_values)
            running_max = np.maximum.accumulate(values)
            drawdown = (running_max - values) / running_max
            max_drawdown = np.max(drawdown)
            
            return max_drawdown
            
        except Exception as e:
            print(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            returns_array = np.array(returns)
            excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
            
            return sharpe
            
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_value_at_risk(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        try:
            if len(returns) < 10:
                return 0.0
            
            returns_array = np.array(returns)
            var = np.percentile(returns_array, (1 - confidence_level) * 100)
            
            return abs(var)
            
        except Exception as e:
            print(f"Error calculating VaR: {e}")
            return 0.0
    
    def assess_market_risk(self, data: pd.DataFrame) -> Dict[str, float]:
        """Assess current market risk conditions"""
        try:
            if len(data) < 50:
                return {'risk_level': 0.5, 'volatility': 0.0, 'trend_strength': 0.0}
            
            # Calculate volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate trend strength
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            trend_strength = abs((sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1])
            
            # Volume analysis
            volume_ratio = data['volume'].iloc[-20:].mean() / data['volume'].iloc[-50:-20].mean()
            
            # Risk score (0-1, where 1 is highest risk)
            volatility_score = min(1.0, volatility / 0.5)  # Normalize to 50% annual vol
            trend_score = min(1.0, trend_strength * 10)
            volume_score = min(1.0, abs(volume_ratio - 1) * 2)
            
            risk_level = (volatility_score * 0.5 + trend_score * 0.3 + volume_score * 0.2)
            
            return {
                'risk_level': risk_level,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio,
                'volatility_score': volatility_score,
                'trend_score': trend_score,
                'volume_score': volume_score
            }
            
        except Exception as e:
            print(f"Error assessing market risk: {e}")
            return {'risk_level': 0.5, 'volatility': 0.0, 'trend_strength': 0.0}
    
    def adjust_risk_for_regime(self, base_risk: float, market_regime: str) -> float:
        """Adjust risk based on market regime"""
        try:
            regime_multipliers = {
                'trending': 1.0,    # Normal risk in trending markets
                'ranging': 0.7,     # Reduced risk in ranging markets
                'volatile': 0.5,    # Significantly reduced risk in volatile markets
                'stable': 1.2       # Slightly increased risk in stable markets
            }
            
            multiplier = regime_multipliers.get(market_regime, 1.0)
            adjusted_risk = base_risk * multiplier
            
            return max(0.1, min(2.0, adjusted_risk))  # Clamp between 0.1 and 2.0
            
        except Exception as e:
            print(f"Error adjusting risk for regime: {e}")
            return base_risk
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        try:
            return {
                'portfolio_value': self.portfolio_value,
                'portfolio_risk': self.portfolio_risk,
                'portfolio_risk_pct': (self.portfolio_risk / self.portfolio_value) * 100 if self.portfolio_value > 0 else 0,
                'max_portfolio_risk_pct': self.max_portfolio_risk * 100,
                'max_position_size_pct': self.max_position_size * 100,
                'stop_loss_pct': self.stop_loss_pct * 100,
                'take_profit_pct': self.take_profit_pct * 100,
                'position_risks': self.position_risks.copy(),
                'volatility_target': self.volatility_target
            }
            
        except Exception as e:
            print(f"Error getting risk metrics: {e}")
            return {}
    
    def update_risk_parameters(self, **kwargs):
        """Update risk parameters dynamically"""
        try:
            if 'max_position_size' in kwargs:
                self.max_position_size = max(0.01, min(1.0, kwargs['max_position_size']))
            
            if 'stop_loss_pct' in kwargs:
                self.stop_loss_pct = max(0.005, min(0.2, kwargs['stop_loss_pct']))
            
            if 'take_profit_pct' in kwargs:
                self.take_profit_pct = max(0.01, min(0.5, kwargs['take_profit_pct']))
            
            if 'max_portfolio_risk' in kwargs:
                self.max_portfolio_risk = max(0.05, min(1.0, kwargs['max_portfolio_risk']))
            
            print("Risk parameters updated successfully")
            
        except Exception as e:
            print(f"Error updating risk parameters: {e}")
    
    def simulate_position_outcome(self, entry_price: float, quantity: float, 
                                stop_loss: float, take_profit: float, 
                                win_probability: float = 0.5) -> Dict[str, float]:
        """Simulate potential position outcomes"""
        try:
            # Calculate potential loss and profit
            loss = abs(entry_price - stop_loss) * quantity
            profit = abs(take_profit - entry_price) * quantity
            
            # Expected value
            expected_value = (profit * win_probability) - (loss * (1 - win_probability))
            
            # Risk-reward ratio
            risk_reward = profit / loss if loss > 0 else 0
            
            return {
                'potential_loss': loss,
                'potential_profit': profit,
                'expected_value': expected_value,
                'risk_reward_ratio': risk_reward,
                'win_probability': win_probability
            }
            
        except Exception as e:
            print(f"Error simulating position outcome: {e}")
            return {}
