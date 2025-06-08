import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RiskMetrics:
    """Risk metrics container"""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0

@dataclass
class PositionLimits:
    """Position sizing limits"""
    max_position_size: float = 0.1  # 10% of portfolio
    max_leverage: float = 1.0
    max_correlation_exposure: float = 0.3  # 30% in correlated assets
    max_sector_exposure: float = 0.4  # 40% in single sector
    max_daily_trades: int = 10
    max_drawdown_limit: float = 0.2  # 20% max drawdown

class AdvancedRiskManager:
    """Advanced risk management with dynamic position sizing and portfolio optimization"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Risk parameters
        self.risk_params = {
            'confidence_level': 0.95,
            'lookback_period': 252,  # Trading days
            'rebalance_frequency': 'daily',
            'volatility_target': 0.15,  # 15% annual volatility target
            'max_individual_weight': 0.2,  # 20% max per asset
            'correlation_threshold': 0.7,  # High correlation threshold
        }
        
        # Position tracking
        self.positions = {}
        self.position_history = []
        self.trade_history = []
        
        # Risk monitoring
        self.daily_returns = []
        self.portfolio_values = []
        self.risk_metrics_history = []
        
        # Volatility regimes
        self.volatility_regimes = {
            'low': {'threshold': 0.01, 'multiplier': 1.2},
            'medium': {'threshold': 0.03, 'multiplier': 1.0},
            'high': {'threshold': 0.05, 'multiplier': 0.7},
            'extreme': {'threshold': float('inf'), 'multiplier': 0.4}
        }
        
        # Correlation matrix for portfolio optimization
        self.correlation_matrix = pd.DataFrame()
        self.covariance_matrix = pd.DataFrame()
        
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              current_price: float, volatility: float,
                              market_regime: str = 'normal') -> Dict[str, Any]:
        """Calculate optimal position size using multiple methods"""
        try:
            # Base Kelly Criterion sizing
            kelly_size = self._kelly_criterion_sizing(signal_strength, volatility)
            
            # Volatility-adjusted sizing
            vol_adjusted_size = self._volatility_adjusted_sizing(volatility, market_regime)
            
            # Risk parity sizing
            risk_parity_size = self._risk_parity_sizing(symbol, volatility)
            
            # Portfolio heat sizing
            portfolio_heat_size = self._portfolio_heat_sizing()
            
            # Combine sizing methods
            combined_size = np.mean([kelly_size, vol_adjusted_size, risk_parity_size, portfolio_heat_size])
            
            # Apply risk limits
            final_size = self._apply_risk_limits(symbol, combined_size, current_price)
            
            # Calculate position value
            position_value = final_size * current_price
            
            return {
                'symbol': symbol,
                'recommended_size': final_size,
                'position_value': position_value,
                'kelly_size': kelly_size,
                'vol_adjusted_size': vol_adjusted_size,
                'risk_parity_size': risk_parity_size,
                'portfolio_heat_size': portfolio_heat_size,
                'risk_score': self._calculate_risk_score(symbol, final_size, volatility),
                'max_loss': position_value * 0.05,  # 5% stop loss
                'confidence': min(0.95, signal_strength)
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'recommended_size': 0.0,
                'error': str(e)
            }
    
    def _kelly_criterion_sizing(self, signal_strength: float, volatility: float) -> float:
        """Calculate position size using Kelly Criterion"""
        try:
            # Estimate win probability and average win/loss from signal strength
            win_prob = 0.5 + (signal_strength * 0.2)  # Convert to probability
            avg_win = 0.02 + (signal_strength * 0.01)  # Expected return
            avg_loss = 0.01 + (volatility * 0.02)  # Expected loss
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds received (avg_win/avg_loss), p = win_prob, q = 1-p
            if avg_loss > 0:
                b = avg_win / avg_loss
                kelly_fraction = (b * win_prob - (1 - win_prob)) / b
            else:
                kelly_fraction = 0.0
            
            # Apply fractional Kelly (quarter Kelly) for safety
            kelly_size = max(0, min(0.25, kelly_fraction * 0.25))
            
            return kelly_size
            
        except Exception as e:
            print(f"Error in Kelly sizing: {e}")
            return 0.05  # Conservative default
    
    def _volatility_adjusted_sizing(self, volatility: float, market_regime: str) -> float:
        """Adjust position size based on volatility regime"""
        try:
            # Base size inversely proportional to volatility
            base_size = self.risk_params['volatility_target'] / max(volatility, 0.01)
            
            # Apply regime multiplier
            regime_multiplier = 1.0
            for regime, config in self.volatility_regimes.items():
                if volatility <= config['threshold']:
                    regime_multiplier = config['multiplier']
                    break
            
            # Market regime adjustment
            regime_adjustments = {
                'trending': 1.1,
                'ranging': 0.9,
                'volatile': 0.7,
                'stable': 1.2
            }
            
            market_multiplier = regime_adjustments.get(market_regime, 1.0)
            
            adjusted_size = base_size * regime_multiplier * market_multiplier
            return min(0.2, max(0.01, adjusted_size))
            
        except Exception as e:
            print(f"Error in volatility sizing: {e}")
            return 0.05
    
    def _risk_parity_sizing(self, symbol: str, volatility: float) -> float:
        """Calculate risk parity position size"""
        try:
            # Get current portfolio volatilities
            portfolio_vols = []
            for pos_symbol, position in self.positions.items():
                if pos_symbol != symbol and position.get('volatility'):
                    portfolio_vols.append(position['volatility'])
            
            if not portfolio_vols:
                return 0.1  # Default if no other positions
            
            # Risk parity: equal risk contribution
            avg_portfolio_vol = np.mean(portfolio_vols)
            target_vol = avg_portfolio_vol
            
            # Size inversely proportional to volatility for equal risk
            risk_parity_size = target_vol / max(volatility, 0.01)
            
            return min(0.15, max(0.02, risk_parity_size))
            
        except Exception as e:
            print(f"Error in risk parity sizing: {e}")
            return 0.05
    
    def _portfolio_heat_sizing(self) -> float:
        """Calculate size based on current portfolio heat (risk exposure)"""
        try:
            # Calculate current portfolio risk exposure
            total_risk = 0.0
            for position in self.positions.values():
                position_risk = position.get('risk_score', 0.0)
                total_risk += position_risk
            
            # Target portfolio heat
            target_heat = 1.0
            available_heat = max(0, target_heat - total_risk)
            
            # Convert available heat to position size
            heat_size = available_heat * 0.1
            
            return min(0.15, max(0.01, heat_size))
            
        except Exception as e:
            print(f"Error in portfolio heat sizing: {e}")
            return 0.05
    
    def _apply_risk_limits(self, symbol: str, proposed_size: float, price: float) -> float:
        """Apply portfolio-level risk limits"""
        try:
            # Maximum position size limit
            max_size = self.risk_params['max_individual_weight']
            proposed_size = min(proposed_size, max_size)
            
            # Check portfolio concentration
            current_exposure = sum(
                pos.get('weight', 0) for pos in self.positions.values()
            )
            
            if current_exposure + proposed_size > 0.9:  # Max 90% invested
                proposed_size = max(0, 0.9 - current_exposure)
            
            # Minimum position size (transaction cost consideration)
            min_position_value = 100  # $100 minimum
            min_size = min_position_value / price
            
            if proposed_size < min_size:
                return 0.0  # Too small, don't trade
            
            return proposed_size
            
        except Exception as e:
            print(f"Error applying risk limits: {e}")
            return min(0.05, proposed_size)
    
    def _calculate_risk_score(self, symbol: str, size: float, volatility: float) -> float:
        """Calculate risk score for position"""
        try:
            # Base risk from size and volatility
            base_risk = size * volatility
            
            # Correlation risk
            correlation_risk = self._calculate_correlation_risk(symbol, size)
            
            # Concentration risk
            concentration_risk = size  # Higher size = higher concentration risk
            
            # Combined risk score
            total_risk = base_risk + correlation_risk + concentration_risk
            
            return min(1.0, total_risk)
            
        except Exception as e:
            print(f"Error calculating risk score: {e}")
            return 0.1
    
    def _calculate_correlation_risk(self, symbol: str, size: float) -> float:
        """Calculate correlation risk with existing positions"""
        try:
            if symbol not in self.correlation_matrix.index:
                return 0.0  # No correlation data
            
            correlation_risk = 0.0
            for pos_symbol, position in self.positions.items():
                if pos_symbol in self.correlation_matrix.columns:
                    correlation = abs(self.correlation_matrix.loc[symbol, pos_symbol])
                    pos_weight = position.get('weight', 0)
                    correlation_risk += correlation * pos_weight * size
            
            return correlation_risk
            
        except Exception as e:
            print(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def calculate_stop_loss(self, entry_price: float, position_size: float, 
                           volatility: float, atr: float = None) -> Dict[str, float]:
        """Calculate dynamic stop loss levels"""
        try:
            # ATR-based stop loss
            if atr:
                atr_stop = entry_price - (atr * 2.0)  # 2x ATR
            else:
                atr_stop = entry_price * 0.95  # 5% default
            
            # Volatility-based stop loss
            vol_stop = entry_price - (entry_price * volatility * 2.0)
            
            # Fixed percentage stop loss
            fixed_stop = entry_price * 0.95  # 5%
            
            # Portfolio heat stop loss
            max_loss_per_trade = self.current_capital * 0.02  # 2% of capital
            position_value = entry_price * position_size
            heat_stop_pct = max_loss_per_trade / position_value
            heat_stop = entry_price * (1 - heat_stop_pct)
            
            # Choose the most conservative (highest) stop loss
            final_stop = max(atr_stop, vol_stop, fixed_stop, heat_stop)
            
            return {
                'stop_loss_price': final_stop,
                'atr_stop': atr_stop,
                'volatility_stop': vol_stop,
                'fixed_stop': fixed_stop,
                'heat_stop': heat_stop,
                'stop_loss_pct': (entry_price - final_stop) / entry_price
            }
            
        except Exception as e:
            print(f"Error calculating stop loss: {e}")
            return {
                'stop_loss_price': entry_price * 0.95,
                'stop_loss_pct': 0.05
            }
    
    def calculate_take_profit(self, entry_price: float, signal_strength: float,
                             volatility: float, risk_reward_ratio: float = 2.0) -> Dict[str, float]:
        """Calculate dynamic take profit levels"""
        try:
            # Risk-reward based take profit
            stop_loss = entry_price * 0.95  # Assume 5% stop
            risk_amount = entry_price - stop_loss
            rr_take_profit = entry_price + (risk_amount * risk_reward_ratio)
            
            # Signal strength based take profit
            signal_multiplier = 1.0 + (signal_strength * 0.5)
            signal_take_profit = entry_price * signal_multiplier
            
            # Volatility based take profit
            vol_take_profit = entry_price + (entry_price * volatility * 3.0)
            
            # Technical resistance levels (simplified)
            resistance_level = entry_price * 1.1  # 10% above entry
            
            # Choose based on signal strength
            if signal_strength > 0.7:
                final_take_profit = max(rr_take_profit, signal_take_profit)
            else:
                final_take_profit = min(rr_take_profit, vol_take_profit)
            
            return {
                'take_profit_price': final_take_profit,
                'risk_reward_take_profit': rr_take_profit,
                'signal_take_profit': signal_take_profit,
                'volatility_take_profit': vol_take_profit,
                'resistance_take_profit': resistance_level,
                'take_profit_pct': (final_take_profit - entry_price) / entry_price
            }
            
        except Exception as e:
            print(f"Error calculating take profit: {e}")
            return {
                'take_profit_price': entry_price * 1.1,
                'take_profit_pct': 0.1
            }
    
    def update_position(self, symbol: str, size: float, price: float, 
                       volatility: float, timestamp: datetime = None):
        """Update position tracking"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            weight = (size * price) / self.current_capital
            
            self.positions[symbol] = {
                'size': size,
                'price': price,
                'value': size * price,
                'weight': weight,
                'volatility': volatility,
                'timestamp': timestamp,
                'risk_score': self._calculate_risk_score(symbol, weight, volatility)
            }
            
            # Update position history
            self.position_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'update',
                'size': size,
                'price': price,
                'weight': weight
            })
            
        except Exception as e:
            print(f"Error updating position: {e}")
    
    def remove_position(self, symbol: str, timestamp: datetime = None):
        """Remove position from tracking"""
        try:
            if symbol in self.positions:
                if timestamp is None:
                    timestamp = datetime.now()
                
                # Record removal
                self.position_history.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'remove',
                    'size': 0,
                    'price': 0,
                    'weight': 0
                })
                
                del self.positions[symbol]
                
        except Exception as e:
            print(f"Error removing position: {e}")
    
    def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if len(self.daily_returns) < 30:
                return RiskMetrics()
            
            returns = np.array(self.daily_returns[-252:])  # Last year
            
            # Value at Risk
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional Value at Risk
            cvar_95 = np.mean(returns[returns <= var_95])
            cvar_99 = np.mean(returns[returns <= var_99])
            
            # Maximum Drawdown
            portfolio_values = np.array(self.portfolio_values[-252:])
            if len(portfolio_values) > 0:
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (peak - portfolio_values) / peak
                max_drawdown = np.max(drawdown)
            else:
                max_drawdown = 0.0
            
            # Sharpe Ratio
            if len(returns) > 1:
                excess_returns = returns - 0.02/252  # Assume 2% risk-free rate
                sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252)
            else:
                sortino_ratio = 0.0
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                volatility=volatility
            )
            
        except Exception as e:
            print(f"Error calculating portfolio risk: {e}")
            return RiskMetrics()
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """Check if portfolio exceeds risk limits"""
        try:
            risk_metrics = self.calculate_portfolio_risk()
            warnings = []
            violations = []
            
            # Check maximum drawdown
            if risk_metrics.max_drawdown > 0.2:  # 20% limit
                violations.append(f"Maximum drawdown exceeded: {risk_metrics.max_drawdown:.2%}")
            elif risk_metrics.max_drawdown > 0.15:  # 15% warning
                warnings.append(f"High drawdown warning: {risk_metrics.max_drawdown:.2%}")
            
            # Check portfolio concentration
            total_weight = sum(pos.get('weight', 0) for pos in self.positions.values())
            if total_weight > 0.95:  # 95% limit
                violations.append(f"Over-invested: {total_weight:.1%} of portfolio")
            elif total_weight > 0.9:  # 90% warning
                warnings.append(f"High exposure: {total_weight:.1%} of portfolio")
            
            # Check individual position limits
            for symbol, position in self.positions.items():
                weight = position.get('weight', 0)
                if weight > 0.25:  # 25% limit per position
                    violations.append(f"{symbol} exceeds position limit: {weight:.1%}")
                elif weight > 0.2:  # 20% warning
                    warnings.append(f"{symbol} high concentration: {weight:.1%}")
            
            # Check volatility
            if risk_metrics.volatility > 0.3:  # 30% annual volatility limit
                violations.append(f"Excessive volatility: {risk_metrics.volatility:.1%}")
            elif risk_metrics.volatility > 0.25:  # 25% warning
                warnings.append(f"High volatility: {risk_metrics.volatility:.1%}")
            
            return {
                'risk_status': 'VIOLATION' if violations else ('WARNING' if warnings else 'OK'),
                'violations': violations,
                'warnings': warnings,
                'risk_metrics': risk_metrics,
                'total_exposure': total_weight,
                'position_count': len(self.positions)
            }
            
        except Exception as e:
            print(f"Error checking risk limits: {e}")
            return {'risk_status': 'ERROR', 'error': str(e)}
    
    def optimize_portfolio(self, expected_returns: Dict[str, float], 
                          target_risk: float = None) -> Dict[str, float]:
        """Optimize portfolio weights using mean-variance optimization"""
        try:
            if not expected_returns or len(self.correlation_matrix) == 0:
                return {}
            
            symbols = list(expected_returns.keys())
            n_assets = len(symbols)
            
            if n_assets < 2:
                return {symbols[0]: 1.0} if symbols else {}
            
            # Create expected returns vector
            mu = np.array([expected_returns[symbol] for symbol in symbols])
            
            # Create covariance matrix
            if self.covariance_matrix.empty:
                # Use simple variance if no covariance data
                variances = [0.01] * n_assets  # 1% variance assumption
                cov_matrix = np.diag(variances)
            else:
                # Use available correlation data
                cov_matrix = self.covariance_matrix.loc[symbols, symbols].values
            
            # Target risk level
            if target_risk is None:
                target_risk = self.risk_params['volatility_target']
            
            # Optimize using simplified mean-variance (equal weights with adjustments)
            base_weights = np.ones(n_assets) / n_assets
            
            # Adjust weights based on expected returns
            return_adjustments = mu / np.sum(np.abs(mu)) if np.sum(np.abs(mu)) > 0 else np.zeros(n_assets)
            
            # Combine base weights with return adjustments
            optimized_weights = base_weights + (return_adjustments * 0.2)
            
            # Normalize weights
            optimized_weights = np.maximum(0, optimized_weights)  # No short selling
            optimized_weights = optimized_weights / np.sum(optimized_weights)
            
            # Apply position limits
            max_weight = self.risk_params['max_individual_weight']
            optimized_weights = np.minimum(optimized_weights, max_weight)
            optimized_weights = optimized_weights / np.sum(optimized_weights)
            
            return dict(zip(symbols, optimized_weights))
            
        except Exception as e:
            print(f"Error optimizing portfolio: {e}")
            return {}
    
    def update_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]):
        """Update correlation matrix from price data"""
        try:
            returns_data = {}
            
            for symbol, df in price_data.items():
                if 'close' in df.columns and len(df) > 1:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 10:  # Minimum data requirement
                        returns_data[symbol] = returns
            
            if len(returns_data) >= 2:
                # Align data by common dates
                returns_df = pd.DataFrame(returns_data)
                returns_df = returns_df.dropna()
                
                if len(returns_df) > 10:
                    self.correlation_matrix = returns_df.corr()
                    self.covariance_matrix = returns_df.cov()
                    
        except Exception as e:
            print(f"Error updating correlation matrix: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            risk_metrics = self.calculate_portfolio_risk()
            risk_check = self.check_risk_limits()
            
            # Position breakdown
            position_summary = {}
            total_value = 0
            
            for symbol, position in self.positions.items():
                value = position.get('value', 0)
                total_value += value
                
                position_summary[symbol] = {
                    'weight': position.get('weight', 0),
                    'value': value,
                    'risk_score': position.get('risk_score', 0),
                    'volatility': position.get('volatility', 0)
                }
            
            return {
                'portfolio_value': self.current_capital,
                'invested_value': total_value,
                'cash_available': self.current_capital - total_value,
                'risk_metrics': risk_metrics.__dict__,
                'risk_status': risk_check['risk_status'],
                'violations': risk_check.get('violations', []),
                'warnings': risk_check.get('warnings', []),
                'positions': position_summary,
                'correlation_matrix_size': len(self.correlation_matrix),
                'recent_trades': len(self.trade_history[-30:])  # Last 30 trades
            }
            
        except Exception as e:
            print(f"Error getting risk summary: {e}")
            return {'error': str(e)}