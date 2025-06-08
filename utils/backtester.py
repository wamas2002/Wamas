import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import copy
from dataclasses import dataclass

@dataclass
class Trade:
    """Trade record for backtesting"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    value: float
    fees: float = 0.0
    signal_strength: float = 0.0
    strategy: str = ""

@dataclass
class Position:
    """Position record for backtesting"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

class Backtester:
    """Comprehensive backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # Commission rate (e.g., 0.001 = 0.1%)
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # symbol -> Position
        self.portfolio_value = initial_capital
        
        # Trade tracking
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        
        # Performance metrics
        self.metrics = {}
        self.drawdown_history = []
        
    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        self.metrics = {}
        self.drawdown_history = []
    
    def execute_trade(self, timestamp: datetime, symbol: str, action: str, 
                     quantity: float, price: float, signal_strength: float = 0.0,
                     strategy: str = "") -> bool:
        """Execute a trade and update portfolio"""
        try:
            if action.upper() == 'BUY':
                return self._execute_buy(timestamp, symbol, quantity, price, signal_strength, strategy)
            elif action.upper() == 'SELL':
                return self._execute_sell(timestamp, symbol, quantity, price, signal_strength, strategy)
            else:
                print(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False
    
    def _execute_buy(self, timestamp: datetime, symbol: str, quantity: float, 
                    price: float, signal_strength: float, strategy: str) -> bool:
        """Execute buy order"""
        try:
            total_cost = quantity * price
            fees = total_cost * self.commission
            total_required = total_cost + fees
            
            if self.cash < total_required:
                print(f"Insufficient cash for buy order: {self.cash:.2f} < {total_required:.2f}")
                return False
            
            # Update cash
            self.cash -= total_required
            
            # Update position
            if symbol in self.positions:
                # Average down existing position
                existing_pos = self.positions[symbol]
                total_quantity = existing_pos.quantity + quantity
                avg_price = ((existing_pos.quantity * existing_pos.entry_price) + 
                           (quantity * price)) / total_quantity
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=total_quantity,
                    entry_price=avg_price,
                    entry_time=existing_pos.entry_time,
                    current_price=price
                )
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=timestamp,
                    current_price=price
                )
            
            # Record trade
            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                action='BUY',
                quantity=quantity,
                price=price,
                value=total_cost,
                fees=fees,
                signal_strength=signal_strength,
                strategy=strategy
            )
            self.trades.append(trade)
            
            return True
            
        except Exception as e:
            print(f"Error executing buy order: {e}")
            return False
    
    def _execute_sell(self, timestamp: datetime, symbol: str, quantity: float, 
                     price: float, signal_strength: float, strategy: str) -> bool:
        """Execute sell order"""
        try:
            if symbol not in self.positions:
                print(f"No position to sell for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            if position.quantity < quantity:
                print(f"Insufficient quantity to sell: {position.quantity} < {quantity}")
                quantity = position.quantity  # Sell all available
            
            total_proceeds = quantity * price
            fees = total_proceeds * self.commission
            net_proceeds = total_proceeds - fees
            
            # Update cash
            self.cash += net_proceeds
            
            # Update position
            if position.quantity == quantity:
                # Close entire position
                del self.positions[symbol]
            else:
                # Partial sell
                self.positions[symbol].quantity -= quantity
                self.positions[symbol].current_price = price
            
            # Record trade
            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                action='SELL',
                quantity=quantity,
                price=price,
                value=total_proceeds,
                fees=fees,
                signal_strength=signal_strength,
                strategy=strategy
            )
            self.trades.append(trade)
            
            return True
            
        except Exception as e:
            print(f"Error executing sell order: {e}")
            return False
    
    def update_portfolio_value(self, timestamp: datetime, prices: Dict[str, float]):
        """Update portfolio value with current market prices"""
        try:
            # Update position current prices and calculate unrealized P&L
            total_positions_value = 0.0
            
            for symbol, position in self.positions.items():
                if symbol in prices:
                    position.current_price = prices[symbol]
                    position_value = position.quantity * position.current_price
                    position.unrealized_pnl = position_value - (position.quantity * position.entry_price)
                    total_positions_value += position_value
            
            # Calculate total portfolio value
            self.portfolio_value = self.cash + total_positions_value
            
            # Record portfolio state
            portfolio_record = {
                'timestamp': timestamp,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions_value': total_positions_value,
                'positions': copy.deepcopy(self.positions)
            }
            self.portfolio_history.append(portfolio_record)
            
            # Calculate daily return if we have previous data
            if len(self.portfolio_history) > 1:
                prev_value = self.portfolio_history[-2]['portfolio_value']
                daily_return = (self.portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
            
        except Exception as e:
            print(f"Error updating portfolio value: {e}")
    
    def run_backtest(self, data: pd.DataFrame, strategy, start_date: str = None, 
                    end_date: str = None, rebalance_frequency: str = 'daily') -> Dict[str, Any]:
        """Run complete backtest on historical data"""
        try:
            self.reset()
            
            # Filter data by date range if specified
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            if data.empty:
                return {'error': 'No data in specified date range'}
            
            print(f"Running backtest on {len(data)} data points from {data.index[0]} to {data.index[-1]}")
            
            # Ensure strategy has minimum data for training
            training_period = max(100, len(data) // 4)  # Use 25% for training, minimum 100 points
            
            for i in range(training_period, len(data)):
                try:
                    current_time = data.index[i]
                    current_price = data.iloc[i]['close']
                    
                    # Get historical data up to current point
                    historical_data = data.iloc[:i+1]
                    
                    # Generate signal from strategy
                    signal = strategy.generate_signal(historical_data)
                    
                    if 'error' in signal:
                        continue
                    
                    signal_type = signal.get('signal', 'HOLD')
                    confidence = signal.get('confidence', 0.0)
                    strength = signal.get('strength', 0.0)
                    
                    # Execute trades based on signal
                    if confidence > 0.6 and strength > 0.3:  # Minimum thresholds
                        symbol = 'BACKTEST'  # Generic symbol for backtesting
                        
                        if signal_type == 'BUY':
                            # Calculate position size (risk-based)
                            position_size = self._calculate_position_size(current_price, strength)
                            
                            if position_size > 0:
                                self.execute_trade(
                                    current_time, symbol, 'BUY', position_size, 
                                    current_price, strength, strategy.name
                                )
                        
                        elif signal_type == 'SELL' and symbol in self.positions:
                            # Sell existing position
                            position_quantity = self.positions[symbol].quantity
                            sell_quantity = position_quantity * strength  # Partial or full sell
                            
                            if sell_quantity > 0:
                                self.execute_trade(
                                    current_time, symbol, 'SELL', sell_quantity, 
                                    current_price, strength, strategy.name
                                )
                    
                    # Update portfolio value
                    prices = {'BACKTEST': current_price}
                    self.update_portfolio_value(current_time, prices)
                    
                except Exception as e:
                    print(f"Error processing data point {i}: {e}")
                    continue
            
            # Calculate final metrics
            self.calculate_performance_metrics()
            
            return self.get_backtest_results()
            
        except Exception as e:
            print(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def _calculate_position_size(self, price: float, signal_strength: float, 
                               max_position_pct: float = 0.1) -> float:
        """Calculate position size based on available cash and risk parameters"""
        try:
            # Maximum position value based on portfolio percentage
            max_position_value = self.portfolio_value * max_position_pct * signal_strength
            
            # Maximum shares we can afford
            max_affordable_shares = (self.cash * 0.95) / price  # Leave 5% cash buffer
            
            # Position size based on risk
            risk_adjusted_value = min(max_position_value, self.cash * 0.95)
            risk_adjusted_shares = risk_adjusted_value / price
            
            # Return the minimum of affordable and risk-adjusted position
            position_size = min(max_affordable_shares, risk_adjusted_shares)
            
            return max(0.0, position_size)
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        try:
            if len(self.portfolio_history) < 2:
                self.metrics = {'error': 'Insufficient data for metrics calculation'}
                return
            
            # Basic metrics
            initial_value = self.initial_capital
            final_value = self.portfolio_value
            total_return = (final_value - initial_value) / initial_value
            
            # Time-based metrics
            start_date = self.portfolio_history[0]['timestamp']
            end_date = self.portfolio_history[-1]['timestamp']
            total_days = (end_date - start_date).days
            
            # Returns analysis
            returns_array = np.array(self.daily_returns) if self.daily_returns else np.array([0])
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if len(returns_array) > 1 and np.std(returns_array) > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Maximum drawdown
            portfolio_values = [p['portfolio_value'] for p in self.portfolio_history]
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            # Win rate
            winning_trades = [t for t in self.trades if t.action == 'SELL']
            if winning_trades and len(winning_trades) > 0:
                # Calculate P&L for each sell trade
                wins = 0
                total_sell_trades = 0
                
                for sell_trade in winning_trades:
                    # Find corresponding buy trade
                    buy_trades = [t for t in self.trades 
                                if t.symbol == sell_trade.symbol and 
                                t.action == 'BUY' and 
                                t.timestamp < sell_trade.timestamp]
                    
                    if buy_trades:
                        # Use most recent buy trade
                        buy_trade = buy_trades[-1]
                        pnl = (sell_trade.price - buy_trade.price) * sell_trade.quantity
                        if pnl > 0:
                            wins += 1
                        total_sell_trades += 1
                
                win_rate = wins / total_sell_trades if total_sell_trades > 0 else 0
            else:
                win_rate = 0
            
            # Volatility
            volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0
            
            # Trade statistics
            total_trades = len(self.trades)
            buy_trades = len([t for t in self.trades if t.action == 'BUY'])
            sell_trades = len([t for t in self.trades if t.action == 'SELL'])
            
            # Fees paid
            total_fees = sum(t.fees for t in self.trades)
            
            self.metrics = {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'annualized_return': total_return * (365 / total_days) if total_days > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'total_fees': total_fees,
                'avg_return': np.mean(returns_array) if len(returns_array) > 0 else 0,
                'total_days': total_days,
                'calmar_ratio': total_return / max_drawdown if max_drawdown > 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            self.metrics = {'error': str(e)}
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        try:
            if len(portfolio_values) < 2:
                return 0.0
            
            values = np.array(portfolio_values)
            running_max = np.maximum.accumulate(values)
            drawdown = (running_max - values) / running_max
            
            # Store drawdown history
            self.drawdown_history = drawdown.tolist()
            
            return np.max(drawdown)
            
        except Exception as e:
            print(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def get_backtest_results(self) -> Dict[str, Any]:
        """Get comprehensive backtest results"""
        try:
            return {
                'performance_metrics': self.metrics,
                'portfolio_history': self.portfolio_history,
                'trades': [
                    {
                        'timestamp': t.timestamp.isoformat() if hasattr(t.timestamp, 'isoformat') else str(t.timestamp),
                        'symbol': t.symbol,
                        'action': t.action,
                        'quantity': t.quantity,
                        'price': t.price,
                        'value': t.value,
                        'fees': t.fees,
                        'signal_strength': t.signal_strength,
                        'strategy': t.strategy
                    }
                    for t in self.trades
                ],
                'final_positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'entry_time': pos.entry_time.isoformat() if hasattr(pos.entry_time, 'isoformat') else str(pos.entry_time)
                    }
                    for symbol, pos in self.positions.items()
                },
                'drawdown_history': self.drawdown_history,
                'daily_returns': self.daily_returns
            }
            
        except Exception as e:
            print(f"Error getting backtest results: {e}")
            return {'error': str(e)}
    
    def compare_strategies(self, data: pd.DataFrame, strategies: List[Any], 
                          start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Compare multiple strategies"""
        try:
            results = {}
            
            for strategy in strategies:
                print(f"Backtesting strategy: {strategy.name}")
                strategy_result = self.run_backtest(data, strategy, start_date, end_date)
                results[strategy.name] = strategy_result
                
                # Reset for next strategy
                self.reset()
            
            # Calculate comparison metrics
            comparison = self._generate_strategy_comparison(results)
            
            return {
                'individual_results': results,
                'comparison': comparison
            }
            
        except Exception as e:
            print(f"Error comparing strategies: {e}")
            return {'error': str(e)}
    
    def _generate_strategy_comparison(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate strategy comparison summary"""
        try:
            comparison = {
                'strategy_rankings': {},
                'best_strategy': None,
                'summary_table': []
            }
            
            # Extract key metrics for comparison
            strategy_metrics = {}
            
            for strategy_name, result in results.items():
                if 'error' not in result and 'performance_metrics' in result:
                    metrics = result['performance_metrics']
                    if 'error' not in metrics:
                        strategy_metrics[strategy_name] = metrics
            
            if not strategy_metrics:
                return {'error': 'No valid strategy results for comparison'}
            
            # Rank strategies by total return
            sorted_strategies = sorted(
                strategy_metrics.items(), 
                key=lambda x: x[1].get('total_return', 0), 
                reverse=True
            )
            
            comparison['best_strategy'] = sorted_strategies[0][0] if sorted_strategies else None
            
            # Create summary table
            for strategy_name, metrics in strategy_metrics.items():
                comparison['summary_table'].append({
                    'strategy': strategy_name,
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'volatility': metrics.get('volatility', 0)
                })
            
            # Rankings for different metrics
            comparison['strategy_rankings'] = {
                'total_return': [s[0] for s in sorted_strategies],
                'sharpe_ratio': sorted(
                    strategy_metrics.keys(), 
                    key=lambda x: strategy_metrics[x].get('sharpe_ratio', 0), 
                    reverse=True
                ),
                'max_drawdown': sorted(
                    strategy_metrics.keys(), 
                    key=lambda x: strategy_metrics[x].get('max_drawdown', float('inf'))
                )
            }
            
            return comparison
            
        except Exception as e:
            print(f"Error generating strategy comparison: {e}")
            return {'error': str(e)}
