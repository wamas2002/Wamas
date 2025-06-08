import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestResult:
    """Backtest result container"""
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

class WalkForwardAnalyzer:
    """Walk-forward analysis for strategy optimization and validation"""
    
    def __init__(self, in_sample_periods: int = 252, out_sample_periods: int = 63, 
                 step_size: int = 21, min_trades: int = 10):
        self.in_sample_periods = in_sample_periods  # 1 year training
        self.out_sample_periods = out_sample_periods  # 3 months testing
        self.step_size = step_size  # 1 month step
        self.min_trades = min_trades
        
    def run_walk_forward(self, data: pd.DataFrame, strategy_func, 
                        param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Run walk-forward analysis with parameter optimization"""
        try:
            if len(data) < self.in_sample_periods + self.out_sample_periods:
                return {'error': 'Insufficient data for walk-forward analysis'}
            
            walk_forward_results = []
            optimal_params_history = []
            
            # Create walk-forward windows
            start_idx = 0
            while start_idx + self.in_sample_periods + self.out_sample_periods <= len(data):
                # Define in-sample and out-of-sample periods
                in_sample_end = start_idx + self.in_sample_periods
                out_sample_end = in_sample_end + self.out_sample_periods
                
                in_sample_data = data.iloc[start_idx:in_sample_end]
                out_sample_data = data.iloc[in_sample_end:out_sample_end]
                
                # Optimize parameters on in-sample data
                optimal_params = self._optimize_parameters(
                    in_sample_data, strategy_func, param_ranges
                )
                
                # Test on out-of-sample data
                out_sample_result = strategy_func(out_sample_data, **optimal_params)
                
                walk_forward_results.append({
                    'period_start': data.index[in_sample_end],
                    'period_end': data.index[out_sample_end - 1],
                    'optimal_params': optimal_params,
                    'out_sample_return': out_sample_result.get('total_return', 0),
                    'out_sample_sharpe': out_sample_result.get('sharpe_ratio', 0),
                    'out_sample_max_dd': out_sample_result.get('max_drawdown', 0),
                    'trades_count': out_sample_result.get('total_trades', 0)
                })
                
                optimal_params_history.append(optimal_params)
                start_idx += self.step_size
            
            # Aggregate results
            aggregate_stats = self._aggregate_walk_forward_results(walk_forward_results)
            
            return {
                'walk_forward_results': walk_forward_results,
                'aggregate_performance': aggregate_stats,
                'parameter_stability': self._analyze_parameter_stability(optimal_params_history),
                'total_periods': len(walk_forward_results)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _optimize_parameters(self, data: pd.DataFrame, strategy_func, 
                           param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search"""
        try:
            best_params = {}
            best_score = -float('inf')
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(param_ranges)
            
            for params in param_combinations:
                try:
                    result = strategy_func(data, **params)
                    
                    # Calculate optimization score (Sharpe ratio with drawdown penalty)
                    sharpe = result.get('sharpe_ratio', 0)
                    max_dd = result.get('max_drawdown', 0)
                    trades = result.get('total_trades', 0)
                    
                    # Penalize strategies with too few trades or excessive drawdown
                    if trades < self.min_trades:
                        score = -1
                    elif max_dd > 0.3:  # 30% max drawdown limit
                        score = sharpe * 0.5  # Heavy penalty
                    else:
                        score = sharpe - (max_dd * 2)  # Drawdown penalty
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        
                except Exception as e:
                    continue  # Skip invalid parameter combinations
            
            return best_params
            
        except Exception as e:
            print(f"Error in parameter optimization: {e}")
            return {}
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search"""
        try:
            import itertools
            
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())
            
            combinations = []
            for combo in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combo))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            print(f"Error generating parameter combinations: {e}")
            return [{}]
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate walk-forward results"""
        try:
            if not results:
                return {}
            
            returns = [r['out_sample_return'] for r in results]
            sharpes = [r['out_sample_sharpe'] for r in results]
            max_dds = [r['out_sample_max_dd'] for r in results]
            trades = [r['trades_count'] for r in results]
            
            return {
                'avg_return': np.mean(returns),
                'return_volatility': np.std(returns),
                'avg_sharpe': np.mean(sharpes),
                'avg_max_drawdown': np.mean(max_dds),
                'total_trades': sum(trades),
                'win_rate': len([r for r in returns if r > 0]) / len(returns),
                'consistency_score': len([s for s in sharpes if s > 0]) / len(sharpes)
            }
            
        except Exception as e:
            print(f"Error aggregating results: {e}")
            return {}
    
    def _analyze_parameter_stability(self, params_history: List[Dict]) -> Dict[str, Any]:
        """Analyze parameter stability across periods"""
        try:
            if not params_history:
                return {}
            
            stability_analysis = {}
            
            for param_name in params_history[0].keys():
                values = [params[param_name] for params in params_history if param_name in params]
                
                if values:
                    stability_analysis[param_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'stability_score': 1 / (1 + np.std(values))  # Higher = more stable
                    }
            
            return stability_analysis
            
        except Exception as e:
            print(f"Error analyzing parameter stability: {e}")
            return {}

class BacktestingEngine:
    """Advanced backtesting engine with comprehensive analysis"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = 0.0005  # 0.05% slippage
        
        # Transaction tracking
        self.trades = []
        self.portfolio_history = []
        self.positions = {}
        
        # Performance metrics
        self.daily_returns = []
        self.benchmark_returns = []
        
    def run_backtest(self, data: pd.DataFrame, strategy, 
                    benchmark_symbol: str = None) -> BacktestResult:
        """Run comprehensive backtest"""
        try:
            # Initialize portfolio
            current_capital = self.initial_capital
            positions = {}
            portfolio_values = [self.initial_capital]
            
            # Track daily portfolio values
            for i in range(1, len(data)):
                current_data = data.iloc[:i+1]
                
                # Generate signals
                signals = strategy.generate_signal(current_data)
                
                if not isinstance(signals, dict):
                    continue
                
                signal_action = signals.get('signal', 'HOLD')
                signal_strength = signals.get('strength', 0)
                current_price = data['close'].iloc[i]
                
                # Execute trades based on signals
                trade_result = self._execute_signal(
                    signal_action, signal_strength, current_price, 
                    current_capital, positions, data.index[i]
                )
                
                if trade_result:
                    current_capital = trade_result['new_capital']
                    positions = trade_result['positions']
                    
                    if trade_result.get('trade_executed'):
                        self.trades.append(trade_result['trade_details'])
                
                # Calculate portfolio value
                portfolio_value = self._calculate_portfolio_value(
                    current_capital, positions, current_price
                )
                portfolio_values.append(portfolio_value)
                
                # Track daily returns
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                    self.daily_returns.append(daily_return)
            
            # Calculate performance metrics
            result = self._calculate_performance_metrics(
                portfolio_values, self.trades, data
            )
            
            # Store portfolio history
            self.portfolio_history = portfolio_values
            
            return result
            
        except Exception as e:
            print(f"Error in backtesting: {e}")
            return BacktestResult()
    
    def _execute_signal(self, signal: str, strength: float, price: float,
                       capital: float, positions: Dict, timestamp) -> Optional[Dict]:
        """Execute trading signal with realistic constraints"""
        try:
            position_size = 0
            current_position = positions.get('size', 0)
            
            # Determine position size based on signal
            if signal == 'BUY' and current_position <= 0:
                position_size = self._calculate_position_size(strength, capital, price)
            elif signal == 'SELL' and current_position >= 0:
                position_size = -self._calculate_position_size(strength, capital, price)
            elif signal == 'STRONG_BUY' and current_position <= 0:
                position_size = self._calculate_position_size(strength * 1.5, capital, price)
            elif signal == 'STRONG_SELL' and current_position >= 0:
                position_size = -self._calculate_position_size(strength * 1.5, capital, price)
            
            if abs(position_size) < 0.01:  # Minimum position size
                return None
            
            # Apply slippage and commission
            execution_price = price * (1 + self.slippage if position_size > 0 else 1 - self.slippage)
            commission_cost = abs(position_size * execution_price * self.commission)
            
            # Check if we have enough capital
            required_capital = abs(position_size * execution_price) + commission_cost
            if required_capital > capital:
                return None
            
            # Execute trade
            new_capital = capital - (position_size * execution_price) - commission_cost
            new_positions = {
                'size': current_position + position_size,
                'avg_price': execution_price,
                'timestamp': timestamp
            }
            
            # Record trade details
            trade_details = {
                'timestamp': timestamp,
                'signal': signal,
                'size': position_size,
                'price': execution_price,
                'commission': commission_cost,
                'capital_after': new_capital,
                'strength': strength
            }
            
            return {
                'new_capital': new_capital,
                'positions': new_positions,
                'trade_executed': True,
                'trade_details': trade_details
            }
            
        except Exception as e:
            print(f"Error executing signal: {e}")
            return None
    
    def _calculate_position_size(self, strength: float, capital: float, price: float) -> float:
        """Calculate position size based on signal strength and risk management"""
        try:
            # Base position size (percentage of capital)
            base_size_pct = 0.1  # 10% base allocation
            
            # Adjust based on signal strength
            strength_multiplier = min(2.0, max(0.5, strength))
            position_size_pct = base_size_pct * strength_multiplier
            
            # Calculate actual position size
            position_value = capital * position_size_pct
            position_size = position_value / price
            
            return position_size
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.0
    
    def _calculate_portfolio_value(self, cash: float, positions: Dict, current_price: float) -> float:
        """Calculate total portfolio value"""
        try:
            position_value = 0
            if positions and positions.get('size', 0) != 0:
                position_value = positions['size'] * current_price
            
            return cash + position_value
            
        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return cash
    
    def _calculate_performance_metrics(self, portfolio_values: List[float], 
                                     trades: List[Dict], data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        try:
            if len(portfolio_values) < 2:
                return BacktestResult()
            
            # Basic returns
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # Annual return
            days = len(portfolio_values)
            annual_return = (final_value / initial_value) ** (252 / days) - 1
            
            # Daily returns
            returns = []
            for i in range(1, len(portfolio_values)):
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(ret)
            
            returns = np.array(returns)
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% annual
            excess_returns = returns - (risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown)
            
            # Calmar ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            # Trade statistics
            if trades:
                trade_returns = []
                for trade in trades:
                    # Simplified trade return calculation
                    trade_returns.append(trade.get('strength', 0) * 0.01)  # Approximate
                
                win_trades = [t for t in trade_returns if t > 0]
                lose_trades = [t for t in trade_returns if t <= 0]
                
                win_rate = len(win_trades) / len(trades) if trades else 0
                avg_trade_return = np.mean(trade_returns) if trade_returns else 0
                best_trade = max(trade_returns) if trade_returns else 0
                worst_trade = min(trade_returns) if trade_returns else 0
                
                # Profit factor
                gross_profit = sum(win_trades) if win_trades else 0
                gross_loss = abs(sum(lose_trades)) if lose_trades else 0.01
                profit_factor = gross_profit / gross_loss
            else:
                win_rate = 0
                avg_trade_return = 0
                best_trade = 0
                worst_trade = 0
                profit_factor = 0
            
            return BacktestResult(
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(trades),
                avg_trade_return=avg_trade_return,
                best_trade=best_trade,
                worst_trade=worst_trade
            )
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return BacktestResult()
    
    def run_monte_carlo_simulation(self, strategy, data: pd.DataFrame, 
                                  num_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for strategy robustness testing"""
        try:
            simulation_results = []
            
            for _ in range(num_simulations):
                # Bootstrap resample the data
                sample_data = data.sample(n=len(data), replace=True).sort_index()
                
                # Run backtest on resampled data
                result = self.run_backtest(sample_data, strategy)
                
                simulation_results.append({
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate
                })
            
            # Analyze simulation results
            returns = [r['total_return'] for r in simulation_results]
            sharpes = [r['sharpe_ratio'] for r in simulation_results]
            max_dds = [r['max_drawdown'] for r in simulation_results]
            win_rates = [r['win_rate'] for r in simulation_results]
            
            return {
                'simulations_run': num_simulations,
                'return_statistics': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'percentile_5': np.percentile(returns, 5),
                    'percentile_95': np.percentile(returns, 95),
                    'positive_returns': len([r for r in returns if r > 0]) / len(returns)
                },
                'sharpe_statistics': {
                    'mean': np.mean(sharpes),
                    'std': np.std(sharpes),
                    'percentile_5': np.percentile(sharpes, 5),
                    'percentile_95': np.percentile(sharpes, 95)
                },
                'drawdown_statistics': {
                    'mean': np.mean(max_dds),
                    'std': np.std(max_dds),
                    'percentile_95': np.percentile(max_dds, 95),
                    'percentile_99': np.percentile(max_dds, 99)
                },
                'win_rate_statistics': {
                    'mean': np.mean(win_rates),
                    'std': np.std(win_rates),
                    'percentile_5': np.percentile(win_rates, 5),
                    'percentile_95': np.percentile(win_rates, 95)
                }
            }
            
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {e}")
            return {'error': str(e)}
    
    def compare_strategies(self, strategies: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Compare multiple strategies"""
        try:
            strategy_results = {}
            
            for strategy_name, strategy in strategies.items():
                result = self.run_backtest(data, strategy)
                strategy_results[strategy_name] = {
                    'total_return': result.total_return,
                    'annual_return': result.annual_return,
                    'volatility': result.volatility,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades
                }
            
            # Rank strategies
            ranking_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
            strategy_rankings = {}
            
            for metric in ranking_metrics:
                if metric == 'max_drawdown':
                    # Lower is better for drawdown
                    sorted_strategies = sorted(
                        strategy_results.items(),
                        key=lambda x: x[1][metric]
                    )
                else:
                    # Higher is better
                    sorted_strategies = sorted(
                        strategy_results.items(),
                        key=lambda x: x[1][metric],
                        reverse=True
                    )
                
                strategy_rankings[metric] = [s[0] for s in sorted_strategies]
            
            return {
                'strategy_results': strategy_results,
                'rankings': strategy_rankings,
                'best_overall': strategy_rankings['sharpe_ratio'][0],
                'comparison_period': f"{data.index[0]} to {data.index[-1]}"
            }
            
        except Exception as e:
            print(f"Error comparing strategies: {e}")
            return {'error': str(e)}
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed backtest analysis"""
        try:
            if not self.trades or not self.portfolio_history:
                return {'error': 'No backtest data available'}
            
            # Trade analysis
            trade_analysis = self._analyze_trades()
            
            # Portfolio analysis
            portfolio_analysis = self._analyze_portfolio()
            
            # Risk analysis
            risk_analysis = self._analyze_risk()
            
            return {
                'trade_analysis': trade_analysis,
                'portfolio_analysis': portfolio_analysis,
                'risk_analysis': risk_analysis,
                'summary': {
                    'total_trades': len(self.trades),
                    'final_portfolio_value': self.portfolio_history[-1],
                    'total_return_pct': ((self.portfolio_history[-1] / self.initial_capital) - 1) * 100
                }
            }
            
        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze individual trades"""
        try:
            if not self.trades:
                return {}
            
            # Trade frequency
            trade_dates = [trade['timestamp'] for trade in self.trades]
            if len(trade_dates) > 1:
                avg_days_between_trades = np.mean([
                    (trade_dates[i] - trade_dates[i-1]).days 
                    for i in range(1, len(trade_dates))
                ])
            else:
                avg_days_between_trades = 0
            
            # Commission analysis
            total_commissions = sum(trade.get('commission', 0) for trade in self.trades)
            
            return {
                'total_trades': len(self.trades),
                'avg_days_between_trades': avg_days_between_trades,
                'total_commissions': total_commissions,
                'commission_pct_of_capital': total_commissions / self.initial_capital * 100
            }
            
        except Exception as e:
            print(f"Error analyzing trades: {e}")
            return {}
    
    def _analyze_portfolio(self) -> Dict[str, Any]:
        """Analyze portfolio performance"""
        try:
            if len(self.portfolio_history) < 2:
                return {}
            
            values = np.array(self.portfolio_history)
            
            # Underwater curve (drawdown over time)
            peak = np.maximum.accumulate(values)
            underwater = (values - peak) / peak
            
            # Recovery analysis
            recovery_periods = []
            in_drawdown = False
            drawdown_start = 0
            
            for i, dd in enumerate(underwater):
                if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                    in_drawdown = True
                    drawdown_start = i
                elif dd >= 0 and in_drawdown:  # Recovery
                    recovery_periods.append(i - drawdown_start)
                    in_drawdown = False
            
            avg_recovery_periods = np.mean(recovery_periods) if recovery_periods else 0
            
            return {
                'max_underwater': np.min(underwater) * 100,
                'avg_recovery_periods': avg_recovery_periods,
                'current_underwater': underwater[-1] * 100,
                'recovery_periods_count': len(recovery_periods)
            }
            
        except Exception as e:
            print(f"Error analyzing portfolio: {e}")
            return {}
    
    def _analyze_risk(self) -> Dict[str, Any]:
        """Analyze risk metrics"""
        try:
            if len(self.daily_returns) < 30:
                return {}
            
            returns = np.array(self.daily_returns)
            
            # Value at Risk
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            
            # Expected Shortfall (Conditional VaR)
            es_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * 100
            es_99 = np.mean(returns[returns <= np.percentile(returns, 1)]) * 100
            
            # Skewness and Kurtosis
            skewness = self._calculate_skewness(returns)
            kurtosis = self._calculate_kurtosis(returns)
            
            return {
                'var_95_daily': var_95,
                'var_99_daily': var_99,
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'tail_ratio': np.percentile(returns, 95) / abs(np.percentile(returns, 5))
            }
            
        except Exception as e:
            print(f"Error analyzing risk: {e}")
            return {}
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            return np.mean(((returns - mean_return) / std_return) ** 3)
            
        except:
            return 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            return np.mean(((returns - mean_return) / std_return) ** 4) - 3
            
        except:
            return 0.0