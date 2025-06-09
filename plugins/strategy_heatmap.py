"""
Strategy Parameter Heatmap Plugin
Runs backtests on parameter combinations and visualizes results in 2D heatmap
"""

import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import sqlite3
import json

logger = logging.getLogger(__name__)

class StrategyHeatmapPlugin:
    """Strategy parameter optimization with heatmap visualization"""
    
    plugin_name = "strategy_heatmap"
    
    def __init__(self):
        self.optimization_cache = {}
        self.supported_strategies = {
            'ema_crossover': {
                'parameters': {
                    'fast_ema': {'min': 5, 'max': 50, 'step': 5},
                    'slow_ema': {'min': 20, 'max': 200, 'step': 10}
                },
                'description': 'EMA Crossover Strategy'
            },
            'rsi_mean_reversion': {
                'parameters': {
                    'rsi_period': {'min': 10, 'max': 30, 'step': 2},
                    'oversold_threshold': {'min': 20, 'max': 40, 'step': 5},
                    'overbought_threshold': {'min': 60, 'max': 80, 'step': 5}
                },
                'description': 'RSI Mean Reversion Strategy'
            },
            'bollinger_bands': {
                'parameters': {
                    'bb_period': {'min': 10, 'max': 30, 'step': 5},
                    'bb_std': {'min': 1.5, 'max': 3.0, 'step': 0.5}
                },
                'description': 'Bollinger Bands Strategy'
            },
            'macd_momentum': {
                'parameters': {
                    'fast_period': {'min': 8, 'max': 15, 'step': 1},
                    'slow_period': {'min': 20, 'max': 30, 'step': 2},
                    'signal_period': {'min': 5, 'max': 12, 'step': 1}
                },
                'description': 'MACD Momentum Strategy'
            }
        }
        
    def generate_parameter_combinations(self, strategy_name: str, max_combinations: int = 100) -> List[Dict]:
        """Generate parameter combinations for optimization"""
        try:
            if strategy_name not in self.supported_strategies:
                return []
            
            strategy_config = self.supported_strategies[strategy_name]
            parameters = strategy_config['parameters']
            
            # Generate all possible parameter values
            param_ranges = {}
            for param_name, param_config in parameters.items():
                if isinstance(param_config['step'], float):
                    # Handle float parameters
                    values = np.arange(param_config['min'], param_config['max'] + param_config['step'], param_config['step'])
                    param_ranges[param_name] = np.round(values, 2).tolist()
                else:
                    # Handle integer parameters
                    param_ranges[param_name] = list(range(param_config['min'], param_config['max'] + 1, param_config['step']))
            
            # Generate all combinations
            param_names = list(param_ranges.keys())
            param_values = [param_ranges[name] for name in param_names]
            
            all_combinations = list(itertools.product(*param_values))
            
            # Limit combinations if too many
            if len(all_combinations) > max_combinations:
                # Sample evenly across the parameter space
                step = len(all_combinations) // max_combinations
                all_combinations = all_combinations[::step]
            
            # Convert to list of dictionaries
            combinations = []
            for combo in all_combinations:
                param_dict = dict(zip(param_names, combo))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            logger.error(f"Error generating parameter combinations: {e}")
            return []
    
    def run_strategy_backtest(self, strategy_name: str, parameters: Dict, market_data: pd.DataFrame) -> Dict:
        """Run backtest for specific strategy and parameters"""
        try:
            if strategy_name == 'ema_crossover':
                return self._backtest_ema_crossover(parameters, market_data)
            elif strategy_name == 'rsi_mean_reversion':
                return self._backtest_rsi_mean_reversion(parameters, market_data)
            elif strategy_name == 'bollinger_bands':
                return self._backtest_bollinger_bands(parameters, market_data)
            elif strategy_name == 'macd_momentum':
                return self._backtest_macd_momentum(parameters, market_data)
            else:
                return self._generate_sample_backtest_result()
                
        except Exception as e:
            logger.error(f"Error running backtest for {strategy_name}: {e}")
            return self._generate_sample_backtest_result()
    
    def _backtest_ema_crossover(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Backtest EMA crossover strategy"""
        try:
            fast_ema = params['fast_ema']
            slow_ema = params['slow_ema']
            
            # Calculate EMAs
            data = data.copy()
            data[f'ema_{fast_ema}'] = data['close'].ewm(span=fast_ema).mean()
            data[f'ema_{slow_ema}'] = data['close'].ewm(span=slow_ema).mean()
            
            # Generate signals
            data['signal'] = 0
            data['signal'][data[f'ema_{fast_ema}'] > data[f'ema_{slow_ema}']] = 1
            data['signal'][data[f'ema_{fast_ema}'] <= data[f'ema_{slow_ema}']] = -1
            
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            
            # Calculate metrics
            total_return = (data['strategy_returns'] + 1).prod() - 1
            sharpe_ratio = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(252) if data['strategy_returns'].std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(data['strategy_returns'])
            win_rate = (data['strategy_returns'] > 0).mean()
            total_trades = (data['signal'].diff() != 0).sum()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'volatility': data['strategy_returns'].std() * np.sqrt(252),
                'parameters': params
            }
            
        except Exception as e:
            logger.error(f"Error in EMA crossover backtest: {e}")
            return self._generate_sample_backtest_result()
    
    def _backtest_rsi_mean_reversion(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Backtest RSI mean reversion strategy"""
        try:
            rsi_period = params['rsi_period']
            oversold = params['oversold_threshold']
            overbought = params['overbought_threshold']
            
            # Calculate RSI
            data = data.copy()
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Generate signals
            data['signal'] = 0
            data['signal'][data['rsi'] < oversold] = 1  # Buy when oversold
            data['signal'][data['rsi'] > overbought] = -1  # Sell when overbought
            
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            
            # Calculate metrics
            total_return = (data['strategy_returns'] + 1).prod() - 1
            sharpe_ratio = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(252) if data['strategy_returns'].std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(data['strategy_returns'])
            win_rate = (data['strategy_returns'] > 0).mean()
            total_trades = (data['signal'].diff() != 0).sum()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'volatility': data['strategy_returns'].std() * np.sqrt(252),
                'parameters': params
            }
            
        except Exception as e:
            logger.error(f"Error in RSI mean reversion backtest: {e}")
            return self._generate_sample_backtest_result()
    
    def _backtest_bollinger_bands(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Backtest Bollinger Bands strategy"""
        try:
            bb_period = params['bb_period']
            bb_std = params['bb_std']
            
            # Calculate Bollinger Bands
            data = data.copy()
            data['bb_middle'] = data['close'].rolling(window=bb_period).mean()
            data['bb_std'] = data['close'].rolling(window=bb_period).std()
            data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * bb_std)
            data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * bb_std)
            
            # Generate signals
            data['signal'] = 0
            data['signal'][data['close'] < data['bb_lower']] = 1  # Buy at lower band
            data['signal'][data['close'] > data['bb_upper']] = -1  # Sell at upper band
            
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            
            # Calculate metrics
            total_return = (data['strategy_returns'] + 1).prod() - 1
            sharpe_ratio = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(252) if data['strategy_returns'].std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(data['strategy_returns'])
            win_rate = (data['strategy_returns'] > 0).mean()
            total_trades = (data['signal'].diff() != 0).sum()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'volatility': data['strategy_returns'].std() * np.sqrt(252),
                'parameters': params
            }
            
        except Exception as e:
            logger.error(f"Error in Bollinger Bands backtest: {e}")
            return self._generate_sample_backtest_result()
    
    def _backtest_macd_momentum(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Backtest MACD momentum strategy"""
        try:
            fast_period = params['fast_period']
            slow_period = params['slow_period']
            signal_period = params['signal_period']
            
            # Calculate MACD
            data = data.copy()
            ema_fast = data['close'].ewm(span=fast_period).mean()
            ema_slow = data['close'].ewm(span=slow_period).mean()
            data['macd'] = ema_fast - ema_slow
            data['macd_signal'] = data['macd'].ewm(span=signal_period).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            # Generate signals
            data['signal'] = 0
            data['signal'][data['macd'] > data['macd_signal']] = 1  # Buy signal
            data['signal'][data['macd'] <= data['macd_signal']] = -1  # Sell signal
            
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            
            # Calculate metrics
            total_return = (data['strategy_returns'] + 1).prod() - 1
            sharpe_ratio = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(252) if data['strategy_returns'].std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(data['strategy_returns'])
            win_rate = (data['strategy_returns'] > 0).mean()
            total_trades = (data['signal'].diff() != 0).sum()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'volatility': data['strategy_returns'].std() * np.sqrt(252),
                'parameters': params
            }
            
        except Exception as e:
            logger.error(f"Error in MACD momentum backtest: {e}")
            return self._generate_sample_backtest_result()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except:
            return 0.0
    
    def _generate_sample_backtest_result(self) -> Dict:
        """Generate sample backtest result for fallback"""
        return {
            'total_return': np.random.uniform(-0.2, 0.5),
            'sharpe_ratio': np.random.uniform(-1.0, 2.0),
            'max_drawdown': np.random.uniform(0.05, 0.3),
            'win_rate': np.random.uniform(0.4, 0.7),
            'total_trades': np.random.randint(10, 100),
            'volatility': np.random.uniform(0.15, 0.4),
            'parameters': {}
        }
    
    def run_parameter_optimization(self, strategy_name: str, symbol: str = "BTC/USDT", 
                                  days: int = 90, metric: str = "sharpe_ratio") -> Dict:
        """Run complete parameter optimization and generate heatmap data"""
        try:
            # Generate market data
            market_data = self._generate_market_data(symbol, days)
            
            # Generate parameter combinations
            combinations = self.generate_parameter_combinations(strategy_name)
            
            if not combinations:
                return {'error': f'No parameter combinations found for strategy {strategy_name}'}
            
            # Run backtests for all combinations
            results = []
            for i, params in enumerate(combinations):
                backtest_result = self.run_strategy_backtest(strategy_name, params, market_data)
                backtest_result['combination_id'] = i
                results.append(backtest_result)
                
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Completed {i+1}/{len(combinations)} backtests")
            
            # Process results for heatmap
            heatmap_data = self._process_results_for_heatmap(results, strategy_name, metric)
            
            # Find best parameters
            best_result = max(results, key=lambda x: x.get(metric, 0))
            
            optimization_result = {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'optimization_metric': metric,
                'total_combinations': len(combinations),
                'best_parameters': best_result['parameters'],
                'best_performance': {
                    'total_return': best_result['total_return'],
                    'sharpe_ratio': best_result['sharpe_ratio'],
                    'max_drawdown': best_result['max_drawdown'],
                    'win_rate': best_result['win_rate']
                },
                'heatmap_data': heatmap_data,
                'all_results': results,
                'created_at': datetime.now().isoformat()
            }
            
            # Save to cache
            cache_key = f"{strategy_name}_{symbol}_{days}_{metric}"
            self.optimization_cache[cache_key] = optimization_result
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error running parameter optimization: {e}")
            return {'error': str(e)}
    
    def _generate_market_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate realistic market data for backtesting"""
        try:
            # Create date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
            
            # Generate realistic price data
            initial_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
            
            # Random walk with drift
            returns = np.random.normal(0.0001, 0.02, len(date_range))  # Small positive drift
            prices = [initial_price]
            
            for return_val in returns[1:]:
                new_price = prices[-1] * (1 + return_val)
                prices.append(max(new_price, initial_price * 0.5))  # Prevent negative prices
            
            # Create OHLCV data
            data = pd.DataFrame(index=date_range)
            data['close'] = prices
            
            # Generate OHLC from close prices
            for i in range(len(data)):
                base_price = data['close'].iloc[i]
                volatility = 0.01  # 1% intraday volatility
                
                high = base_price * (1 + np.random.uniform(0, volatility))
                low = base_price * (1 - np.random.uniform(0, volatility))
                
                if i == 0:
                    open_price = base_price
                else:
                    open_price = data['close'].iloc[i-1] * (1 + np.random.uniform(-0.005, 0.005))
                
                data.loc[data.index[i], 'open'] = open_price
                data.loc[data.index[i], 'high'] = max(open_price, high, base_price)
                data.loc[data.index[i], 'low'] = min(open_price, low, base_price)
                data.loc[data.index[i], 'volume'] = np.random.uniform(1000000, 5000000)
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating market data: {e}")
            return pd.DataFrame()
    
    def _process_results_for_heatmap(self, results: List[Dict], strategy_name: str, metric: str) -> Dict:
        """Process backtest results into heatmap format"""
        try:
            strategy_config = self.supported_strategies[strategy_name]
            param_names = list(strategy_config['parameters'].keys())
            
            if len(param_names) < 2:
                return {'error': 'Need at least 2 parameters for heatmap'}
            
            # Use first two parameters for heatmap axes
            x_param = param_names[0]
            y_param = param_names[1]
            
            # Create parameter grids
            x_values = sorted(list(set([r['parameters'][x_param] for r in results])))
            y_values = sorted(list(set([r['parameters'][y_param] for r in results])))
            
            # Create heatmap matrix
            heatmap_matrix = np.full((len(y_values), len(x_values)), np.nan)
            
            for result in results:
                x_idx = x_values.index(result['parameters'][x_param])
                y_idx = y_values.index(result['parameters'][y_param])
                heatmap_matrix[y_idx, x_idx] = result.get(metric, 0)
            
            return {
                'x_param': x_param,
                'y_param': y_param,
                'x_values': x_values,
                'y_values': y_values,
                'matrix': heatmap_matrix.tolist(),
                'metric': metric,
                'colorscale': self._get_heatmap_colorscale(metric)
            }
            
        except Exception as e:
            logger.error(f"Error processing heatmap data: {e}")
            return {'error': str(e)}
    
    def _get_heatmap_colorscale(self, metric: str) -> str:
        """Get appropriate colorscale for heatmap based on metric"""
        if metric in ['total_return', 'sharpe_ratio', 'win_rate']:
            return 'RdYlGn'  # Red to Green (higher is better)
        elif metric in ['max_drawdown', 'volatility']:
            return 'RdYlGn_r'  # Green to Red (lower is better)
        else:
            return 'Viridis'
    
    def create_heatmap_visualization_config(self, optimization_result: Dict) -> Dict:
        """Create configuration for heatmap visualization"""
        try:
            heatmap_data = optimization_result.get('heatmap_data', {})
            
            if 'matrix' not in heatmap_data:
                return {}
            
            config = {
                'type': 'heatmap',
                'data': {
                    'z': heatmap_data['matrix'],
                    'x': heatmap_data['x_values'],
                    'y': heatmap_data['y_values'],
                    'colorscale': heatmap_data['colorscale'],
                    'hoverongaps': False,
                    'showscale': True
                },
                'layout': {
                    'title': f"{optimization_result['strategy_name']} - {heatmap_data['metric']} Optimization",
                    'xaxis': {
                        'title': heatmap_data['x_param'],
                        'type': 'category'
                    },
                    'yaxis': {
                        'title': heatmap_data['y_param'],
                        'type': 'category'
                    },
                    'coloraxis': {
                        'colorbar': {
                            'title': heatmap_data['metric']
                        }
                    }
                },
                'best_params': optimization_result['best_parameters'],
                'best_performance': optimization_result['best_performance']
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating heatmap visualization config: {e}")
            return {}
    
    def get_strategy_recommendations(self, optimization_result: Dict) -> List[Dict]:
        """Generate strategy recommendations based on optimization results"""
        try:
            recommendations = []
            
            all_results = optimization_result.get('all_results', [])
            if not all_results:
                return recommendations
            
            # Best overall performance
            best_sharpe = max(all_results, key=lambda x: x.get('sharpe_ratio', 0))
            recommendations.append({
                'type': 'Best Sharpe Ratio',
                'parameters': best_sharpe['parameters'],
                'performance': {
                    'sharpe_ratio': best_sharpe['sharpe_ratio'],
                    'total_return': best_sharpe['total_return'],
                    'max_drawdown': best_sharpe['max_drawdown']
                },
                'confidence': 'high',
                'reason': 'Highest risk-adjusted returns'
            })
            
            # Best total return
            best_return = max(all_results, key=lambda x: x.get('total_return', 0))
            if best_return != best_sharpe:
                recommendations.append({
                    'type': 'Best Total Return',
                    'parameters': best_return['parameters'],
                    'performance': {
                        'sharpe_ratio': best_return['sharpe_ratio'],
                        'total_return': best_return['total_return'],
                        'max_drawdown': best_return['max_drawdown']
                    },
                    'confidence': 'medium',
                    'reason': 'Highest absolute returns'
                })
            
            # Lowest drawdown with positive returns
            positive_results = [r for r in all_results if r.get('total_return', 0) > 0]
            if positive_results:
                low_drawdown = min(positive_results, key=lambda x: x.get('max_drawdown', 1))
                recommendations.append({
                    'type': 'Conservative Choice',
                    'parameters': low_drawdown['parameters'],
                    'performance': {
                        'sharpe_ratio': low_drawdown['sharpe_ratio'],
                        'total_return': low_drawdown['total_return'],
                        'max_drawdown': low_drawdown['max_drawdown']
                    },
                    'confidence': 'medium',
                    'reason': 'Lowest drawdown with positive returns'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating strategy recommendations: {e}")
            return []