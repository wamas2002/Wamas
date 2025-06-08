import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from dataclasses import dataclass
from collections import deque

@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time"""
    timestamp: datetime
    portfolio_value: float
    cash_balance: float
    positions_value: float
    daily_return: float
    total_return: float
    drawdown: float
    volatility: float
    sharpe_ratio: float

class PerformanceTracker:
    """Comprehensive performance tracking and analysis"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.benchmark_symbol = "BTCUSDT"  # Default benchmark
        
        # Performance history
        self.snapshots = deque(maxlen=10000)  # Keep last 10k snapshots
        self.daily_returns = deque(maxlen=1000)  # Keep last 1k daily returns
        self.trade_returns = []
        
        # Rolling metrics
        self.rolling_windows = [7, 30, 90, 365]  # Days
        self.rolling_metrics = {}
        
        # Benchmark tracking
        self.benchmark_returns = deque(maxlen=1000)
        self.benchmark_prices = deque(maxlen=1000)
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.drawdown_start = None
        self.peak_value = initial_capital
        
        # Trade analysis
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        
    def update_portfolio_snapshot(self, portfolio_value: float, cash_balance: float, 
                                 positions_value: float, benchmark_price: float = None):
        """Update portfolio performance snapshot"""
        try:
            current_time = datetime.now()
            
            # Calculate returns
            if len(self.snapshots) > 0:
                prev_value = self.snapshots[-1].portfolio_value
                daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
            else:
                daily_return = 0.0
            
            total_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
            # Update peak and drawdown
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
                self.current_drawdown = 0.0
                self.drawdown_start = None
            else:
                self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
                if self.drawdown_start is None:
                    self.drawdown_start = current_time
            
            # Calculate volatility (last 30 days)
            recent_returns = list(self.daily_returns)[-30:]
            volatility = np.std(recent_returns) * np.sqrt(252) if len(recent_returns) > 1 else 0.0
            
            # Calculate Sharpe ratio
            if len(recent_returns) > 1 and volatility > 0:
                avg_return = np.mean(recent_returns)
                sharpe_ratio = avg_return / (volatility / np.sqrt(252))
            else:
                sharpe_ratio = 0.0
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                portfolio_value=portfolio_value,
                cash_balance=cash_balance,
                positions_value=positions_value,
                daily_return=daily_return,
                total_return=total_return,
                drawdown=self.current_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio
            )
            
            self.snapshots.append(snapshot)
            
            if daily_return != 0:  # Only add non-zero returns
                self.daily_returns.append(daily_return)
            
            # Update benchmark if provided
            if benchmark_price is not None:
                if len(self.benchmark_prices) > 0:
                    prev_benchmark = self.benchmark_prices[-1]
                    benchmark_return = (benchmark_price - prev_benchmark) / prev_benchmark
                    self.benchmark_returns.append(benchmark_return)
                
                self.benchmark_prices.append(benchmark_price)
            
            # Update rolling metrics
            self._update_rolling_metrics()
            
        except Exception as e:
            print(f"Error updating portfolio snapshot: {e}")
    
    def record_trade(self, entry_price: float, exit_price: float, quantity: float, 
                    action: str, fees: float = 0.0):
        """Record individual trade for analysis"""
        try:
            if action.upper() == 'SELL':  # Only record on trade completion
                trade_return = ((exit_price - entry_price) / entry_price) * quantity - fees
                self.trade_returns.append(trade_return)
                self.total_trades += 1
                
                if trade_return > 0:
                    self.winning_trades += 1
                    self.gross_profit += trade_return
                else:
                    self.losing_trades += 1
                    self.gross_loss += abs(trade_return)
                    
        except Exception as e:
            print(f"Error recording trade: {e}")
    
    def _update_rolling_metrics(self):
        """Update rolling performance metrics"""
        try:
            current_time = datetime.now()
            
            for window in self.rolling_windows:
                cutoff_time = current_time - timedelta(days=window)
                
                # Filter snapshots for this window
                window_snapshots = [
                    s for s in self.snapshots 
                    if s.timestamp >= cutoff_time
                ]
                
                if len(window_snapshots) < 2:
                    continue
                
                # Calculate metrics for this window
                start_value = window_snapshots[0].portfolio_value
                end_value = window_snapshots[-1].portfolio_value
                window_return = (end_value - start_value) / start_value
                
                # Extract daily returns for this window
                window_returns = [s.daily_return for s in window_snapshots if s.daily_return != 0]
                
                if len(window_returns) > 1:
                    window_volatility = np.std(window_returns) * np.sqrt(252)
                    window_sharpe = (np.mean(window_returns) / (window_volatility / np.sqrt(252))) if window_volatility > 0 else 0
                    
                    # Maximum drawdown in window
                    window_values = [s.portfolio_value for s in window_snapshots]
                    window_peak = max(window_values)
                    window_trough = min(window_values[window_values.index(window_peak):])
                    window_max_dd = (window_peak - window_trough) / window_peak if window_peak > 0 else 0
                    
                    self.rolling_metrics[f'{window}d'] = {
                        'return': window_return,
                        'volatility': window_volatility,
                        'sharpe_ratio': window_sharpe,
                        'max_drawdown': window_max_dd,
                        'snapshots_count': len(window_snapshots)
                    }
                    
        except Exception as e:
            print(f"Error updating rolling metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            if len(self.snapshots) == 0:
                return {'error': 'No performance data available'}
            
            latest_snapshot = self.snapshots[-1]
            
            # Basic metrics
            summary = {
                'current_portfolio_value': latest_snapshot.portfolio_value,
                'initial_capital': self.initial_capital,
                'total_return': latest_snapshot.total_return,
                'total_return_pct': latest_snapshot.total_return * 100,
                'current_drawdown': self.current_drawdown,
                'current_drawdown_pct': self.current_drawdown * 100,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown * 100,
                'peak_value': self.peak_value,
                'volatility': latest_snapshot.volatility,
                'sharpe_ratio': latest_snapshot.sharpe_ratio,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.winning_trades / max(1, self.total_trades),
                'profit_factor': self.gross_profit / max(0.01, self.gross_loss),
                'gross_profit': self.gross_profit,
                'gross_loss': self.gross_loss,
                'net_profit': self.gross_profit - self.gross_loss
            }
            
            # Calculate additional metrics if we have enough data
            if len(self.daily_returns) > 30:
                returns_array = np.array(list(self.daily_returns))
                
                summary.update({
                    'avg_daily_return': np.mean(returns_array),
                    'avg_daily_return_pct': np.mean(returns_array) * 100,
                    'best_day': np.max(returns_array),
                    'best_day_pct': np.max(returns_array) * 100,
                    'worst_day': np.min(returns_array),
                    'worst_day_pct': np.min(returns_array) * 100,
                    'positive_days': np.sum(returns_array > 0),
                    'negative_days': np.sum(returns_array < 0),
                    'positive_days_pct': np.sum(returns_array > 0) / len(returns_array) * 100
                })
                
                # Risk-adjusted metrics
                if summary['volatility'] > 0:
                    summary['calmar_ratio'] = summary['total_return'] / summary['max_drawdown'] if summary['max_drawdown'] > 0 else 0
                    summary['sortino_ratio'] = self._calculate_sortino_ratio(returns_array)
            
            # Rolling metrics
            summary['rolling_metrics'] = self.rolling_metrics.copy()
            
            # Benchmark comparison
            if len(self.benchmark_returns) > 0:
                benchmark_return = sum(self.benchmark_returns)
                summary['benchmark_return'] = benchmark_return
                summary['benchmark_return_pct'] = benchmark_return * 100
                summary['alpha'] = summary['total_return'] - benchmark_return
                summary['alpha_pct'] = summary['alpha'] * 100
                
                # Beta calculation
                if len(self.daily_returns) > 30 and len(self.benchmark_returns) > 30:
                    portfolio_returns = np.array(list(self.daily_returns)[-len(self.benchmark_returns):])
                    benchmark_returns_array = np.array(list(self.benchmark_returns))
                    
                    if len(portfolio_returns) == len(benchmark_returns_array):
                        covariance = np.cov(portfolio_returns, benchmark_returns_array)[0][1]
                        benchmark_variance = np.var(benchmark_returns_array)
                        summary['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            return summary
            
        except Exception as e:
            print(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation adjusted return)"""
        try:
            excess_returns = returns - target_return
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_deviation = np.sqrt(np.mean(downside_returns**2))
            
            if downside_deviation == 0:
                return float('inf')
            
            return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            
        except Exception as e:
            print(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def get_return_distribution(self) -> Dict[str, Any]:
        """Get return distribution analysis"""
        try:
            if len(self.daily_returns) < 10:
                return {'error': 'Insufficient return data'}
            
            returns_array = np.array(list(self.daily_returns))
            
            return {
                'count': len(returns_array),
                'mean': np.mean(returns_array),
                'std': np.std(returns_array),
                'min': np.min(returns_array),
                'max': np.max(returns_array),
                'skewness': self._calculate_skewness(returns_array),
                'kurtosis': self._calculate_kurtosis(returns_array),
                'percentiles': {
                    '5th': np.percentile(returns_array, 5),
                    '25th': np.percentile(returns_array, 25),
                    '50th': np.percentile(returns_array, 50),
                    '75th': np.percentile(returns_array, 75),
                    '95th': np.percentile(returns_array, 95)
                },
                'var_95': np.percentile(returns_array, 5),  # 95% VaR
                'var_99': np.percentile(returns_array, 1),  # 99% VaR
                'cvar_95': np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)]),
                'cvar_99': np.mean(returns_array[returns_array <= np.percentile(returns_array, 1)])
            }
            
        except Exception as e:
            print(f"Error getting return distribution: {e}")
            return {'error': str(e)}
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            skewness = np.mean(((returns - mean_return) / std_return) ** 3)
            return skewness
            
        except Exception as e:
            print(f"Error calculating skewness: {e}")
            return 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
            return kurtosis
            
        except Exception as e:
            print(f"Error calculating kurtosis: {e}")
            return 0.0
    
    def get_drawdown_analysis(self) -> Dict[str, Any]:
        """Get detailed drawdown analysis"""
        try:
            if len(self.snapshots) < 10:
                return {'error': 'Insufficient data for drawdown analysis'}
            
            # Extract portfolio values and timestamps
            values = [s.portfolio_value for s in self.snapshots]
            timestamps = [s.timestamp for s in self.snapshots]
            
            # Calculate running maximum and drawdowns
            running_max = np.maximum.accumulate(values)
            drawdowns = (running_max - values) / running_max
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_idx = 0
            
            for i, dd in enumerate(drawdowns):
                if dd > 0.01 and not in_drawdown:  # Start of drawdown (>1%)
                    in_drawdown = True
                    start_idx = i
                elif dd < 0.001 and in_drawdown:  # End of drawdown (<0.1%)
                    in_drawdown = False
                    
                    period_drawdowns = drawdowns[start_idx:i+1]
                    max_dd = np.max(period_drawdowns)
                    duration = timestamps[i] - timestamps[start_idx]
                    
                    drawdown_periods.append({
                        'start_date': timestamps[start_idx].isoformat(),
                        'end_date': timestamps[i].isoformat(),
                        'duration_days': duration.days,
                        'max_drawdown': max_dd,
                        'max_drawdown_pct': max_dd * 100,
                        'start_value': values[start_idx],
                        'trough_value': min(values[start_idx:i+1]),
                        'end_value': values[i]
                    })
            
            # Current drawdown info
            current_dd_info = {}
            if self.current_drawdown > 0:
                current_dd_info = {
                    'current_drawdown': self.current_drawdown,
                    'current_drawdown_pct': self.current_drawdown * 100,
                    'drawdown_start': self.drawdown_start.isoformat() if self.drawdown_start else None,
                    'duration_days': (datetime.now() - self.drawdown_start).days if self.drawdown_start else 0
                }
            
            return {
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown * 100,
                'current_drawdown_info': current_dd_info,
                'total_drawdown_periods': len(drawdown_periods),
                'drawdown_periods': drawdown_periods[-10:],  # Last 10 drawdown periods
                'avg_drawdown_duration': np.mean([p['duration_days'] for p in drawdown_periods]) if drawdown_periods else 0,
                'avg_drawdown_magnitude': np.mean([p['max_drawdown'] for p in drawdown_periods]) if drawdown_periods else 0
            }
            
        except Exception as e:
            print(f"Error getting drawdown analysis: {e}")
            return {'error': str(e)}
    
    def get_monthly_returns(self) -> Dict[str, Any]:
        """Get monthly return breakdown"""
        try:
            if len(self.snapshots) < 30:
                return {'error': 'Insufficient data for monthly analysis'}
            
            # Group snapshots by month
            monthly_data = {}
            
            for snapshot in self.snapshots:
                month_key = snapshot.timestamp.strftime('%Y-%m')
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = {
                        'start_value': snapshot.portfolio_value,
                        'end_value': snapshot.portfolio_value,
                        'high_value': snapshot.portfolio_value,
                        'low_value': snapshot.portfolio_value,
                        'snapshots': []
                    }
                
                monthly_data[month_key]['end_value'] = snapshot.portfolio_value
                monthly_data[month_key]['high_value'] = max(
                    monthly_data[month_key]['high_value'], 
                    snapshot.portfolio_value
                )
                monthly_data[month_key]['low_value'] = min(
                    monthly_data[month_key]['low_value'], 
                    snapshot.portfolio_value
                )
                monthly_data[month_key]['snapshots'].append(snapshot)
            
            # Calculate monthly returns
            monthly_returns = []
            
            for month, data in monthly_data.items():
                if data['start_value'] > 0:
                    monthly_return = (data['end_value'] - data['start_value']) / data['start_value']
                    
                    monthly_returns.append({
                        'month': month,
                        'return': monthly_return,
                        'return_pct': monthly_return * 100,
                        'start_value': data['start_value'],
                        'end_value': data['end_value'],
                        'high_value': data['high_value'],
                        'low_value': data['low_value'],
                        'volatility': np.std([s.daily_return for s in data['snapshots']]) * np.sqrt(30)
                    })
            
            # Sort by month
            monthly_returns.sort(key=lambda x: x['month'])
            
            # Calculate statistics
            returns_only = [m['return'] for m in monthly_returns]
            
            if returns_only:
                stats = {
                    'total_months': len(returns_only),
                    'positive_months': sum(1 for r in returns_only if r > 0),
                    'negative_months': sum(1 for r in returns_only if r < 0),
                    'best_month': max(returns_only),
                    'worst_month': min(returns_only),
                    'avg_monthly_return': np.mean(returns_only),
                    'monthly_volatility': np.std(returns_only),
                    'positive_month_rate': sum(1 for r in returns_only if r > 0) / len(returns_only)
                }
            else:
                stats = {}
            
            return {
                'monthly_returns': monthly_returns,
                'statistics': stats
            }
            
        except Exception as e:
            print(f"Error getting monthly returns: {e}")
            return {'error': str(e)}
    
    def export_performance_data(self, format: str = 'csv') -> str:
        """Export performance data to file"""
        try:
            if len(self.snapshots) == 0:
                return ""
            
            # Convert snapshots to DataFrame
            data = []
            for snapshot in self.snapshots:
                data.append({
                    'timestamp': snapshot.timestamp.isoformat(),
                    'portfolio_value': snapshot.portfolio_value,
                    'cash_balance': snapshot.cash_balance,
                    'positions_value': snapshot.positions_value,
                    'daily_return': snapshot.daily_return,
                    'total_return': snapshot.total_return,
                    'drawdown': snapshot.drawdown,
                    'volatility': snapshot.volatility,
                    'sharpe_ratio': snapshot.sharpe_ratio
                })
            
            df = pd.DataFrame(data)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == 'csv':
                filename = f"performance_data_{timestamp}.csv"
                df.to_csv(filename, index=False)
            elif format.lower() == 'json':
                filename = f"performance_data_{timestamp}.json"
                df.to_json(filename, orient='records', date_format='iso')
            else:
                return ""
            
            return filename
            
        except Exception as e:
            print(f"Error exporting performance data: {e}")
            return ""
    
    def reset_performance_tracking(self):
        """Reset all performance tracking data"""
        try:
            self.snapshots.clear()
            self.daily_returns.clear()
            self.trade_returns.clear()
            self.rolling_metrics.clear()
            self.benchmark_returns.clear()
            self.benchmark_prices.clear()
            
            self.max_drawdown = 0.0
            self.current_drawdown = 0.0
            self.drawdown_start = None
            self.peak_value = self.initial_capital
            
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_trades = 0
            self.gross_profit = 0.0
            self.gross_loss = 0.0
            
            print("Performance tracking reset successfully")
            
        except Exception as e:
            print(f"Error resetting performance tracking: {e}")
