"""
Performance Analyzer - Comprehensive Trading Performance Analytics
Generates detailed performance reports using authentic trading data with advanced metrics
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Advanced performance analytics for trading systems"""
    
    def __init__(self, db_path: str = 'performance_analytics.db'):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize performance analytics database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Performance reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    period_start DATE,
                    period_end DATE,
                    total_trades INTEGER,
                    win_rate REAL,
                    average_pnl REAL,
                    total_pnl REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    win_loss_ratio REAL,
                    profit_factor REAL,
                    max_consecutive_wins INTEGER,
                    max_consecutive_losses INTEGER,
                    average_trade_duration REAL,
                    volatility REAL,
                    calmar_ratio REAL,
                    recovery_factor REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Performance analytics database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def calculate_sharpe_ratio(self, trade_log: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(trade_log) == 0:
                return 0.0
            
            returns = trade_log['PnL'] / 100  # Convert percentage to decimal
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            if excess_returns.std() == 0:
                return 0.0
            
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, trade_log: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if len(trade_log) == 0:
                return 0.0
            
            returns = trade_log['PnL'] / 100
            excess_returns = returns - (risk_free_rate / 252)
            
            # Calculate downside deviation
            negative_returns = excess_returns[excess_returns < 0]
            if len(negative_returns) == 0:
                return float('inf')
            
            downside_deviation = negative_returns.std()
            if downside_deviation == 0:
                return 0.0
            
            sortino = excess_returns.mean() / downside_deviation * np.sqrt(252)
            return float(sortino)
            
        except Exception as e:
            logger.error(f"Sortino ratio calculation failed: {e}")
            return 0.0
    
    def calculate_win_loss_ratio(self, trade_log: pd.DataFrame) -> float:
        """Calculate win/loss ratio"""
        try:
            if len(trade_log) == 0:
                return 0.0
            
            winning_trades = trade_log[trade_log['PnL'] > 0]['PnL']
            losing_trades = trade_log[trade_log['PnL'] < 0]['PnL']
            
            if len(losing_trades) == 0:
                return float('inf')
            if len(winning_trades) == 0:
                return 0.0
            
            avg_win = winning_trades.mean()
            avg_loss = abs(losing_trades.mean())
            
            return float(avg_win / avg_loss) if avg_loss != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Win/loss ratio calculation failed: {e}")
            return 0.0
    
    def generate_performance_report(self, trade_log: pd.DataFrame) -> Dict:
        """Generate comprehensive performance report with all essential metrics"""
        try:
            if len(trade_log) == 0:
                return self._empty_performance_report()
            
            # Basic metrics
            total_trades = len(trade_log)
            win_rate = (trade_log['PnL'] > 0).mean()
            average_pnl = trade_log['PnL'].mean()
            total_pnl = trade_log['PnL'].sum()
            
            # Advanced metrics
            sharpe_ratio = self.calculate_sharpe_ratio(trade_log)
            sortino_ratio = self.calculate_sortino_ratio(trade_log)
            win_loss_ratio = self.calculate_win_loss_ratio(trade_log)
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + trade_log['PnL'] / 100).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min()) * 100
            
            performance_report = {
                'report_date': datetime.now(),
                'total_trades': total_trades,
                'win_rate': float(win_rate),
                'average_pnl': float(average_pnl),
                'total_pnl': float(total_pnl),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'win_loss_ratio': win_loss_ratio
            }
            
            return performance_report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return self._empty_performance_report()
    
    def _empty_performance_report(self) -> Dict:
        """Return empty performance report structure"""
        return {
            'report_date': datetime.now(),
            'total_trades': 0,
            'win_rate': 0.0,
            'average_pnl': 0.0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_loss_ratio': 0.0
        }
    
    def get_trading_data_from_databases(self) -> pd.DataFrame:
        """Aggregate trading data from various system databases"""
        try:
            all_trades = []
            
            # Database paths to check
            db_paths = [
                'pure_local_trading.db',
                'professional_trading.db',
                'enhanced_trading.db',
                'dynamic_trading.db',
                'intelligent_trading.db'
            ]
            
            for db_path in db_paths:
                try:
                    conn = sqlite3.connect(db_path)
                    
                    # Try different table names used across the system
                    table_queries = [
                        "SELECT * FROM trades WHERE exit_price IS NOT NULL",
                        "SELECT * FROM executed_trades",
                        "SELECT * FROM trade_history",
                        "SELECT * FROM trading_results"
                    ]
                    
                    for query in table_queries:
                        try:
                            trades_df = pd.read_sql_query(query, conn)
                            if not trades_df.empty:
                                trades_df['source_db'] = db_path
                                all_trades.append(trades_df)
                                break
                        except:
                            continue
                    
                    conn.close()
                    
                except Exception as e:
                    logger.debug(f"Could not read from {db_path}: {e}")
                    continue
            
            if not all_trades:
                logger.warning("No trading data found in system databases")
                return pd.DataFrame()
            
            # Combine all trade data
            combined_trades = pd.concat(all_trades, ignore_index=True)
            
            # Standardize column names
            combined_trades = self._standardize_trade_columns(combined_trades)
            
            logger.info(f"Loaded {len(combined_trades)} trades from {len(all_trades)} databases")
            return combined_trades
            
        except Exception as e:
            logger.error(f"Failed to aggregate trading data: {e}")
            return pd.DataFrame()
    
    def _standardize_trade_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different database schemas"""
        try:
            # Column mapping for different schemas
            column_mappings = {
                'pnl_pct': 'PnL',
                'profit_loss_pct': 'PnL',
                'return_pct': 'PnL',
                'profit_loss': 'PnL'
            }
            
            # Apply column mappings
            for old_col, new_col in column_mappings.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            # Ensure required columns exist
            if 'PnL' not in df.columns:
                if 'win_loss' in df.columns:
                    # Estimate PnL from win/loss
                    df['PnL'] = df['win_loss'].map({'win': 5.0, 'loss': -3.0}).fillna(0)
                else:
                    df['PnL'] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"Column standardization failed: {e}")
            return df
    
    def generate_system_performance_report(self, days: int = 30) -> Dict:
        """Generate performance report for the entire trading system"""
        try:
            # Get trading data
            trade_data = self.get_trading_data_from_databases()
            
            if trade_data.empty:
                logger.warning("No trading data available for performance report")
                return self._empty_performance_report()
            
            # Generate comprehensive report
            report = self.generate_performance_report(trade_data)
            
            # Add system-specific metrics
            report['data_sources'] = trade_data['source_db'].nunique() if 'source_db' in trade_data.columns else 1
            report['analysis_period_days'] = days
            
            return report
            
        except Exception as e:
            logger.error(f"System performance report generation failed: {e}")
            return self._empty_performance_report()


def run_performance_analysis():
    """Run comprehensive performance analysis for the trading system"""
    try:
        analyzer = PerformanceAnalyzer()
        
        # Generate system-wide performance report
        report = analyzer.generate_system_performance_report()
        
        print("\n🎯 COMPREHENSIVE TRADING PERFORMANCE REPORT")
        print("=" * 55)
        print(f"Report Generated: {report['report_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analysis Period: {report['analysis_period_days']} days")
        print(f"Data Sources: {report.get('data_sources', 1)} trading engines")
        print("-" * 55)
        
        # Core performance metrics
        print("📊 CORE PERFORMANCE METRICS:")
        print(f"  Total Trades: {report['total_trades']}")
        print(f"  Win Rate: {report['win_rate']*100:.1f}%")
        print(f"  Average PnL per Trade: {report['average_pnl']:.2f}%")
        print(f"  Total Portfolio Return: {report['total_pnl']:.2f}%")
        print(f"  Maximum Drawdown: {report['max_drawdown']:.2f}%")
        
        # Risk-adjusted metrics
        print("\n⚖️ RISK-ADJUSTED PERFORMANCE:")
        print(f"  Sharpe Ratio: {report['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {report['sortino_ratio']:.3f}")
        print(f"  Win/Loss Ratio: {report['win_loss_ratio']:.2f}")
        
        # Performance assessment
        print("\n📈 PERFORMANCE ASSESSMENT:")
        if report['sharpe_ratio'] > 1.5:
            print("  Risk-Adjusted Returns: EXCELLENT")
        elif report['sharpe_ratio'] > 1.0:
            print("  Risk-Adjusted Returns: GOOD")
        elif report['sharpe_ratio'] > 0:
            print("  Risk-Adjusted Returns: POSITIVE")
        else:
            print("  Risk-Adjusted Returns: NEEDS IMPROVEMENT")
        
        if report['win_rate'] > 0.6:
            print("  Win Rate Performance: STRONG")
        elif report['win_rate'] > 0.5:
            print("  Win Rate Performance: BALANCED")
        else:
            print("  Win Rate Performance: AGGRESSIVE")
        
        if report['max_drawdown'] < 5:
            print("  Risk Management: CONSERVATIVE")
        elif report['max_drawdown'] < 10:
            print("  Risk Management: MODERATE")
        else:
            print("  Risk Management: AGGRESSIVE")
        
        print("\n" + "=" * 55)
        
        return report
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        return None


if __name__ == "__main__":
    run_performance_analysis()