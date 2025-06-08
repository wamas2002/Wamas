"""
Portfolio Data Handler with Null Safety and Empty State Management
Addresses infinite extent warnings and data visualization issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import os

class PortfolioDataHandler:
    """Handles portfolio data with proper null safety and empty state management"""
    
    def __init__(self):
        self.default_portfolio = {
            'total_value': 0.0,
            'daily_pnl': 0.0,
            'daily_pnl_percent': 0.0,
            'positions': {},
            'trades_24h': 0,
            'win_rate_7d': 0.0
        }
    
    def get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data with proper null handling"""
        try:
            # Check for actual trading data
            if os.path.exists('data/trading_data.db'):
                portfolio_data = self._load_from_database()
                if portfolio_data:
                    return self._validate_portfolio_data(portfolio_data)
            
            # Return safe default state
            return self._get_empty_state_portfolio()
            
        except Exception as e:
            print(f"Error loading portfolio data: {e}")
            return self._get_empty_state_portfolio()
    
    def _load_from_database(self) -> Optional[Dict[str, Any]]:
        """Load portfolio data from database with error handling"""
        try:
            conn = sqlite3.connect('data/trading_data.db')
            
            # Check if tables exist
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                conn.close()
                return None
            
            # Try to calculate portfolio from trades
            portfolio_data = self._calculate_portfolio_from_trades(conn)
            conn.close()
            
            return portfolio_data
            
        except Exception as e:
            print(f"Database loading error: {e}")
            return None
    
    def _calculate_portfolio_from_trades(self, conn) -> Dict[str, Any]:
        """Calculate portfolio metrics from trade history"""
        try:
            # Get recent trades
            query = """
            SELECT symbol, action, quantity, price, timestamp 
            FROM trades 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
            """
            
            cutoff_time = (datetime.now() - timedelta(days=7)).isoformat()
            trades_df = pd.read_sql_query(query, conn, params=[cutoff_time])
            
            if trades_df.empty:
                return None
            
            # Calculate basic metrics
            portfolio = {
                'total_value': self._calculate_portfolio_value(trades_df),
                'daily_pnl': self._calculate_daily_pnl(trades_df),
                'daily_pnl_percent': self._calculate_daily_pnl_percent(trades_df),
                'positions': self._calculate_positions(trades_df),
                'trades_24h': self._count_recent_trades(trades_df, hours=24),
                'win_rate_7d': self._calculate_win_rate(trades_df)
            }
            
            return portfolio
            
        except Exception as e:
            print(f"Portfolio calculation error: {e}")
            return None
    
    def _get_empty_state_portfolio(self) -> Dict[str, Any]:
        """Return safe empty state for portfolio"""
        return {
            'total_value': 0.0,
            'daily_pnl': 0.0,
            'daily_pnl_percent': 0.0,
            'positions': {},
            'trades_24h': 0,
            'win_rate_7d': 0.0,
            'status': 'No trading activity',
            'message': 'System ready for trading - no positions currently open'
        }
    
    def _validate_portfolio_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize portfolio data"""
        validated = {}
        
        # Ensure all numeric values are finite
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if np.isfinite(value):
                    validated[key] = float(value)
                else:
                    validated[key] = 0.0
            else:
                validated[key] = value
        
        return validated
    
    def get_portfolio_chart_data(self) -> pd.DataFrame:
        """Get portfolio chart data with proper null handling"""
        try:
            # Generate safe chart data
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=30),
                end=datetime.now(),
                freq='D'
            )
            
            # Check for real data first
            real_data = self._load_portfolio_history()
            
            if real_data is not None and not real_data.empty:
                return self._validate_chart_data(real_data)
            
            # Return empty state chart data
            chart_data = pd.DataFrame({
                'timestamp': dates,
                'Portfolio Value': [0.0] * len(dates)
            })
            
            return chart_data
            
        except Exception as e:
            print(f"Chart data error: {e}")
            return self._get_empty_chart_data()
    
    def _load_portfolio_history(self) -> Optional[pd.DataFrame]:
        """Load portfolio history from database"""
        try:
            if not os.path.exists('data/trading_data.db'):
                return None
            
            conn = sqlite3.connect('data/trading_data.db')
            
            # Try to load portfolio history
            query = """
            SELECT timestamp, portfolio_value 
            FROM portfolio_history 
            WHERE timestamp > ? 
            ORDER BY timestamp
            """
            
            cutoff_time = (datetime.now() - timedelta(days=30)).isoformat()
            df = pd.read_sql_query(query, conn, params=[cutoff_time])
            conn.close()
            
            if df.empty:
                return None
            
            return df
            
        except Exception:
            return None
    
    def _validate_chart_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate chart data to prevent infinite extent errors"""
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with interpolation or forward fill
        data = data.fillna(method='ffill').fillna(0)
        
        # Ensure timestamp column is datetime
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        return data
    
    def _get_empty_chart_data(self) -> pd.DataFrame:
        """Return empty chart data for safe visualization"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='H'
        )
        
        return pd.DataFrame({
            'timestamp': dates,
            'Portfolio Value': [0.0] * len(dates)
        })
    
    def _calculate_portfolio_value(self, trades_df: pd.DataFrame) -> float:
        """Calculate current portfolio value"""
        try:
            # Simple calculation based on trades
            return 0.0  # Placeholder for real calculation
        except:
            return 0.0
    
    def _calculate_daily_pnl(self, trades_df: pd.DataFrame) -> float:
        """Calculate daily P&L"""
        try:
            return 0.0  # Placeholder for real calculation
        except:
            return 0.0
    
    def _calculate_daily_pnl_percent(self, trades_df: pd.DataFrame) -> float:
        """Calculate daily P&L percentage"""
        try:
            return 0.0  # Placeholder for real calculation
        except:
            return 0.0
    
    def _calculate_positions(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate current positions"""
        try:
            return {}  # Placeholder for real calculation
        except:
            return {}
    
    def _count_recent_trades(self, trades_df: pd.DataFrame, hours: int = 24) -> int:
        """Count recent trades"""
        try:
            cutoff = datetime.now() - timedelta(hours=hours)
            recent_trades = trades_df[pd.to_datetime(trades_df['timestamp']) > cutoff]
            return len(recent_trades)
        except:
            return 0
    
    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate win rate"""
        try:
            return 0.0  # Placeholder for real calculation
        except:
            return 0.0