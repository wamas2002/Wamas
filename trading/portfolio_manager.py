"""
Portfolio Manager
Tracks portfolio performance, positions, and analytics
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

class PortfolioManager:
    """Portfolio tracking and management"""
    
    def __init__(self, db_path: str = "data/portfolio.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
        # Portfolio configuration
        self.initial_balance = 10000.0  # $10k initial balance
        self.current_balance = self.initial_balance
        
    def _init_database(self):
        """Initialize portfolio database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_value REAL,
                    cash_balance REAL,
                    positions_value REAL,
                    daily_pnl REAL,
                    total_pnl REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price REAL,
                    quantity REAL,
                    entry_time DATETIME,
                    current_price REAL,
                    unrealized_pnl REAL,
                    strategy TEXT,
                    status TEXT DEFAULT 'open'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL,
                    price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT,
                    pnl REAL DEFAULT 0
                )
            """)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get latest portfolio value
                latest_value = conn.execute("""
                    SELECT total_value, cash_balance, positions_value, total_pnl
                    FROM portfolio_history 
                    ORDER BY timestamp DESC LIMIT 1
                """).fetchone()
                
                if not latest_value:
                    return {
                        'total_value': self.initial_balance,
                        'cash_balance': self.initial_balance,
                        'positions_value': 0.0,
                        'total_pnl': 0.0,
                        'total_pnl_percent': 0.0
                    }
                
                total_value, cash_balance, positions_value, total_pnl = latest_value
                
                return {
                    'total_value': total_value or self.initial_balance,
                    'cash_balance': cash_balance or self.initial_balance,
                    'positions_value': positions_value or 0.0,
                    'total_pnl': total_pnl or 0.0,
                    'total_pnl_percent': (total_pnl / self.initial_balance * 100) if total_pnl else 0.0
                }
        
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {
                'total_value': self.initial_balance,
                'cash_balance': self.initial_balance,
                'positions_value': 0.0,
                'total_pnl': 0.0,
                'total_pnl_percent': 0.0
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get portfolio performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get portfolio history for metrics
                df = pd.read_sql_query("""
                    SELECT timestamp, total_value, daily_pnl, total_pnl
                    FROM portfolio_history 
                    ORDER BY timestamp DESC LIMIT 100
                """, conn)
                
                if df.empty:
                    return {
                        'total_return': 0.0,
                        'daily_return': 0.0,
                        'max_drawdown': 0.0,
                        'sharpe_ratio': 0.0,
                        'win_rate': 0.0,
                        'total_trades': 0
                    }
                
                # Calculate metrics
                total_return = df['total_pnl'].iloc[0] if not df.empty else 0.0
                daily_return = df['daily_pnl'].iloc[0] if not df.empty else 0.0
                
                # Calculate max drawdown
                if len(df) > 1:
                    peak = df['total_value'].expanding().max()
                    drawdown = (df['total_value'] - peak) / peak
                    max_drawdown = drawdown.min()
                else:
                    max_drawdown = 0.0
                
                # Get trade statistics
                trades_df = pd.read_sql_query("""
                    SELECT pnl FROM trades WHERE pnl != 0
                """, conn)
                
                if not trades_df.empty:
                    win_rate = (trades_df['pnl'] > 0).mean()
                    total_trades = len(trades_df)
                    
                    # Simple Sharpe ratio approximation
                    if trades_df['pnl'].std() > 0:
                        sharpe_ratio = trades_df['pnl'].mean() / trades_df['pnl'].std()
                    else:
                        sharpe_ratio = 0.0
                else:
                    win_rate = 0.0
                    total_trades = 0
                    sharpe_ratio = 0.0
                
                return {
                    'total_return': total_return,
                    'daily_return': daily_return,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'win_rate': win_rate,
                    'total_trades': total_trades
                }
        
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_return': 0.0,
                'daily_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }
    
    def update_portfolio_value(self, total_value: float, positions_value: float):
        """Update portfolio value tracking"""
        try:
            cash_balance = total_value - positions_value
            
            # Get previous total for daily P&L calculation
            with sqlite3.connect(self.db_path) as conn:
                prev_value = conn.execute("""
                    SELECT total_value FROM portfolio_history 
                    ORDER BY timestamp DESC LIMIT 1
                """).fetchone()
                
                prev_total = prev_value[0] if prev_value else self.initial_balance
                daily_pnl = total_value - prev_total
                total_pnl = total_value - self.initial_balance
                
                # Insert new portfolio record
                conn.execute("""
                    INSERT INTO portfolio_history 
                    (total_value, cash_balance, positions_value, daily_pnl, total_pnl)
                    VALUES (?, ?, ?, ?, ?)
                """, (total_value, cash_balance, positions_value, daily_pnl, total_pnl))
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def record_trade(self, symbol: str, side: str, quantity: float, 
                    price: float, strategy: str, pnl: float = 0.0):
        """Record a trade"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades (symbol, side, quantity, price, strategy, pnl)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (symbol, side, quantity, price, strategy, pnl))
                
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT symbol, entry_price, quantity, entry_time, 
                           current_price, unrealized_pnl, strategy
                    FROM positions WHERE status = 'open'
                """)
                
                positions = []
                for row in cursor.fetchall():
                    positions.append({
                        'symbol': row[0],
                        'entry_price': row[1],
                        'quantity': row[2],
                        'entry_time': row[3],
                        'current_price': row[4],
                        'unrealized_pnl': row[5],
                        'strategy': row[6]
                    })
                
                return positions
        
        except Exception as e:
            self.logger.error(f"Error getting active positions: {e}")
            return []