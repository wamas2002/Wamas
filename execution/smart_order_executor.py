"""
Smart Order Executor - TWAP and Iceberg Implementation
Intelligent order splitting and execution strategies based on market conditions
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
import threading
import sqlite3
from datetime import datetime, timedelta
import asyncio
from .order_book_monitor import OrderBookMonitor

logger = logging.getLogger(__name__)

class SmartOrderExecutor:
    """Execute orders using TWAP, Iceberg, or Market strategies based on conditions"""
    
    def __init__(self):
        self.exchange = None
        self.order_book_monitor = OrderBookMonitor()
        self.active_orders = {}
        self.execution_history = []
        
        # Execution parameters
        self.twap_slice_duration = 300  # 5 minutes per slice
        self.iceberg_slice_ratio = 0.1  # 10% of order per slice
        self.min_order_size = 10.0      # Minimum USDT order size
        
        self.setup_database()
        self.initialize_exchange()
    
    def setup_database(self):
        """Initialize smart execution database"""
        try:
            conn = sqlite3.connect('smart_execution.db')
            cursor = conn.cursor()
            
            # Execution orders table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    target_price REAL,
                    execution_strategy TEXT,
                    
                    -- Progress tracking
                    filled_amount REAL DEFAULT 0,
                    remaining_amount REAL,
                    average_fill_price REAL,
                    slices_total INTEGER,
                    slices_completed INTEGER,
                    
                    -- Status
                    status TEXT DEFAULT 'active',
                    started_at DATETIME,
                    completed_at DATETIME,
                    
                    -- Performance metrics
                    slippage_pct REAL,
                    execution_time_minutes REAL,
                    implementation_shortfall REAL
                )
            ''')
            
            # Order slices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_slices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_order_id TEXT,
                    slice_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL,
                    
                    -- Execution details
                    exchange_order_id TEXT,
                    filled_amount REAL DEFAULT 0,
                    fill_price REAL,
                    status TEXT DEFAULT 'pending',
                    
                    -- Market conditions at execution
                    spread_pct REAL,
                    order_book_quality TEXT,
                    market_impact_est REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Smart execution database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            import os
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            self.exchange.load_markets()
            logger.info("Smart order executor connected to OKX")
            
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
    
    def determine_execution_strategy(self, symbol: str, amount: float, 
                                   current_price: float) -> str:
        """Determine optimal execution strategy based on order size and market conditions"""
        try:
            # Get market quality assessment
            can_execute, quality_reason = self.order_book_monitor.should_execute_trade(symbol)
            
            if not can_execute:
                return 'delay'
            
            # Calculate order size relative to market
            order_value = amount * current_price
            
            # Get recent volume for context
            ticker = self.exchange.fetch_ticker(symbol)
            daily_volume = ticker.get('quoteVolume', 0)
            
            if daily_volume == 0:
                return 'market'  # Default to market order
            
            # Calculate order size as percentage of daily volume
            volume_ratio = order_value / daily_volume
            
            # Strategy selection logic
            if order_value < 50:  # Small orders (<$50)
                return 'market'
            elif volume_ratio < 0.001:  # <0.1% of daily volume
                return 'limit'
            elif volume_ratio < 0.005:  # <0.5% of daily volume
                return 'iceberg'
            else:  # Large orders (>0.5% of daily volume)
                return 'twap'
                
        except Exception as e:
            logger.error(f"Strategy determination failed: {e}")
            return 'market'  # Safe default
    
    def execute_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """Execute immediate market order"""
        try:
            logger.info(f"Executing market order: {side} {amount} {symbol}")
            
            order = self.exchange.create_market_order(symbol, side, amount)
            
            # Record execution
            execution_record = {
                'order_id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'strategy': 'market',
                'status': 'completed',
                'exchange_order': order
            }
            
            self.save_execution_record(execution_record)
            
            return {
                'success': True,
                'order_id': order['id'],
                'strategy': 'market',
                'message': f"Market order executed: {amount} {symbol}"
            }
            
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'market'
            }
    
    def execute_limit_order(self, symbol: str, side: str, amount: float, 
                          target_price: float = None) -> Dict:
        """Execute limit order with smart pricing"""
        try:
            if target_price is None:
                # Get current best price
                order_book = self.exchange.fetch_order_book(symbol, 5)
                if side.lower() == 'buy':
                    target_price = order_book['bids'][0][0] * 1.001  # Slightly above best bid
                else:
                    target_price = order_book['asks'][0][0] * 0.999  # Slightly below best ask
            
            logger.info(f"Executing limit order: {side} {amount} {symbol} @ {target_price}")
            
            order = self.exchange.create_limit_order(symbol, side, amount, target_price)
            
            # Record execution
            execution_record = {
                'order_id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': target_price,
                'strategy': 'limit',
                'status': 'pending',
                'exchange_order': order
            }
            
            self.save_execution_record(execution_record)
            
            return {
                'success': True,
                'order_id': order['id'],
                'strategy': 'limit',
                'price': target_price,
                'message': f"Limit order placed: {amount} {symbol} @ {target_price}"
            }
            
        except Exception as e:
            logger.error(f"Limit order execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'limit'
            }
    
    def execute_iceberg_order(self, symbol: str, side: str, total_amount: float, 
                            target_price: float = None) -> Dict:
        """Execute iceberg order with hidden quantity"""
        try:
            order_id = f"iceberg_{symbol}_{int(time.time())}"
            slice_size = max(total_amount * self.iceberg_slice_ratio, 
                           self.min_order_size / self.exchange.fetch_ticker(symbol)['last'])
            
            logger.info(f"Starting iceberg execution: {side} {total_amount} {symbol}, "
                       f"slice size: {slice_size}")
            
            # Create parent order record
            self.create_parent_order(order_id, symbol, side, total_amount, 'iceberg', target_price)
            
            # Start iceberg execution in background
            iceberg_thread = threading.Thread(
                target=self._execute_iceberg_slices,
                args=(order_id, symbol, side, total_amount, slice_size, target_price),
                daemon=True
            )
            iceberg_thread.start()
            
            return {
                'success': True,
                'order_id': order_id,
                'strategy': 'iceberg',
                'slice_size': slice_size,
                'message': f"Iceberg order started: {total_amount} {symbol}"
            }
            
        except Exception as e:
            logger.error(f"Iceberg order initialization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'iceberg'
            }
    
    def _execute_iceberg_slices(self, order_id: str, symbol: str, side: str, 
                              total_amount: float, slice_size: float, target_price: float = None):
        """Execute iceberg order slices"""
        try:
            remaining_amount = total_amount
            slice_count = 0
            
            while remaining_amount > 0:
                # Calculate current slice amount
                current_slice = min(slice_size, remaining_amount)
                
                # Check market conditions before each slice
                can_execute, reason = self.order_book_monitor.should_execute_trade(symbol)
                if not can_execute:
                    logger.warning(f"Pausing iceberg execution for {symbol}: {reason}")
                    time.sleep(60)  # Wait before retrying
                    continue
                
                # Execute slice
                slice_result = self.execute_limit_order(symbol, side, current_slice, target_price)
                
                if slice_result['success']:
                    remaining_amount -= current_slice
                    slice_count += 1
                    
                    # Update parent order progress
                    self.update_order_progress(order_id, current_slice)
                    
                    logger.info(f"Iceberg slice {slice_count} completed: {current_slice} {symbol}, "
                               f"remaining: {remaining_amount}")
                    
                    # Wait between slices
                    time.sleep(30)
                else:
                    logger.error(f"Iceberg slice failed: {slice_result.get('error')}")
                    break
            
            # Mark order as completed
            if remaining_amount <= 0:
                self.complete_order(order_id)
                logger.info(f"Iceberg order completed: {order_id}")
            
        except Exception as e:
            logger.error(f"Iceberg execution failed: {e}")
            self.fail_order(order_id, str(e))
    
    def execute_twap_order(self, symbol: str, side: str, total_amount: float, 
                         duration_minutes: int = 60) -> Dict:
        """Execute TWAP (Time-Weighted Average Price) order"""
        try:
            order_id = f"twap_{symbol}_{int(time.time())}"
            
            # Calculate number of slices
            num_slices = max(4, duration_minutes // 15)  # At least 4 slices, 15 min intervals
            slice_size = total_amount / num_slices
            slice_interval = (duration_minutes * 60) / num_slices  # seconds
            
            logger.info(f"Starting TWAP execution: {side} {total_amount} {symbol}, "
                       f"{num_slices} slices over {duration_minutes} minutes")
            
            # Create parent order record
            self.create_parent_order(order_id, symbol, side, total_amount, 'twap')
            
            # Start TWAP execution in background
            twap_thread = threading.Thread(
                target=self._execute_twap_slices,
                args=(order_id, symbol, side, slice_size, num_slices, slice_interval),
                daemon=True
            )
            twap_thread.start()
            
            return {
                'success': True,
                'order_id': order_id,
                'strategy': 'twap',
                'num_slices': num_slices,
                'duration_minutes': duration_minutes,
                'message': f"TWAP order started: {total_amount} {symbol} over {duration_minutes}min"
            }
            
        except Exception as e:
            logger.error(f"TWAP order initialization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'twap'
            }
    
    def _execute_twap_slices(self, order_id: str, symbol: str, side: str, 
                           slice_size: float, num_slices: int, slice_interval: float):
        """Execute TWAP order slices at timed intervals"""
        try:
            for slice_num in range(num_slices):
                # Check market conditions
                can_execute, reason = self.order_book_monitor.should_execute_trade(symbol)
                if not can_execute:
                    logger.warning(f"Skipping TWAP slice for {symbol}: {reason}")
                else:
                    # Execute slice with market order for guaranteed fill
                    slice_result = self.execute_market_order(symbol, side, slice_size)
                    
                    if slice_result['success']:
                        self.update_order_progress(order_id, slice_size)
                        logger.info(f"TWAP slice {slice_num + 1}/{num_slices} completed: "
                                   f"{slice_size} {symbol}")
                    else:
                        logger.error(f"TWAP slice failed: {slice_result.get('error')}")
                
                # Wait for next slice (except for last slice)
                if slice_num < num_slices - 1:
                    time.sleep(slice_interval)
            
            # Mark order as completed
            self.complete_order(order_id)
            logger.info(f"TWAP order completed: {order_id}")
            
        except Exception as e:
            logger.error(f"TWAP execution failed: {e}")
            self.fail_order(order_id, str(e))
    
    def smart_execute_order(self, symbol: str, side: str, amount: float, 
                          target_price: float = None, strategy: str = 'auto') -> Dict:
        """Main entry point for smart order execution"""
        try:
            # Get current market price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Auto-determine strategy if not specified
            if strategy == 'auto':
                strategy = self.determine_execution_strategy(symbol, amount, current_price)
            
            # Check if execution should be delayed
            if strategy == 'delay':
                return {
                    'success': False,
                    'strategy': 'delay',
                    'message': 'Execution delayed due to poor market conditions'
                }
            
            # Execute based on strategy
            if strategy == 'market':
                return self.execute_market_order(symbol, side, amount)
            elif strategy == 'limit':
                return self.execute_limit_order(symbol, side, amount, target_price)
            elif strategy == 'iceberg':
                return self.execute_iceberg_order(symbol, side, amount, target_price)
            elif strategy == 'twap':
                return self.execute_twap_order(symbol, side, amount)
            else:
                # Default to market order
                return self.execute_market_order(symbol, side, amount)
                
        except Exception as e:
            logger.error(f"Smart order execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy': strategy
            }
    
    def create_parent_order(self, order_id: str, symbol: str, side: str, 
                          total_amount: float, strategy: str, target_price: float = None):
        """Create parent order record in database"""
        try:
            conn = sqlite3.connect('smart_execution.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO execution_orders (
                    order_id, symbol, side, total_amount, target_price,
                    execution_strategy, remaining_amount, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order_id, symbol, side, total_amount, target_price,
                strategy, total_amount, datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to create parent order: {e}")
    
    def update_order_progress(self, order_id: str, filled_amount: float):
        """Update order execution progress"""
        try:
            conn = sqlite3.connect('smart_execution.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE execution_orders 
                SET filled_amount = filled_amount + ?,
                    remaining_amount = total_amount - (filled_amount + ?)
                WHERE order_id = ?
            ''', (filled_amount, filled_amount, order_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update order progress: {e}")
    
    def complete_order(self, order_id: str):
        """Mark order as completed"""
        try:
            conn = sqlite3.connect('smart_execution.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE execution_orders 
                SET status = 'completed', completed_at = ?
                WHERE order_id = ?
            ''', (datetime.now(), order_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to complete order: {e}")
    
    def fail_order(self, order_id: str, error_message: str):
        """Mark order as failed"""
        try:
            conn = sqlite3.connect('smart_execution.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE execution_orders 
                SET status = 'failed', completed_at = ?
                WHERE order_id = ?
            ''', (datetime.now(), order_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to mark order as failed: {e}")
    
    def save_execution_record(self, record: Dict):
        """Save execution record for analysis"""
        try:
            self.execution_history.append({
                'timestamp': datetime.now(),
                'record': record
            })
            
            # Keep only recent history
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
                
        except Exception as e:
            logger.error(f"Failed to save execution record: {e}")
    
    def get_execution_summary(self) -> Dict:
        """Get summary of recent executions"""
        try:
            conn = sqlite3.connect('smart_execution.db')
            
            summary = pd.read_sql_query('''
                SELECT 
                    execution_strategy,
                    COUNT(*) as count,
                    AVG(execution_time_minutes) as avg_time,
                    AVG(slippage_pct) as avg_slippage
                FROM execution_orders 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY execution_strategy
            ''', conn)
            
            conn.close()
            
            return summary.to_dict('records') if not summary.empty else []
            
        except Exception as e:
            logger.error(f"Execution summary failed: {e}")
            return []