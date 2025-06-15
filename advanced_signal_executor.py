#!/usr/bin/env python3
"""
Advanced Signal Executor
High-frequency signal execution with intelligent position sizing and risk management
"""

import ccxt
import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedSignalExecutor:
    def __init__(self):
        self.exchange = None
        self.db_path = 'advanced_signal_executor.db'
        self.initialize_exchange()
        self.setup_database()
        
        # Execution parameters optimized for your current balance
        self.base_position_size = 1.5  # $1.50 per trade (ultra-conservative)
        self.max_daily_trades = 12     # Maximum trades per day
        self.min_confidence = 70       # Minimum signal confidence
        self.max_positions = 10        # Maximum concurrent positions
        
        # Track execution metrics
        self.daily_trade_count = 0
        self.execution_success_rate = 0
        self.last_reset_date = datetime.now().date()

    def initialize_exchange(self):
        """Initialize OKX exchange for signal execution"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            self.exchange.load_markets()
            logger.info("Advanced signal executor connected to OKX")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")

    def setup_database(self):
        """Setup signal execution database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Signal execution tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL,
                    position_size REAL,
                    leverage REAL DEFAULT 1,
                    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_system TEXT,
                    status TEXT DEFAULT 'EXECUTED',
                    order_id TEXT,
                    fill_price REAL,
                    fees_paid REAL DEFAULT 0
                )
            ''')
            
            # Daily execution metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    trades_executed INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0,
                    total_volume REAL DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    profit_loss REAL DEFAULT 0
                )
            ''')
            
            # Signal quality scoring
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    source_system TEXT NOT NULL,
                    avg_confidence REAL,
                    success_rate REAL,
                    execution_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Advanced signal executor database ready")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")

    def get_account_status(self) -> Dict:
        """Get current account status for position sizing"""
        try:
            balance = self.exchange.fetch_balance()
            positions = self.exchange.fetch_positions()
            
            total_balance = float(balance['USDT']['total'])
            active_positions = len([p for p in positions if p['contracts'] and float(p['contracts']) > 0])
            
            # Calculate available margin for new trades
            used_margin = sum(float(p.get('initialMargin', 0)) for p in positions if p['contracts'])
            available_balance = total_balance - used_margin
            
            return {
                'total_balance': total_balance,
                'available_balance': available_balance,
                'active_positions': active_positions,
                'max_position_value': min(self.base_position_size, available_balance * 0.01),  # 1% max per trade
                'can_trade': active_positions < self.max_positions and available_balance > self.base_position_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get account status: {e}")
            return {'can_trade': False}

    def get_pending_signals(self) -> List[Dict]:
        """Get high-quality signals from all trading systems"""
        signals = []
        
        # Database sources for signals
        signal_sources = {
            'enhanced_ai': 'enhanced_ai_trading.db',
            'futures_trading': 'advanced_futures_trading.db',
            'market_scanner': 'advanced_market_scanner.db'
        }
        
        for source, db_path in signal_sources.items():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check available tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                if 'futures_signals' in tables:
                    cursor.execute('''
                        SELECT symbol, signal_type, confidence, target_price, stop_loss, timestamp
                        FROM futures_signals 
                        WHERE confidence >= ? AND timestamp >= datetime('now', '-30 minutes')
                        ORDER BY confidence DESC LIMIT 5
                    ''', (self.min_confidence,))
                    
                    for row in cursor.fetchall():
                        if len(row) >= 6:
                            signals.append({
                                'symbol': row[0],
                                'signal_type': row[1],
                                'confidence': row[2],
                                'target_price': row[3],
                                'stop_loss': row[4],
                                'timestamp': row[5],
                                'source': source
                            })
                
                elif 'scan_results' in tables:
                    cursor.execute('''
                        SELECT symbol, signal_type, confidence, target_price, timestamp
                        FROM scan_results 
                        WHERE confidence >= ? AND timestamp >= datetime('now', '-30 minutes')
                        ORDER BY confidence DESC LIMIT 5
                    ''', (self.min_confidence,))
                    
                    for row in cursor.fetchall():
                        if len(row) >= 5:
                            signals.append({
                                'symbol': row[0],
                                'signal_type': row[1],
                                'confidence': row[2],
                                'target_price': row[3],
                                'timestamp': row[4],
                                'source': source
                            })
                
                conn.close()
                
            except Exception as e:
                logger.debug(f"Could not access {source} signals: {e}")
                continue
        
        # Filter and prioritize signals
        return self.filter_and_rank_signals(signals)

    def filter_and_rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """Filter and rank signals by quality and opportunity"""
        if not signals:
            return []
        
        # Remove duplicates (same symbol within 15 minutes)
        unique_signals = {}
        for signal in signals:
            key = signal['symbol']
            if key not in unique_signals or signal['confidence'] > unique_signals[key]['confidence']:
                unique_signals[key] = signal
        
        # Convert back to list and sort by confidence
        filtered_signals = list(unique_signals.values())
        filtered_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Check for existing positions to avoid over-concentration
        try:
            positions = self.exchange.fetch_positions()
            position_symbols = {p['symbol'] for p in positions if p['contracts'] and float(p['contracts']) > 0}
            
            # Filter out symbols we already have positions in
            filtered_signals = [s for s in filtered_signals if s['symbol'] not in position_symbols]
            
        except Exception as e:
            logger.error(f"Error filtering existing positions: {e}")
        
        return filtered_signals[:5]  # Top 5 signals

    def calculate_optimal_position_size(self, signal: Dict, account_status: Dict) -> Tuple[float, float]:
        """Calculate optimal position size and leverage"""
        max_position_value = account_status.get('max_position_value', self.base_position_size)
        
        # Confidence-based position sizing
        confidence_multiplier = min(signal['confidence'] / 100, 1.0)
        base_size = max_position_value * confidence_multiplier
        
        # Ensure minimum viable position
        position_value = max(base_size, 1.0)  # Minimum $1 position
        
        # Conservative leverage based on confidence
        if signal['confidence'] >= 85:
            leverage = 2.0  # Higher confidence = slightly higher leverage
        elif signal['confidence'] >= 75:
            leverage = 1.5
        else:
            leverage = 1.0  # No leverage for moderate confidence
        
        return position_value, leverage

    def execute_signal(self, signal: Dict) -> bool:
        """Execute a trading signal with proper risk management"""
        try:
            account_status = self.get_account_status()
            
            if not account_status.get('can_trade', False):
                logger.warning(f"Cannot execute {signal['symbol']}: Account constraints")
                return False
            
            # Calculate position size
            position_value, leverage = self.calculate_optimal_position_size(signal, account_status)
            
            # Get current market price
            ticker = self.exchange.fetch_ticker(signal['symbol'])
            current_price = float(ticker['last'])
            
            # Calculate position size in base currency
            position_size = position_value / current_price
            
            # Determine order side
            side = 'buy' if signal['signal_type'].upper() in ['BUY', 'LONG'] else 'sell'
            
            # Execute market order
            order = self.exchange.create_market_order(
                symbol=signal['symbol'],
                side=side,
                amount=position_size,
                params={'leverage': leverage} if leverage > 1 else {}
            )
            
            fill_price = float(order.get('price', current_price))
            fees = float(order.get('fee', {}).get('cost', 0))
            
            logger.info(f"✅ EXECUTED: {signal['symbol']} {side.upper()} "
                       f"Size: ${position_value:.2f} ({position_size:.4f}) "
                       f"Price: ${fill_price:.4f} "
                       f"Confidence: {signal['confidence']:.1f}% "
                       f"Leverage: {leverage}x")
            
            # Save execution record
            self.save_execution_record(signal, order, position_value, leverage, fill_price, fees)
            
            # Update daily metrics
            self.update_daily_metrics(True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute signal {signal['symbol']}: {e}")
            self.update_daily_metrics(False)
            return False

    def save_execution_record(self, signal: Dict, order: Dict, position_value: float, 
                            leverage: float, fill_price: float, fees: float):
        """Save execution record to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signal_executions 
                (symbol, signal_type, confidence, entry_price, position_size, leverage,
                 source_system, order_id, fill_price, fees_paid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (signal['symbol'], signal['signal_type'], signal['confidence'],
                  signal.get('target_price', fill_price), position_value, leverage,
                  signal.get('source', 'unknown'), order.get('id', ''), fill_price, fees))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save execution record: {e}")

    def update_daily_metrics(self, success: bool):
        """Update daily execution metrics"""
        try:
            today = datetime.now().date().isoformat()
            
            # Reset daily counter if new day
            if self.last_reset_date != datetime.now().date():
                self.daily_trade_count = 0
                self.last_reset_date = datetime.now().date()
            
            self.daily_trade_count += 1
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update or insert daily metrics
            cursor.execute('''
                INSERT OR REPLACE INTO daily_metrics 
                (date, trades_executed, success_rate)
                VALUES (?, ?, ?)
            ''', (today, self.daily_trade_count, 
                  (self.execution_success_rate * (self.daily_trade_count - 1) + (1 if success else 0)) / self.daily_trade_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update daily metrics: {e}")

    def run_signal_execution_cycle(self):
        """Main signal execution cycle"""
        try:
            logger.info("🎯 Running advanced signal execution cycle...")
            
            # Check daily trade limits
            if self.daily_trade_count >= self.max_daily_trades:
                logger.info(f"📊 Daily trade limit reached ({self.max_daily_trades})")
                return
            
            # Get account status
            account_status = self.get_account_status()
            logger.info(f"💰 Account: ${account_status.get('total_balance', 0):.2f} "
                       f"Available: ${account_status.get('available_balance', 0):.2f} "
                       f"Positions: {account_status.get('active_positions', 0)}")
            
            if not account_status.get('can_trade', False):
                logger.info("⚠️ Trading constraints active - holding current positions")
                return
            
            # Get and execute high-quality signals
            signals = self.get_pending_signals()
            
            if not signals:
                logger.info("📭 No high-quality signals available for execution")
                return
            
            executions = 0
            max_executions = min(3, self.max_daily_trades - self.daily_trade_count)
            
            for signal in signals[:max_executions]:
                if self.execute_signal(signal):
                    executions += 1
                    time.sleep(2)  # Brief pause between executions
                
                # Check if we've hit daily limits
                if self.daily_trade_count >= self.max_daily_trades:
                    break
            
            if executions > 0:
                logger.info(f"✅ Signal execution cycle complete: {executions} trades executed")
            else:
                logger.info("✅ Signal execution cycle complete: No suitable signals executed")
                
        except Exception as e:
            logger.error(f"Signal execution cycle error: {e}")

def main():
    """Main signal execution function"""
    executor = AdvancedSignalExecutor()
    
    logger.info("🚀 Starting Advanced Signal Executor")
    logger.info("⚡ High-frequency automated trade execution with conservative risk management")
    
    while True:
        try:
            executor.run_signal_execution_cycle()
            
            logger.info("⏰ Next execution cycle in 5 minutes...")
            time.sleep(300)  # Check every 5 minutes for new signals
            
        except KeyboardInterrupt:
            logger.info("🛑 Signal executor stopped by user")
            break
        except Exception as e:
            logger.error(f"Executor error: {e}")
            time.sleep(300)

if __name__ == "__main__":
    main()