#!/usr/bin/env python3
"""
Initialize Complete Trading Database Schema
Creates all required tables for the unified trading platform
"""

import sqlite3
import logging
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_enhanced_trading_tables():
    """Create all enhanced trading database tables"""
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            # Create live_trades table for ML training
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL DEFAULT 0,
                    profit_loss REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create trading_performance table  
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    price REAL NOT NULL,
                    profit_loss REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert sample live trades for ML training
            cursor.execute("SELECT COUNT(*) FROM live_trades")
            if cursor.fetchone()[0] == 0:
                # Generate realistic trading data
                symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
                base_prices = {'BTC': 108500, 'ETH': 2770, 'SOL': 160, 'ADA': 0.45, 'DOT': 6.85, 'AVAX': 42}
                
                trades = []
                for i in range(50):  # Generate 50 sample trades
                    symbol = random.choice(symbols)
                    side = random.choice(['buy', 'sell'])
                    amount = random.uniform(0.01, 2.0)
                    price_variation = random.uniform(0.95, 1.05)
                    price = base_prices[symbol] * price_variation
                    fee = amount * price * 0.001  # 0.1% fee
                    profit_loss = random.uniform(-50, 100)
                    
                    # Generate timestamp within last 30 days
                    days_ago = random.randint(0, 30)
                    timestamp = datetime.now() - timedelta(days=days_ago)
                    
                    trades.append((symbol, side, amount, price, fee, profit_loss, timestamp))
                
                cursor.executemany('''
                    INSERT INTO live_trades (symbol, side, amount, price, fee, profit_loss, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', trades)
                
                logger.info(f"Added {len(trades)} sample trades for ML training")
            
            # Insert sample trading performance data
            cursor.execute("SELECT COUNT(*) FROM trading_performance")
            if cursor.fetchone()[0] == 0:
                performance_data = [
                    ('BTC', 'buy', 0.00892, 108594.50, 156.75),
                    ('ETH', 'buy', 0.5421, 2771.90, 89.30),
                    ('SOL', 'sell', 2.156, 160.73, -23.45),
                    ('ADA', 'buy', 125.45, 0.452, 12.80),
                    ('DOT', 'buy', 15.78, 6.85, 34.20),
                    ('AVAX', 'sell', 3.42, 42.15, -18.50),
                    ('BTC', 'sell', 0.00234, 108123.00, 78.90),
                    ('ETH', 'buy', 0.3214, 2755.40, 45.67),
                    ('SOL', 'buy', 1.789, 159.85, 67.89),
                    ('ADA', 'sell', 89.12, 0.448, -15.23),
                    ('DOT', 'sell', 8.45, 6.92, 23.45),
                    ('AVAX', 'buy', 2.87, 41.95, 56.78),
                    ('BTC', 'buy', 0.00156, 109012.00, 189.34)
                ]
                
                cursor.executemany('''
                    INSERT INTO trading_performance (symbol, side, size, price, profit_loss)
                    VALUES (?, ?, ?, ?, ?)
                ''', performance_data)
                
                logger.info(f"Added {len(performance_data)} performance records")
            
            conn.commit()
            logger.info("Enhanced trading database initialized successfully")
            
    except Exception as e:
        logger.error(f"Enhanced database initialization error: {e}")

def create_unified_trading_tables():
    """Create unified trading platform tables"""
    try:
        with sqlite3.connect('unified_trading.db') as conn:
            cursor = conn.cursor()
            
            # Create unified_signals table with proper schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unified_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 75.0,
                    current_price REAL NOT NULL DEFAULT 0.0,
                    target_price REAL NOT NULL DEFAULT 0.0,
                    reasoning TEXT DEFAULT 'Enhanced AI analysis',
                    rsi REAL DEFAULT 50.0,
                    macd REAL DEFAULT 0.0,
                    volume_ratio REAL DEFAULT 1.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create unified_portfolio table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unified_portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    balance REAL NOT NULL DEFAULT 0.0,
                    value_usd REAL NOT NULL DEFAULT 0.0,
                    percentage REAL NOT NULL DEFAULT 0.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("Unified trading database initialized successfully")
            
    except Exception as e:
        logger.error(f"Unified database initialization error: {e}")

def main():
    """Initialize all trading databases"""
    logger.info("=== INITIALIZING COMPLETE TRADING DATABASE ===")
    
    create_enhanced_trading_tables()
    create_unified_trading_tables()
    
    logger.info("=== DATABASE INITIALIZATION COMPLETE ===")
    logger.info("All required tables created:")
    logger.info("- live_trades (for ML training)")
    logger.info("- trading_performance (for performance tracking)")
    logger.info("- unified_signals (for signal storage)")
    logger.info("- unified_portfolio (for portfolio data)")

if __name__ == '__main__':
    main()