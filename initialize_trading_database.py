#!/usr/bin/env python3
"""
Initialize Trading Database
Creates missing tables and sample data for the unified trading platform
"""

import sqlite3
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_trading_performance():
    """Create trading_performance table with sample data"""
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            # Create trading performance table
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
            
            # Check if table has data
            cursor.execute("SELECT COUNT(*) FROM trading_performance")
            count = cursor.fetchone()[0]
            
            if count == 0:
                logger.info("Adding sample trading data...")
                
                # Generate realistic trading data for the past week
                base_time = datetime.now() - timedelta(days=7)
                sample_trades = []
                
                # BTC trades
                sample_trades.extend([
                    ('BTC/USDT', 'buy', 0.001, 108500.0, 125.50, (base_time + timedelta(hours=1)).isoformat()),
                    ('BTC/USDT', 'sell', 0.0008, 108800.0, 67.80, (base_time + timedelta(hours=6)).isoformat()),
                    ('BTC/USDT', 'buy', 0.0012, 108300.0, -24.60, (base_time + timedelta(days=1)).isoformat()),
                    ('BTC/USDT', 'sell', 0.001, 108950.0, 89.30, (base_time + timedelta(days=2)).isoformat()),
                ])
                
                # ETH trades  
                sample_trades.extend([
                    ('ETH/USDT', 'buy', 0.1, 2780.0, 22.40, (base_time + timedelta(hours=3)).isoformat()),
                    ('ETH/USDT', 'sell', 0.08, 2770.0, -8.20, (base_time + timedelta(hours=8)).isoformat()),
                    ('ETH/USDT', 'buy', 0.15, 2760.0, 45.60, (base_time + timedelta(days=1, hours=2)).isoformat()),
                ])
                
                # SOL trades
                sample_trades.extend([
                    ('SOL/USDT', 'buy', 1.0, 160.0, 8.75, (base_time + timedelta(hours=4)).isoformat()),
                    ('SOL/USDT', 'sell', 0.5, 162.0, 18.90, (base_time + timedelta(days=1, hours=5)).isoformat()),
                ])
                
                # ADA trades
                sample_trades.extend([
                    ('ADA/USDT', 'buy', 100.0, 0.45, -12.30, (base_time + timedelta(hours=12)).isoformat()),
                    ('ADA/USDT', 'sell', 80.0, 0.46, 8.40, (base_time + timedelta(days=2, hours=3)).isoformat()),
                ])
                
                # DOT trades
                sample_trades.extend([
                    ('DOT/USDT', 'buy', 5.0, 6.80, 15.60, (base_time + timedelta(hours=18)).isoformat()),
                    ('DOT/USDT', 'sell', 3.0, 6.85, 12.50, (base_time + timedelta(days=3)).isoformat()),
                ])
                
                cursor.executemany('''
                    INSERT INTO trading_performance (symbol, side, size, price, profit_loss, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', sample_trades)
                
                logger.info(f"Added {len(sample_trades)} sample trades")
            
            conn.commit()
            logger.info("Trading performance database initialized successfully")
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

def initialize_live_trades():
    """Create live_trades table for ML training"""
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            # Create live_trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add sample data if empty
            cursor.execute("SELECT COUNT(*) FROM live_trades")
            if cursor.fetchone()[0] == 0:
                base_time = datetime.now() - timedelta(days=30)
                sample_live_trades = []
                
                # Generate 50 sample trades over 30 days
                import random
                symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'AVAX/USDT']
                
                for i in range(50):
                    symbol = random.choice(symbols)
                    side = random.choice(['buy', 'sell'])
                    
                    if 'BTC' in symbol:
                        amount = round(random.uniform(0.0001, 0.01), 6)
                        price = round(random.uniform(105000, 110000), 2)
                    elif 'ETH' in symbol:
                        amount = round(random.uniform(0.01, 0.5), 4)
                        price = round(random.uniform(2700, 2800), 2)
                    elif 'SOL' in symbol:
                        amount = round(random.uniform(0.1, 5.0), 2)
                        price = round(random.uniform(155, 165), 2)
                    elif 'ADA' in symbol:
                        amount = round(random.uniform(10, 500), 1)
                        price = round(random.uniform(0.40, 0.50), 4)
                    elif 'DOT' in symbol:
                        amount = round(random.uniform(1, 20), 1)
                        price = round(random.uniform(6.5, 7.0), 3)
                    else:  # AVAX
                        amount = round(random.uniform(0.5, 10), 2)
                        price = round(random.uniform(35, 45), 2)
                    
                    fee = amount * price * 0.001  # 0.1% fee
                    timestamp = (base_time + timedelta(hours=random.randint(0, 720))).isoformat()
                    
                    sample_live_trades.append((symbol, side, amount, price, fee, timestamp))
                
                cursor.executemany('''
                    INSERT INTO live_trades (symbol, side, amount, price, fee, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', sample_live_trades)
                
                logger.info(f"Added {len(sample_live_trades)} sample live trades")
            
            conn.commit()
            logger.info("Live trades database initialized successfully")
            
    except Exception as e:
        logger.error(f"Live trades initialization error: {e}")

if __name__ == '__main__':
    logger.info("Initializing trading databases...")
    initialize_trading_performance()
    initialize_live_trades()
    logger.info("Database initialization complete")