#!/usr/bin/env python3
"""
Critical Database Infrastructure Fix
Rebuilds missing database tables and restores system functionality
"""

import sqlite3
import json
import os
import ccxt
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_database_infrastructure():
    """Fix all critical database issues identified in audit"""
    
    print("="*60)
    print("FIXING CRITICAL DATABASE INFRASTRUCTURE ISSUES")
    print("="*60)
    
    fixes_applied = []
    
    # 1. Create missing unified_signals table
    print("\n1. Creating unified_signals table...")
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unified_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rsi REAL,
                    macd REAL,
                    bollinger REAL,
                    volume_ratio REAL,
                    price REAL,
                    reasoning TEXT,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON unified_signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON unified_signals(symbol)")
            
            conn.commit()
            fixes_applied.append("✓ unified_signals table created with proper indexes")
            print("   ✓ unified_signals table created successfully")
            
    except Exception as e:
        print(f"   ✗ Failed to create unified_signals: {e}")
    
    # 2. Create missing portfolio_data table
    print("\n2. Creating portfolio_data table...")
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    balance REAL NOT NULL,
                    value_usd REAL NOT NULL,
                    price_usd REAL NOT NULL,
                    change_24h REAL,
                    allocation_percent REAL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio_data(symbol)")
            
            conn.commit()
            fixes_applied.append("✓ portfolio_data table created with proper indexes")
            print("   ✓ portfolio_data table created successfully")
            
    except Exception as e:
        print(f"   ✗ Failed to create portfolio_data: {e}")
    
    # 3. Enhance trading_performance table
    print("\n3. Enhancing trading_performance table...")
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            # Check if table exists and has proper structure
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='trading_performance'")
            result = cursor.fetchone()
            
            if not result:
                cursor.execute('''
                    CREATE TABLE trading_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        amount REAL NOT NULL,
                        price REAL NOT NULL,
                        profit_loss REAL,
                        timestamp TEXT NOT NULL,
                        trade_id TEXT,
                        fees REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trading_timestamp ON trading_performance(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trading_symbol ON trading_performance(symbol)")
                
                fixes_applied.append("✓ trading_performance table created")
                print("   ✓ trading_performance table created")
            else:
                print("   ✓ trading_performance table already exists")
            
            conn.commit()
            
    except Exception as e:
        print(f"   ✗ Failed to enhance trading_performance: {e}")
    
    # 4. Create risk management tables
    print("\n4. Creating risk management tables...")
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            # Stop losses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stop_losses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    stop_price REAL NOT NULL,
                    amount REAL NOT NULL,
                    active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    triggered_at DATETIME
                )
            ''')
            
            # Risk metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_portfolio_value REAL,
                    max_drawdown REAL,
                    var_1day REAL,
                    sharpe_ratio REAL,
                    diversification_score REAL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            fixes_applied.append("✓ Risk management tables created")
            print("   ✓ Risk management tables created")
            
    except Exception as e:
        print(f"   ✗ Failed to create risk tables: {e}")
    
    # 5. Populate tables with current data
    print("\n5. Populating tables with authentic current data...")
    
    # Populate with real OKX data if available
    if os.getenv('OKX_API_KEY'):
        try:
            exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Get current portfolio data
            balance = exchange.fetch_balance()
            current_time = datetime.now().isoformat()
            
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Clear old portfolio data
                cursor.execute("DELETE FROM portfolio_data WHERE timestamp < datetime('now', '-1 hour')")
                
                # Insert current portfolio data
                for symbol, amount in balance['total'].items():
                    if amount > 0:
                        try:
                            if symbol != 'USDT':
                                ticker = exchange.fetch_ticker(f'{symbol}/USDT')
                                price = ticker['last']
                            else:
                                price = 1.0
                            
                            value_usd = amount * price
                            
                            cursor.execute('''
                                INSERT INTO portfolio_data 
                                (symbol, balance, value_usd, price_usd, timestamp)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (symbol, amount, value_usd, price, current_time))
                            
                        except Exception as e:
                            # Skip symbols that can't be priced
                            continue
                
                conn.commit()
                fixes_applied.append("✓ Portfolio data populated with live OKX data")
                print("   ✓ Portfolio data populated with live OKX data")
                
        except Exception as e:
            print(f"   ⚠ Could not populate with OKX data: {e}")
    
    # 6. Generate initial AI signals to populate unified_signals
    print("\n6. Generating initial AI signals...")
    try:
        if os.getenv('OKX_API_KEY'):
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
            current_time = datetime.now().isoformat()
            
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                for symbol in symbols:
                    try:
                        # Get basic market data
                        ticker = exchange.fetch_ticker(symbol)
                        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=50)
                        
                        if ohlcv and len(ohlcv) >= 20:
                            # Simple signal generation based on price action
                            recent_prices = [candle[4] for candle in ohlcv[-20:]]  # Close prices
                            current_price = ticker['last']
                            
                            # Simple RSI-like calculation
                            price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
                            gains = [max(0, change) for change in price_changes]
                            losses = [abs(min(0, change)) for change in price_changes]
                            
                            avg_gain = sum(gains) / len(gains) if gains else 0
                            avg_loss = sum(losses) / len(losses) if losses else 1
                            
                            rs = avg_gain / avg_loss if avg_loss > 0 else 1
                            rsi = 100 - (100 / (1 + rs))
                            
                            # Generate signal based on RSI
                            if rsi < 35:
                                action = 'BUY'
                                confidence = min(85, 70 + (35 - rsi))
                                reasoning = f"Oversold condition: RSI {rsi:.1f}"
                            elif rsi > 65:
                                action = 'SELL'
                                confidence = min(85, 70 + (rsi - 65))
                                reasoning = f"Overbought condition: RSI {rsi:.1f}"
                            else:
                                action = 'HOLD'
                                confidence = 60 + abs(50 - rsi)
                                reasoning = f"Neutral condition: RSI {rsi:.1f}"
                            
                            # Volume analysis
                            recent_volumes = [candle[5] for candle in ohlcv[-10:]]
                            avg_volume = sum(recent_volumes) / len(recent_volumes)
                            current_volume = ticker.get('quoteVolume', avg_volume)
                            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                            
                            cursor.execute('''
                                INSERT INTO unified_signals 
                                (symbol, action, confidence, rsi, volume_ratio, price, reasoning, timestamp)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (symbol, action, confidence, rsi, volume_ratio, current_price, reasoning, current_time))
                            
                    except Exception as e:
                        print(f"   ⚠ Could not generate signal for {symbol}: {e}")
                        continue
                
                conn.commit()
                
                # Check how many signals were created
                cursor.execute("SELECT COUNT(*) FROM unified_signals WHERE timestamp = ?", (current_time,))
                signal_count = cursor.fetchone()[0]
                
                if signal_count > 0:
                    fixes_applied.append(f"✓ Generated {signal_count} AI signals")
                    print(f"   ✓ Generated {signal_count} AI signals")
                
    except Exception as e:
        print(f"   ⚠ Signal generation failed: {e}")
    
    # 7. Verify database health after fixes
    print("\n7. Verifying database health after fixes...")
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            tables_to_check = ['unified_signals', 'portfolio_data', 'trading_performance', 'stop_losses', 'risk_metrics']
            healthy_tables = 0
            total_records = 0
            
            for table in tables_to_check:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    healthy_tables += 1
                    total_records += count
                    print(f"   ✓ {table}: {count} records")
                except Exception as e:
                    print(f"   ✗ {table}: {e}")
            
            health_percentage = (healthy_tables / len(tables_to_check)) * 100
            fixes_applied.append(f"✓ Database health improved to {health_percentage:.0f}%")
            print(f"   ✓ Database health: {health_percentage:.0f}% ({total_records} total records)")
            
    except Exception as e:
        print(f"   ✗ Health verification failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("DATABASE INFRASTRUCTURE FIX SUMMARY")
    print("="*60)
    
    if fixes_applied:
        print(f"Successfully applied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"  {fix}")
    else:
        print("No fixes could be applied - manual intervention required")
    
    print(f"\nRecommendations after fixes:")
    print("1. Restart the Unified Trading Platform to use new database structure")
    print("2. Monitor signal generation for 1 hour to ensure continuous operation")
    print("3. Implement automated stop losses using the new risk management tables")
    print("4. Set up regular database maintenance and backup procedures")
    
    return len(fixes_applied)

if __name__ == "__main__":
    fixes_count = fix_database_infrastructure()
    print(f"\nDatabase fix completed with {fixes_count} successful fixes!")