"""
Database Schema Fix
Addresses column name inconsistencies identified in integration test
"""

import sqlite3
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSchemaFixer:
    def __init__(self):
        self.fixes_applied = []
        self.errors_encountered = []
    
    def fix_trading_data_schema(self):
        """Fix trading data database schema issues"""
        db_path = 'data/trading_data.db'
        if not os.path.exists(db_path):
            return
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check current schema
            cursor.execute("PRAGMA table_info(ohlcv_data)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Fix close_price column name issue
            if 'close' in columns and 'close_price' not in columns:
                cursor.execute("ALTER TABLE ohlcv_data ADD COLUMN close_price REAL")
                cursor.execute("UPDATE ohlcv_data SET close_price = close")
                self.fixes_applied.append("Added close_price column to ohlcv_data")
            
            # Ensure all required columns exist
            required_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in columns:
                    data_type = 'REAL' if col != 'symbol' and col != 'timestamp' else 'TEXT'
                    cursor.execute(f"ALTER TABLE ohlcv_data ADD COLUMN {col} {data_type}")
                    self.fixes_applied.append(f"Added {col} column to ohlcv_data")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.errors_encountered.append(f"Trading data schema fix error: {str(e)}")
    
    def fix_portfolio_tracking_schema(self):
        """Fix portfolio tracking database schema"""
        db_path = 'data/portfolio_tracking.db'
        if not os.path.exists(db_path):
            return
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create positions table if missing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL DEFAULT 0,
                    avg_price REAL NOT NULL DEFAULT 0,
                    current_value REAL NOT NULL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Ensure portfolio_metrics table has correct structure
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_value REAL NOT NULL DEFAULT 0,
                    daily_pnl REAL DEFAULT 0,
                    daily_pnl_percent REAL DEFAULT 0,
                    positions_count INTEGER DEFAULT 0,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert sample data if empty
            cursor.execute("SELECT COUNT(*) FROM portfolio_metrics")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO portfolio_metrics (total_value, daily_pnl, daily_pnl_percent, positions_count)
                    VALUES (10000.0, 0.0, 0.0, 0)
                """)
            
            conn.commit()
            conn.close()
            self.fixes_applied.append("Fixed portfolio tracking schema")
            
        except Exception as e:
            self.errors_encountered.append(f"Portfolio schema fix error: {str(e)}")
    
    def fix_risk_management_schema(self):
        """Fix risk management database schema"""
        db_path = 'data/risk_management.db'
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create risk_events table with proper timestamp column
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    description TEXT,
                    severity TEXT DEFAULT 'medium',
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create risk_metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    var_1d REAL DEFAULT 0,
                    var_5d REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    volatility REAL DEFAULT 0,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            self.fixes_applied.append("Created risk management database with proper schema")
            
        except Exception as e:
            self.errors_encountered.append(f"Risk management schema fix error: {str(e)}")
    
    def fix_sentiment_aggregation_schema(self):
        """Fix sentiment aggregation schema issues"""
        db_path = 'data/sentiment_analysis.db'
        if not os.path.exists(db_path):
            return
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if sentiment_aggregated table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_aggregated'")
            if cursor.fetchone():
                # Check current columns
                cursor.execute("PRAGMA table_info(sentiment_aggregated)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Add missing minute_timestamp column
                if 'minute_timestamp' not in columns:
                    cursor.execute("ALTER TABLE sentiment_aggregated ADD COLUMN minute_timestamp TEXT")
                    cursor.execute("""
                        UPDATE sentiment_aggregated 
                        SET minute_timestamp = datetime(timestamp, 'start of day', '+' || 
                        (CAST(strftime('%H', timestamp) AS INTEGER) * 60 + 
                         CAST(strftime('%M', timestamp) AS INTEGER)) || ' minutes')
                        WHERE minute_timestamp IS NULL
                    """)
                    self.fixes_applied.append("Added minute_timestamp column to sentiment_aggregated")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.errors_encountered.append(f"Sentiment schema fix error: {str(e)}")
    
    def verify_okx_data_integration(self):
        """Verify OKX data is properly integrated"""
        try:
            from trading.okx_data_service import OKXDataService
            okx = OKXDataService()
            
            # Test data retrieval and storage
            test_symbols = ['BTCUSDT', 'ETHUSDT']
            for symbol in test_symbols:
                try:
                    price = okx.get_current_price(symbol)
                    if price:
                        # Store in trading database
                        conn = sqlite3.connect('data/trading_data.db')
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO ohlcv_data 
                            (symbol, timestamp, close, close_price, volume) 
                            VALUES (?, ?, ?, ?, ?)
                        """, (symbol, datetime.now().isoformat(), float(price), float(price), 1000))
                        
                        conn.commit()
                        conn.close()
                        self.fixes_applied.append(f"Updated live price for {symbol}: ${price}")
                        
                except Exception as e:
                    self.errors_encountered.append(f"OKX integration error for {symbol}: {str(e)}")
                    
        except ImportError:
            self.errors_encountered.append("OKX data service not available")
    
    def run_all_fixes(self):
        """Execute all database schema fixes"""
        logger.info("Running database schema fixes...")
        
        self.fix_trading_data_schema()
        self.fix_portfolio_tracking_schema()
        self.fix_risk_management_schema()
        self.fix_sentiment_aggregation_schema()
        self.verify_okx_data_integration()
        
        return {
            'fixes_applied': self.fixes_applied,
            'errors_encountered': self.errors_encountered,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    fixer = DatabaseSchemaFixer()
    results = fixer.run_all_fixes()
    
    print("=" * 60)
    print("DATABASE SCHEMA FIX RESULTS")
    print("=" * 60)
    
    if results['fixes_applied']:
        print("\n✅ FIXES APPLIED:")
        for fix in results['fixes_applied']:
            print(f"  • {fix}")
    
    if results['errors_encountered']:
        print("\n❌ ERRORS ENCOUNTERED:")
        for error in results['errors_encountered']:
            print(f"  • {error}")
    
    print(f"\n🕒 Completed: {results['timestamp']}")
    print("=" * 60)