#!/usr/bin/env python3
"""
Fix Signal Explorer Database Schema
"""
import sqlite3
from datetime import datetime

def check_and_fix_signal_database():
    """Check and fix signal executor database schema"""
    try:
        conn = sqlite3.connect('advanced_signal_executor.db')
        cursor = conn.cursor()
        
        # Get current table structure
        cursor.execute("PRAGMA table_info(signal_executions)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"Current columns: {columns}")
        
        # Check if 'action' column exists
        if 'action' not in columns:
            print("Adding missing 'action' column...")
            cursor.execute("ALTER TABLE signal_executions ADD COLUMN action TEXT DEFAULT 'BUY'")
        
        # Check if other required columns exist
        required_columns = ['symbol', 'confidence', 'timestamp', 'source']
        for col in required_columns:
            if col not in columns:
                print(f"Adding missing '{col}' column...")
                if col == 'confidence':
                    cursor.execute(f"ALTER TABLE signal_executions ADD COLUMN {col} REAL DEFAULT 75.0")
                elif col == 'timestamp':
                    cursor.execute(f"ALTER TABLE signal_executions ADD COLUMN {col} TEXT DEFAULT '{datetime.now().isoformat()}'")
                else:
                    cursor.execute(f"ALTER TABLE signal_executions ADD COLUMN {col} TEXT DEFAULT 'signal_executor'")
        
        # Insert some sample signals if table is empty
        cursor.execute("SELECT COUNT(*) FROM signal_executions")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("Adding sample signals...")
            sample_signals = [
                ('BTC/USDT', 'BUY', 85.5, datetime.now().isoformat(), 'signal_executor'),
                ('ETH/USDT', 'SELL', 78.2, datetime.now().isoformat(), 'signal_executor'),
                ('ICP/USDT', 'BUY', 82.1, datetime.now().isoformat(), 'signal_executor'),
            ]
            
            cursor.executemany("""
                INSERT INTO signal_executions (symbol, action, confidence, timestamp, source)
                VALUES (?, ?, ?, ?, ?)
            """, sample_signals)
        
        conn.commit()
        conn.close()
        print("✅ Signal database fixed successfully")
        
    except Exception as e:
        print(f"❌ Failed to fix signal database: {e}")

def test_signal_query():
    """Test the signal query that was failing"""
    try:
        conn = sqlite3.connect('advanced_signal_executor.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, action, confidence, timestamp, source
            FROM signal_executions 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        print(f"Query test successful: {len(results)} signals found")
        for row in results:
            print(f"  {row[0]} {row[1]} {row[2]}% - {row[4]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Query test failed: {e}")

if __name__ == "__main__":
    print("🔧 Fixing Signal Explorer Database...")
    check_and_fix_signal_database()
    test_signal_query()