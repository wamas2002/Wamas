"""
Direct Execution Fix - Immediate solution for high confidence signal execution
"""
import sqlite3
import os
from datetime import datetime, timedelta

def create_execution_bridge_database():
    """Create execution bridge database with proper structure"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_signals (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            signal TEXT,
            confidence REAL,
            timestamp TEXT,
            reasoning TEXT,
            executed BOOLEAN DEFAULT 0
        )
    """)
    
    conn.commit()
    conn.close()

def transfer_high_confidence_signals():
    """Transfer high confidence signals from Pure Local Engine to execution bridge"""
    
    # Create bridge database if needed
    create_execution_bridge_database()
    
    # Get high confidence signals from Pure Local Trading Engine
    source_conn = sqlite3.connect('pure_local_trading.db')
    source_cursor = source_conn.cursor()
    
    # Get recent BUY signals with confidence >= 75%
    source_cursor.execute("""
        SELECT symbol, signal_type, confidence, timestamp
        FROM local_signals 
        WHERE signal_type = 'BUY' 
        AND confidence >= 75.0 
        ORDER BY timestamp DESC LIMIT 20
    """)
    
    signals = source_cursor.fetchall()
    source_conn.close()
    
    # Insert into execution bridge database
    exec_conn = sqlite3.connect('trading_platform.db')
    exec_cursor = exec_conn.cursor()
    
    transferred = 0
    for symbol, signal_type, confidence, timestamp in signals:
        # Check if already exists
        exec_cursor.execute("""
            SELECT COUNT(*) FROM ai_signals 
            WHERE symbol = ? AND timestamp = ?
        """, (symbol, timestamp))
        
        if exec_cursor.fetchone()[0] == 0:
            exec_cursor.execute("""
                INSERT INTO ai_signals (symbol, signal, confidence, timestamp, reasoning, executed)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (symbol, signal_type, confidence, timestamp, 'Pure Local High Confidence'))
            transferred += 1
    
    exec_conn.commit()
    exec_conn.close()
    
    return transferred

def verify_execution_ready():
    """Verify execution bridge has signals ready for trading"""
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM ai_signals 
            WHERE confidence >= 75.0 
            AND signal = 'BUY'
            AND executed = 0
        """)
        
        ready_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT symbol, confidence FROM ai_signals 
            WHERE confidence >= 75.0 
            AND signal = 'BUY'
            AND executed = 0
            ORDER BY confidence DESC LIMIT 5
        """)
        
        top_signals = cursor.fetchall()
        conn.close()
        
        return ready_count, top_signals
        
    except Exception as e:
        return 0, []

def main():
    """Execute direct fix for signal execution"""
    print("DIRECT EXECUTION FIX - Connecting High Confidence Signals")
    print("=" * 58)
    
    # Transfer signals
    transferred = transfer_high_confidence_signals()
    
    # Verify ready for execution
    ready_count, top_signals = verify_execution_ready()
    
    print(f"Signals transferred to execution bridge: {transferred}")
    print(f"Signals ready for execution: {ready_count}")
    
    if top_signals:
        print("\nTop signals ready for execution:")
        for symbol, confidence in top_signals:
            print(f"  - {symbol}: {confidence:.1f}% confidence")
    
    print("\n" + "=" * 58)
    if ready_count > 0:
        print("EXECUTION BRIDGE READY")
        print(f"{ready_count} high confidence signals available for trading")
        print("Signal Execution Bridge will process these automatically")
    else:
        print("NO EXECUTABLE SIGNALS FOUND")
        print("Waiting for new high confidence signals from Pure Local Engine")
    
    return ready_count

if __name__ == "__main__":
    main()