"""
Enable Auto Trading - Final implementation for high confidence signal execution
"""
import sqlite3
import os
import time
from datetime import datetime, timedelta

def create_signal_transfer_bridge():
    """Create automatic signal transfer from Pure Local Engine to Execution Bridge"""
    
    def transfer_signals():
        """Transfer high confidence signals for execution"""
        try:
            # Get recent high confidence signals from Pure Local Engine
            source_conn = sqlite3.connect('pure_local_trading.db')
            source_cursor = source_conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(minutes=5)).isoformat()
            
            source_cursor.execute("""
                SELECT symbol, signal_type, confidence, timestamp, price, target_price
                FROM local_signals 
                WHERE signal_type = 'BUY' 
                AND confidence >= 75.0 
                AND timestamp > ?
                ORDER BY confidence DESC LIMIT 10
            """, (cutoff_time,))
            
            high_conf_signals = source_cursor.fetchall()
            source_conn.close()
            
            if not high_conf_signals:
                return 0
            
            # Create execution database if needed
            exec_conn = sqlite3.connect('trading_platform.db')
            exec_cursor = exec_conn.cursor()
            
            exec_cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_signals (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    signal TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    reasoning TEXT,
                    executed BOOLEAN DEFAULT 0,
                    price REAL,
                    target_price REAL
                )
            """)
            
            transferred = 0
            for signal in high_conf_signals:
                symbol, signal_type, confidence, timestamp, price, target_price = signal
                
                # Check if already exists
                exec_cursor.execute("""
                    SELECT COUNT(*) FROM ai_signals 
                    WHERE symbol = ? AND timestamp = ?
                """, (symbol, timestamp))
                
                if exec_cursor.fetchone()[0] == 0:
                    exec_cursor.execute("""
                        INSERT OR REPLACE INTO ai_signals 
                        (symbol, signal, confidence, timestamp, reasoning, executed, price, target_price)
                        VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                    """, (symbol, signal_type, confidence, timestamp, 
                          'Pure Local High Confidence Analysis', price, target_price))
                    transferred += 1
            
            exec_conn.commit()
            exec_conn.close()
            
            return transferred
            
        except Exception as e:
            print(f"Transfer error: {e}")
            return 0
    
    return transfer_signals()

def enable_execution_bridge():
    """Enable the signal execution bridge with proper configuration"""
    
    # Update execution bridge to process signals automatically
    try:
        with open('signal_execution_bridge.py', 'r') as f:
            content = f.read()
        
        # Ensure execution is enabled
        if 'EXECUTION_DISABLED' in content:
            content = content.replace('EXECUTION_DISABLED = True', 'EXECUTION_DISABLED = False')
        
        # Add auto-execution flag if not present
        if 'auto_execute = True' not in content:
            content = content.replace(
                'self.min_trade_amount = 5',
                'self.min_trade_amount = 5\n        self.auto_execute = True  # Enable automatic execution'
            )
        
        with open('signal_execution_bridge.py', 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"Configuration error: {e}")
        return False

def test_execution_capability():
    """Test if execution bridge can access signals"""
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM ai_signals 
            WHERE confidence >= 75.0 
            AND signal = 'BUY'
            AND executed = 0
        """)
        
        executable_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT symbol, confidence, timestamp FROM ai_signals 
            WHERE confidence >= 75.0 
            AND signal = 'BUY'
            AND executed = 0
            ORDER BY confidence DESC LIMIT 3
        """)
        
        top_signals = cursor.fetchall()
        conn.close()
        
        return executable_count, top_signals
        
    except Exception as e:
        return 0, []

def main():
    """Enable automatic trading for high confidence signals"""
    print("ENABLING AUTOMATIC TRADE EXECUTION")
    print("=" * 40)
    
    # Transfer recent high confidence signals
    transferred = create_signal_transfer_bridge()
    print(f"Signals transferred: {transferred}")
    
    # Enable execution bridge
    bridge_enabled = enable_execution_bridge()
    print(f"Execution bridge enabled: {bridge_enabled}")
    
    # Test execution capability
    executable_count, top_signals = test_execution_capability()
    print(f"Executable signals ready: {executable_count}")
    
    if top_signals:
        print("Top executable signals:")
        for symbol, confidence, timestamp in top_signals:
            print(f"  {symbol}: {confidence:.1f}% confidence")
    
    print("\n" + "=" * 40)
    if executable_count > 0 and bridge_enabled:
        print("AUTO-TRADING ENABLED")
        print("High confidence signals will execute automatically")
        print(f"Ready to trade {executable_count} signals ≥75% confidence")
        print("Minimum $5 per trade, 3.5% position sizing")
    else:
        print("SETUP INCOMPLETE")
        if executable_count == 0:
            print("- No executable signals available")
        if not bridge_enabled:
            print("- Bridge configuration needs update")
    
    return executable_count

if __name__ == "__main__":
    main()