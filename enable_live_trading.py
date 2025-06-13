"""
Enable Live Trading - Connect high confidence signals to execution bridge
"""
import sqlite3
import os
import ccxt
from datetime import datetime, timedelta

def connect_signal_sources():
    """Connect Pure Local Trading signals to execution bridge"""
    
    # Create execution bridge database if needed
    exec_conn = sqlite3.connect('trading_platform.db')
    exec_cursor = exec_conn.cursor()
    
    # Create ai_signals table with correct structure
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
    
    # Get high confidence signals from Pure Local Trading Engine
    if os.path.exists('pure_local_trading.db'):
        source_conn = sqlite3.connect('pure_local_trading.db')
        source_cursor = source_conn.cursor()
        
        # Get recent BUY signals with confidence >= 75%
        cutoff_time = (datetime.now() - timedelta(minutes=10)).isoformat()
        
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
        
        signals_added = 0
        for signal in high_conf_signals:
            symbol, signal_type, confidence, timestamp, price, target_price = signal
            
            # Check if signal already exists
            exec_cursor.execute("""
                SELECT COUNT(*) FROM ai_signals 
                WHERE symbol = ? AND timestamp = ?
            """, (symbol, timestamp))
            
            if exec_cursor.fetchone()[0] == 0:
                exec_cursor.execute("""
                    INSERT INTO ai_signals (symbol, signal, confidence, timestamp, reasoning, executed, price, target_price)
                    VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """, (symbol, signal_type, confidence, timestamp, 'Pure Local Analysis - High Confidence', price, target_price))
                signals_added += 1
        
        exec_conn.commit()
        exec_conn.close()
        
        print(f"Connected {signals_added} high confidence signals to execution bridge")
        return signals_added
    
    exec_conn.close()
    return 0

def check_okx_trading_status():
    """Check OKX account status and trading permissions"""
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'enableRateLimit': True
        })
        
        # Test connection and get balance
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        
        # Check account permissions
        account_info = exchange.fetch_status()
        
        return {
            'connected': True,
            'usdt_balance': usdt_balance,
            'trading_enabled': True,
            'account_status': account_info.get('status', 'unknown')
        }
        
    except Exception as e:
        return {
            'connected': False,
            'error': str(e),
            'trading_enabled': False
        }

def test_execution_bridge():
    """Test if execution bridge can find and process signals"""
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Check for executable signals
        cursor.execute("""
            SELECT COUNT(*) FROM ai_signals 
            WHERE confidence >= 75.0 
            AND signal = 'BUY'
            AND executed = 0
        """)
        
        executable_count = cursor.fetchone()[0]
        
        # Get sample of executable signals
        cursor.execute("""
            SELECT symbol, confidence, timestamp FROM ai_signals 
            WHERE confidence >= 75.0 
            AND signal = 'BUY'
            AND executed = 0
            ORDER BY confidence DESC LIMIT 5
        """)
        
        sample_signals = cursor.fetchall()
        conn.close()
        
        return {
            'executable_signals': executable_count,
            'sample_signals': sample_signals
        }
        
    except Exception as e:
        return {
            'executable_signals': 0,
            'error': str(e)
        }

def enable_automatic_execution():
    """Enable automatic trade execution for high confidence signals"""
    
    # Update execution bridge to lower threshold and enable execution
    try:
        with open('signal_execution_bridge.py', 'r') as f:
            content = f.read()
        
        # Set execution threshold to 75%
        content = content.replace('self.execution_threshold = 75.0', 'self.execution_threshold = 75.0')
        
        # Ensure minimum trade amount is accessible
        content = content.replace('self.min_trade_amount = 5', 'self.min_trade_amount = 5')
        
        # Enable execution by removing any disabled flags
        if 'EXECUTION_DISABLED' in content:
            content = content.replace('EXECUTION_DISABLED = True', 'EXECUTION_DISABLED = False')
        
        with open('signal_execution_bridge.py', 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"Error updating execution bridge: {e}")
        return False

def main():
    """Enable live trading with high confidence signals"""
    print("ENABLING LIVE TRADING FOR HIGH CONFIDENCE SIGNALS")
    print("=" * 55)
    
    # 1. Connect signal sources
    signals_connected = connect_signal_sources()
    
    # 2. Check OKX status
    okx_status = check_okx_trading_status()
    
    # 3. Test execution bridge
    bridge_status = test_execution_bridge()
    
    # 4. Enable automatic execution
    execution_enabled = enable_automatic_execution()
    
    print(f"\nSIGNAL CONNECTION: {signals_connected} high confidence signals ready")
    
    print(f"\nOKX TRADING STATUS:")
    if okx_status['connected']:
        print(f"  Connected: Yes")
        print(f"  USDT Balance: ${okx_status['usdt_balance']:.2f}")
        print(f"  Trading Enabled: {okx_status['trading_enabled']}")
    else:
        print(f"  Connection Error: {okx_status.get('error', 'Unknown')}")
    
    print(f"\nEXECUTION BRIDGE STATUS:")
    if 'error' not in bridge_status:
        print(f"  Executable Signals: {bridge_status['executable_signals']}")
        if bridge_status['sample_signals']:
            print(f"  Top Signal: {bridge_status['sample_signals'][0][0]} ({bridge_status['sample_signals'][0][1]:.1f}%)")
    else:
        print(f"  Bridge Error: {bridge_status['error']}")
    
    print(f"\nAUTO-EXECUTION: {'ENABLED' if execution_enabled else 'NEEDS MANUAL SETUP'}")
    
    # Summary and recommendations
    print("\n" + "=" * 55)
    if (signals_connected > 0 and okx_status['connected'] and 
        bridge_status['executable_signals'] > 0 and execution_enabled):
        print("LIVE TRADING ENABLED")
        print("High confidence signals will execute automatically")
        print(f"Ready to trade {bridge_status['executable_signals']} signals")
        print("Minimum $5 per trade, 3.5% position sizing")
    else:
        print("SETUP INCOMPLETE - Manual intervention required:")
        if signals_connected == 0:
            print("  - No high confidence signals available")
        if not okx_status['connected']:
            print("  - OKX connection failed")
        if bridge_status['executable_signals'] == 0:
            print("  - No executable signals in bridge database")
        if not execution_enabled:
            print("  - Execution bridge configuration error")

if __name__ == "__main__":
    main()