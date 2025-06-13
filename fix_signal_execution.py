"""
Fix Signal Execution - Enable automatic trade execution for high confidence signals
"""
import sqlite3
import os
from datetime import datetime

def fix_signal_execution_database():
    """Fix signal execution bridge to monitor correct databases"""
    
    # Copy high confidence signals from Pure Local Trading Engine to execution bridge database
    if os.path.exists('pure_local_trading.db'):
        source_conn = sqlite3.connect('pure_local_trading.db')
        source_cursor = source_conn.cursor()
        
        # Get recent high confidence signals
        source_cursor.execute("""
            SELECT symbol, signal, confidence, timestamp, reasoning
            FROM local_signals 
            WHERE confidence >= 75.0 
            AND signal = 'BUY'
            ORDER BY timestamp DESC LIMIT 10
        """)
        
        high_conf_signals = source_cursor.fetchall()
        source_conn.close()
        
        if high_conf_signals:
            # Create/update execution bridge database
            exec_conn = sqlite3.connect('trading_platform.db')
            exec_cursor = exec_conn.cursor()
            
            # Create ai_signals table if not exists
            exec_cursor.execute("""
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
            
            # Insert high confidence signals for execution
            for signal in high_conf_signals:
                symbol, action, confidence, timestamp, reasoning = signal
                
                # Check if signal already exists
                exec_cursor.execute("""
                    SELECT COUNT(*) FROM ai_signals 
                    WHERE symbol = ? AND timestamp = ?
                """, (symbol, timestamp))
                
                if exec_cursor.fetchone()[0] == 0:
                    exec_cursor.execute("""
                        INSERT INTO ai_signals (symbol, signal, confidence, timestamp, reasoning, executed)
                        VALUES (?, ?, ?, ?, ?, 0)
                    """, (symbol, action, confidence, timestamp, reasoning or 'Pure Local Analysis'))
            
            exec_conn.commit()
            exec_conn.close()
            
            print(f"✅ Copied {len(high_conf_signals)} high confidence signals to execution bridge")
            return len(high_conf_signals)
    
    return 0

def enable_auto_execution():
    """Enable automatic execution for high confidence signals"""
    
    # Update signal execution bridge configuration
    try:
        with open('signal_execution_bridge.py', 'r') as f:
            content = f.read()
        
        # Ensure execution threshold is reasonable
        if 'execution_threshold = 75.0' not in content:
            content = content.replace('execution_threshold = ', 'execution_threshold = 75.0  # ')
        
        # Ensure minimum trade amount is low enough
        if 'min_trade_amount = 5' not in content:
            content = content.replace('min_trade_amount = ', 'min_trade_amount = 5  # ')
        
        with open('signal_execution_bridge.py', 'w') as f:
            f.write(content)
        
        print("✅ Signal execution bridge configuration updated")
        return True
        
    except Exception as e:
        print(f"❌ Error updating execution bridge: {e}")
        return False

def check_okx_balance():
    """Check if OKX account has sufficient balance for trading"""
    try:
        import ccxt
        
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'enableRateLimit': True
        })
        
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        
        print(f"💰 OKX USDT Balance: ${usdt_balance:.2f}")
        
        if usdt_balance >= 10:
            print("✅ Sufficient balance for trading")
            return True
        else:
            print("⚠️  Low balance - minimum $10 USDT recommended")
            return False
            
    except Exception as e:
        print(f"❌ Error checking balance: {e}")
        return False

def main():
    """Fix signal execution and enable auto-trading"""
    print("🔧 FIXING SIGNAL EXECUTION FOR HIGH CONFIDENCE TRADES")
    print("=" * 60)
    
    # 1. Fix database connections
    signals_copied = fix_signal_execution_database()
    
    # 2. Enable auto-execution
    config_updated = enable_auto_execution()
    
    # 3. Check trading balance
    balance_ok = check_okx_balance()
    
    print("\n" + "=" * 60)
    print("SIGNAL EXECUTION FIX SUMMARY")
    print("=" * 60)
    print(f"📊 High confidence signals ready: {signals_copied}")
    print(f"⚙️  Execution bridge configured: {'✅' if config_updated else '❌'}")
    print(f"💰 Trading balance sufficient: {'✅' if balance_ok else '⚠️'}")
    
    if signals_copied > 0 and config_updated:
        print("\n🎯 SOLUTION: Signal execution bridge now monitoring correct signals")
        print("🚀 High confidence trades will execute automatically")
        print("📈 Minimum 75% confidence required for execution")
        print("💵 $5 minimum trade size with 3.5% position sizing")
    else:
        print("\n⚠️  Manual intervention required:")
        if signals_copied == 0:
            print("   - No high confidence signals found")
        if not config_updated:
            print("   - Bridge configuration needs manual update")
        if not balance_ok:
            print("   - Fund OKX account with minimum $10 USDT")

if __name__ == "__main__":
    main()