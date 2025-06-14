"""
Check Trade Execution Status
Verify if signals are executing actual trades on OKX
"""
import sqlite3
import os
import json
from datetime import datetime, timedelta

def check_trade_execution():
    """Check if trades are being executed on OKX"""
    
    # Check all trading databases
    databases = [
        'enhanced_live_trading.db',
        'pure_local_trading.db', 
        'signal_execution.db',
        'advanced_futures.db'
    ]
    
    trade_summary = {
        'total_signals': 0,
        'total_trades': 0,
        'execution_rate': 0,
        'recent_trades': [],
        'databases_checked': []
    }
    
    for db_name in databases:
        if os.path.exists(db_name):
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Check for signals table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%signal%';")
                signal_tables = cursor.fetchall()
                
                # Check for trades table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%trade%';")
                trade_tables = cursor.fetchall()
                
                db_info = {
                    'database': db_name,
                    'signal_tables': [t[0] for t in signal_tables],
                    'trade_tables': [t[0] for t in trade_tables],
                    'signals_count': 0,
                    'trades_count': 0,
                    'recent_signals': [],
                    'recent_trades': []
                }
                
                # Count signals from each table
                for table_name in db_info['signal_tables']:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name[0]}")
                        count = cursor.fetchone()[0]
                        db_info['signals_count'] += count
                        
                        # Get recent signals
                        cursor.execute(f"""
                            SELECT * FROM {table_name[0]} 
                            ORDER BY timestamp DESC LIMIT 5
                        """)
                        recent = cursor.fetchall()
                        if recent:
                            # Get column names
                            cursor.execute(f"PRAGMA table_info({table_name[0]})")
                            columns = [col[1] for col in cursor.fetchall()]
                            
                            for row in recent:
                                signal_dict = dict(zip(columns, row))
                                db_info['recent_signals'].append(signal_dict)
                        
                    except Exception as e:
                        print(f"Error reading signals from {table_name[0]}: {e}")
                
                # Count trades from each table
                for table_name in db_info['trade_tables']:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name[0]}")
                        count = cursor.fetchone()[0]
                        db_info['trades_count'] += count
                        
                        # Get recent trades
                        cursor.execute(f"""
                            SELECT * FROM {table_name[0]} 
                            ORDER BY timestamp DESC LIMIT 5
                        """)
                        recent = cursor.fetchall()
                        if recent:
                            # Get column names
                            cursor.execute(f"PRAGMA table_info({table_name[0]})")
                            columns = [col[1] for col in cursor.fetchall()]
                            
                            for row in recent:
                                trade_dict = dict(zip(columns, row))
                                db_info['recent_trades'].append(trade_dict)
                                trade_summary['recent_trades'].append(trade_dict)
                    
                    except Exception as e:
                        print(f"Error reading trades from {table_name[0]}: {e}")
                
                trade_summary['total_signals'] += db_info['signals_count']
                trade_summary['total_trades'] += db_info['trades_count']
                trade_summary['databases_checked'].append(db_info)
                
                conn.close()
                
            except Exception as e:
                print(f"Error accessing {db_name}: {e}")
    
    # Calculate execution rate
    if trade_summary['total_signals'] > 0:
        trade_summary['execution_rate'] = (trade_summary['total_trades'] / trade_summary['total_signals']) * 100
    
    return trade_summary

def check_okx_balance():
    """Check current OKX balance to see if trades could be executed"""
    try:
        import ccxt
        
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        balance = exchange.fetch_balance()
        
        # Get USDT balance
        usdt_balance = balance.get('USDT', {})
        free_usdt = usdt_balance.get('free', 0)
        total_usdt = usdt_balance.get('total', 0)
        
        return {
            'free_usdt': free_usdt,
            'total_usdt': total_usdt,
            'can_trade': free_usdt >= 5,  # Minimum trade amount
            'connection_ok': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'connection_ok': False
        }

if __name__ == "__main__":
    print("=== TRADE EXECUTION STATUS CHECK ===")
    print(f"Time: {datetime.now()}")
    print()
    
    # Check trade execution
    execution_status = check_trade_execution()
    
    print("SIGNAL & TRADE SUMMARY:")
    print(f"Total Signals Generated: {execution_status['total_signals']}")
    print(f"Total Trades Executed: {execution_status['total_trades']}")
    print(f"Execution Rate: {execution_status['execution_rate']:.1f}%")
    print()
    
    print("DATABASE DETAILS:")
    for db_info in execution_status['databases_checked']:
        print(f"\nDatabase: {db_info['database']}")
        print(f"  Signal Tables: {db_info['signal_tables']}")
        print(f"  Trade Tables: {db_info['trade_tables']}")
        print(f"  Signals: {db_info['signals_count']}")
        print(f"  Trades: {db_info['trades_count']}")
        
        if db_info['recent_trades']:
            print(f"  Recent Trades:")
            for trade in db_info['recent_trades'][:3]:
                print(f"    - {trade}")
    
    print("\nOKX BALANCE CHECK:")
    balance_status = check_okx_balance()
    
    if balance_status.get('connection_ok'):
        print(f"Free USDT: ${balance_status['free_usdt']:.2f}")
        print(f"Total USDT: ${balance_status['total_usdt']:.2f}")
        print(f"Can Execute Trades: {'Yes' if balance_status['can_trade'] else 'No (insufficient balance)'}")
    else:
        print(f"Connection Error: {balance_status.get('error', 'Unknown error')}")
    
    print("\nTRADE EXECUTION ANALYSIS:")
    if execution_status['total_trades'] == 0:
        print("⚠️  NO TRADES EXECUTED")
        print("Possible reasons:")
        print("1. Insufficient USDT balance")
        print("2. Position sizing too large for available balance")
        print("3. API permissions not set for trading")
        print("4. Exchange connection issues")
        print("5. Signal confidence threshold too high")
        
        if not balance_status.get('can_trade', False):
            print("\n🔍 LIKELY CAUSE: Insufficient USDT balance for minimum trade size")
        
    elif execution_status['execution_rate'] < 5:
        print("⚠️  LOW EXECUTION RATE")
        print("Most signals are not being executed")
        
    else:
        print("✅ TRADES ARE BEING EXECUTED")
        print(f"Recent trade count: {len(execution_status['recent_trades'])}")
    
    print("\nRECOMMENDATIONS:")
    if not balance_status.get('can_trade', False):
        print("1. Add more USDT to your OKX account for trading")
        print("2. Reduce minimum trade size in trading parameters")
    
    if execution_status['execution_rate'] < 10:
        print("3. Lower confidence threshold from 70% to 65%")
        print("4. Reduce position size percentage")
        print("5. Check API trading permissions on OKX")