#!/usr/bin/env python3
import sqlite3
import os
import ccxt

def check_portfolio_sync():
    """Quick portfolio sync check"""
    
    # Check database
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("Database tables:", [t[0] for t in tables])
        
        # Check portfolio data
        if ('portfolio',) in tables:
            cursor.execute("SELECT COUNT(*) FROM portfolio")
            count = cursor.fetchone()[0]
            print(f"Portfolio entries: {count}")
            
            cursor.execute("SELECT * FROM portfolio LIMIT 5")
            data = cursor.fetchall()
            for row in data:
                print(f"  {row}")
        
        conn.close()
        
    except Exception as e:
        print(f"Database error: {e}")
    
    # Test OKX with timeout
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'timeout': 5000,  # 5 second timeout
        })
        
        print("Testing OKX connection...")
        balance = exchange.fetch_balance()
        
        # Check USDT
        usdt = balance.get('USDT', {})
        print(f"USDT free: {usdt.get('free', 0)}")
        
        # Check crypto
        tokens = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        for token in tokens:
            if token in balance:
                amount = balance[token].get('total', 0)
                if amount > 0:
                    print(f"{token}: {amount}")
        
    except Exception as e:
        print(f"OKX connection issue: {e}")

if __name__ == "__main__":
    check_portfolio_sync()