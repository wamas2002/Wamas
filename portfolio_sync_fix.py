#!/usr/bin/env python3
import sqlite3
import os
import ccxt
from datetime import datetime

def fix_portfolio_sync():
    """Fix portfolio synchronization with OKX"""
    
    # Connect to OKX
    exchange = ccxt.okx({
        'apiKey': os.environ.get('OKX_API_KEY'),
        'secret': os.environ.get('OKX_SECRET_KEY'),
        'password': os.environ.get('OKX_PASSPHRASE'),
        'sandbox': False,
        'timeout': 8000
    })
    
    balance = exchange.fetch_balance()
    
    # Work with the correct database path
    db_path = '/home/runner/workspace/trading_platform.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing portfolio data
    cursor.execute("DELETE FROM portfolio")
    
    # Sync USDT balance
    usdt_balance = float(balance['USDT']['free'])
    cursor.execute('''
        INSERT INTO portfolio (symbol, quantity, avg_price, current_price)
        VALUES (?, ?, ?, ?)
    ''', ('USDT', usdt_balance, 1.0, 1.0))
    
    total_value = usdt_balance
    synced_tokens = 0
    
    # Sync crypto holdings
    tokens = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
    
    for token in tokens:
        if token in balance:
            amount = float(balance[token]['total'])
            if amount > 0:
                try:
                    ticker = exchange.fetch_ticker(f"{token}/USDT")
                    price = float(ticker['last'])
                    value = amount * price
                    
                    cursor.execute('''
                        INSERT INTO portfolio (symbol, quantity, avg_price, current_price)
                        VALUES (?, ?, ?, ?)
                    ''', (token, amount, price, price))
                    
                    total_value += value
                    synced_tokens += 1
                    print(f"Synced {token}: {amount:.6f} @ ${price:.2f}")
                    
                except Exception as e:
                    print(f"Error syncing {token}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"Portfolio sync complete:")
    print(f"Total Value: ${total_value:.2f}")
    print(f"USDT: ${usdt_balance:.2f}")
    print(f"Crypto tokens: {synced_tokens}")
    
    return total_value, synced_tokens

if __name__ == "__main__":
    try:
        fix_portfolio_sync()
    except Exception as e:
        print(f"Sync error: {e}")