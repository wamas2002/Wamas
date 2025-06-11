#!/usr/bin/env python3
"""
Portfolio Sync Fix - Updates database with real OKX data
"""

import sqlite3
import os
import ccxt
from datetime import datetime

def sync_okx_to_database():
    """Sync OKX portfolio to database"""
    
    # Connect to OKX
    exchange = ccxt.okx({
        'apiKey': os.environ.get('OKX_API_KEY'),
        'secret': os.environ.get('OKX_SECRET_KEY'),
        'password': os.environ.get('OKX_PASSPHRASE'),
        'sandbox': False,
        'timeout': 10000
    })
    
    balance = exchange.fetch_balance()
    
    # Connect to database
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    # Clear existing portfolio data
    cursor.execute("DELETE FROM portfolio")
    
    # Insert USDT balance
    usdt_balance = float(balance['USDT']['free'])
    cursor.execute('''
        INSERT INTO portfolio (symbol, amount, value, price, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', ('USDT', usdt_balance, usdt_balance, 1.0, datetime.now().isoformat()))
    
    total_value = usdt_balance
    holdings_count = 0
    
    # Process crypto holdings
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
                        INSERT INTO portfolio (symbol, amount, value, price, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (token, amount, value, price, datetime.now().isoformat()))
                    
                    total_value += value
                    holdings_count += 1
                    print(f"Synced {token}: {amount:.6f} @ ${price:.2f} = ${value:.2f}")
                    
                except Exception as e:
                    print(f"Error syncing {token}: {e}")
    
    # Update live_trading database
    live_conn = sqlite3.connect('live_trading.db')
    live_cursor = live_conn.cursor()
    
    # Create portfolio_sync table
    live_cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_sync (
            id INTEGER PRIMARY KEY,
            total_value REAL,
            usdt_balance REAL,
            crypto_holdings INTEGER,
            last_sync TEXT
        )
    ''')
    
    # Clear and insert current sync data
    live_cursor.execute("DELETE FROM portfolio_sync")
    live_cursor.execute('''
        INSERT INTO portfolio_sync (total_value, usdt_balance, crypto_holdings, last_sync)
        VALUES (?, ?, ?, ?)
    ''', (total_value, usdt_balance, holdings_count, datetime.now().isoformat()))
    
    # Commit changes
    conn.commit()
    live_conn.commit()
    
    conn.close()
    live_conn.close()
    
    print(f"\nPortfolio sync complete:")
    print(f"Total Value: ${total_value:.2f}")
    print(f"USDT Balance: ${usdt_balance:.2f}")
    print(f"Crypto Holdings: {holdings_count} tokens")
    
    return True

if __name__ == "__main__":
    try:
        sync_okx_to_database()
        print("Portfolio synchronization successful!")
    except Exception as e:
        print(f"Sync failed: {e}")