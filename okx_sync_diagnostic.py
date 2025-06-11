#!/usr/bin/env python3
"""
OKX Portfolio Synchronization Diagnostic Tool
Identifies and fixes portfolio sync issues with OKX account
"""

import os
import ccxt
import sqlite3
from datetime import datetime
import json

def test_okx_connection():
    """Test OKX connection and credentials"""
    print("🔍 Testing OKX Connection...")
    
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'rateLimit': 2000,
            'enableRateLimit': True,
        })
        
        # Test connection
        balance = exchange.fetch_balance()
        print("✅ OKX Connection: SUCCESS")
        print(f"📊 Account Status: {balance.get('info', {}).get('totalEq', 'N/A')}")
        
        return exchange, balance
        
    except Exception as e:
        print(f"❌ OKX Connection: FAILED - {e}")
        return None, None

def analyze_okx_portfolio(exchange, balance):
    """Analyze actual OKX portfolio"""
    print("\n🔍 Analyzing OKX Portfolio...")
    
    try:
        # Get USDT balance
        usdt_balance = float(balance['USDT']['free']) if 'USDT' in balance else 0
        usdt_total = float(balance['USDT']['total']) if 'USDT' in balance else 0
        
        print(f"💰 USDT Free: ${usdt_balance:.2f}")
        print(f"💰 USDT Total: ${usdt_total:.2f}")
        
        # Check crypto holdings
        crypto_holdings = []
        tokens = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        
        for token in tokens:
            if token in balance:
                free_amount = float(balance[token]['free'])
                total_amount = float(balance[token]['total'])
                
                if total_amount > 0:
                    # Get current price
                    try:
                        ticker = exchange.fetch_ticker(f"{token}/USDT")
                        price = float(ticker['last'])
                        value = total_amount * price
                        
                        crypto_holdings.append({
                            'token': token,
                            'free': free_amount,
                            'total': total_amount,
                            'price': price,
                            'value': value
                        })
                        
                        print(f"🪙 {token}: {total_amount:.6f} (${value:.2f})")
                        
                    except Exception as e:
                        print(f"⚠️ {token}: Price fetch error - {e}")
        
        total_crypto_value = sum(h['value'] for h in crypto_holdings)
        total_portfolio_value = usdt_balance + total_crypto_value
        
        print(f"\n📈 Total Crypto Value: ${total_crypto_value:.2f}")
        print(f"📈 Total Portfolio Value: ${total_portfolio_value:.2f}")
        
        return {
            'usdt_balance': usdt_balance,
            'crypto_holdings': crypto_holdings,
            'total_value': total_portfolio_value
        }
        
    except Exception as e:
        print(f"❌ Portfolio Analysis Failed: {e}")
        return None

def check_database_portfolio():
    """Check what's stored in the database"""
    print("\n🔍 Checking Database Portfolio...")
    
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Check portfolio table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio'")
        if cursor.fetchone():
            cursor.execute("SELECT * FROM portfolio ORDER BY timestamp DESC LIMIT 5")
            portfolio_data = cursor.fetchall()
            
            if portfolio_data:
                print("📊 Recent Portfolio Entries:")
                for row in portfolio_data:
                    print(f"   {row}")
            else:
                print("⚠️ No portfolio data in database")
        else:
            print("❌ Portfolio table does not exist")
        
        # Check live trades
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='live_trades'")
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) FROM live_trades")
            trade_count = cursor.fetchone()[0]
            print(f"📈 Total Trades in DB: {trade_count}")
            
            cursor.execute("SELECT * FROM live_trades ORDER BY timestamp DESC LIMIT 3")
            recent_trades = cursor.fetchall()
            
            if recent_trades:
                print("🔄 Recent Trades:")
                for trade in recent_trades:
                    print(f"   {trade}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database Check Failed: {e}")

def sync_portfolio_to_database(portfolio_data):
    """Sync OKX portfolio data to database"""
    print("\n🔄 Syncing Portfolio to Database...")
    
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Create portfolio table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                amount REAL,
                value REAL,
                price REAL,
                timestamp TEXT
            )
        ''')
        
        # Clear old data
        cursor.execute("DELETE FROM portfolio")
        
        # Insert USDT balance
        cursor.execute('''
            INSERT INTO portfolio (symbol, amount, value, price, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', ('USDT', portfolio_data['usdt_balance'], portfolio_data['usdt_balance'], 1.0, datetime.now().isoformat()))
        
        # Insert crypto holdings
        for holding in portfolio_data['crypto_holdings']:
            cursor.execute('''
                INSERT INTO portfolio (symbol, amount, value, price, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                holding['token'],
                holding['total'],
                holding['value'],
                holding['price'],
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
        
        print("✅ Portfolio synced to database successfully")
        
    except Exception as e:
        print(f"❌ Database Sync Failed: {e}")

def fix_portfolio_sync():
    """Main function to diagnose and fix portfolio sync"""
    print("🚀 OKX Portfolio Sync Diagnostic Starting...\n")
    
    # Test OKX connection
    exchange, balance = test_okx_connection()
    
    if not exchange:
        print("\n❌ Cannot proceed without OKX connection")
        return
    
    # Analyze OKX portfolio
    portfolio_data = analyze_okx_portfolio(exchange, balance)
    
    if not portfolio_data:
        print("\n❌ Cannot analyze portfolio")
        return
    
    # Check database
    check_database_portfolio()
    
    # Sync to database
    sync_portfolio_to_database(portfolio_data)
    
    print("\n✅ Portfolio Sync Diagnostic Complete")
    print(f"📊 Portfolio Value: ${portfolio_data['total_value']:.2f}")
    print(f"💰 USDT Balance: ${portfolio_data['usdt_balance']:.2f}")
    print(f"🪙 Crypto Holdings: {len(portfolio_data['crypto_holdings'])} tokens")

if __name__ == "__main__":
    fix_portfolio_sync()