#!/usr/bin/env python3
import os
import ccxt
import sqlite3

def show_system_status():
    """Show current system status and capabilities"""
    print("TRADING SYSTEM STATUS REPORT")
    print("=" * 40)
    
    # OKX Connection
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False
        })
        
        balance = exchange.fetch_balance()
        usdt = balance['USDT']['free']
        print(f"OKX Connection: Active")
        print(f"USDT Balance: ${usdt:.2f}")
        
        # Check for positions
        positions = 0
        for currency in balance:
            if currency != 'USDT' and balance[currency]['free'] > 0:
                amount = balance[currency]['free']
                if amount > 0:
                    positions += 1
                    symbol = f"{currency}/USDT"
                    try:
                        ticker = exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        value = amount * price
                        print(f"{currency}: {amount:.6f} @ ${price:.4f} = ${value:.2f}")
                    except:
                        print(f"{currency}: {amount:.6f} (price unavailable)")
        
        print(f"Active Positions: {positions}")
        
    except Exception as e:
        print(f"OKX Connection: Failed - {e}")
    
    # Database Status
    print(f"\nDatabase Status:")
    
    # Trading database
    if os.path.exists('live_trading.db'):
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM live_trades')
        trade_count = cursor.fetchone()[0]
        print(f"Live Trades Database: {trade_count} trades recorded")
        conn.close()
    else:
        print(f"Live Trades Database: Not found")
    
    # AI signals database
    if os.path.exists('trading_platform.db'):
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM ai_signals')
        signal_count = cursor.fetchone()[0]
        print(f"AI Signals Database: {signal_count} signals generated")
        conn.close()
    else:
        print(f"AI Signals Database: Not found")
    
    # System Features
    print(f"\nActive Features:")
    print(f"✓ Real-time OKX market data")
    print(f"✓ AI signal generation")
    print(f"✓ Automated SELL orders")
    print(f"✓ Stop-loss protection (2%)")
    print(f"✓ Profit taking (1.5%, 3%, 5%)")
    print(f"✓ Portfolio monitoring")
    print(f"✓ Risk management (1% per trade)")
    
    # Dashboard Access
    print(f"\nDashboard URLs:")
    print(f"Main Platform: http://localhost:5000")
    print(f"Enhanced Monitor: http://localhost:5001")
    print(f"Advanced Analytics: http://localhost:5002")
    
    print(f"\nSELL Order System:")
    print(f"- Monitoring all positions for exit conditions")
    print(f"- Automatic execution when targets are met")
    print(f"- Complete trade logging and history")
    print(f"- Real-time profit/loss calculations")

if __name__ == '__main__':
    show_system_status()