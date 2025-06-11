#!/usr/bin/env python3
"""
Check AI Trading System Performance and Money-Making Status
"""

import requests
import json
import sqlite3
from datetime import datetime, timedelta

def check_ai_signals():
    """Check current AI trading signals"""
    try:
        response = requests.get('http://localhost:5000/api/unified/signals')
        if response.status_code == 200:
            signals = response.json()
            print("=== AI TRADING SIGNALS STATUS ===")
            print(f"Active Signals: {len(signals) if signals else 0}")
            
            if signals:
                for signal in signals:
                    symbol = signal.get('symbol', 'N/A')
                    action = signal.get('action', 'N/A')
                    confidence = signal.get('confidence', 0)
                    price = signal.get('current_price', 0)
                    target = signal.get('target_price', 0)
                    
                    print(f"  {symbol}: {action} - {confidence:.1f}% confidence")
                    print(f"    Current: ${price:,.2f} | Target: ${target:,.2f}")
            else:
                print("  No active signals")
        else:
            print(f"Error fetching signals: {response.status_code}")
    except Exception as e:
        print(f"Signal check error: {e}")

def check_portfolio_performance():
    """Check portfolio performance and profit/loss"""
    try:
        response = requests.get('http://localhost:5000/api/unified/portfolio')
        if response.status_code == 200:
            portfolio = response.json()
            print("\n=== PORTFOLIO STATUS ===")
            
            total_value = 0
            for item in portfolio:
                symbol = item.get('symbol', 'N/A')
                balance = item.get('balance', 0)
                value = item.get('value_usd', 0)
                percentage = item.get('percentage', 0)
                
                print(f"  {symbol}: {balance:.6f} (${value:,.2f} - {percentage:.1f}%)")
                total_value += value
            
            print(f"  Total Portfolio Value: ${total_value:,.2f}")
        else:
            print(f"Error fetching portfolio: {response.status_code}")
    except Exception as e:
        print(f"Portfolio check error: {e}")

def check_trading_history():
    """Check recent trading history and P&L"""
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        # Check for recent trades
        cursor.execute("""
            SELECT symbol, action, quantity, price, timestamp, profit_loss
            FROM trades 
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        trades = cursor.fetchall()
        print("\n=== RECENT TRADING ACTIVITY ===")
        
        if trades:
            total_pnl = 0
            for trade in trades:
                symbol, action, qty, price, timestamp, pnl = trade
                print(f"  {timestamp}: {action} {qty:.6f} {symbol} @ ${price:,.2f}")
                if pnl:
                    print(f"    P&L: ${pnl:,.2f}")
                    total_pnl += pnl
            
            print(f"  Total P&L (7 days): ${total_pnl:,.2f}")
        else:
            print("  No recent trades found")
            
        conn.close()
    except Exception as e:
        print(f"Trading history check error: {e}")

def check_system_health():
    """Check overall system health"""
    try:
        response = requests.get('http://localhost:5000/api/unified/health')
        if response.status_code == 200:
            health = response.json()
            print(f"\n=== SYSTEM HEALTH ===")
            print(f"  Status: {health.get('status', 'UNKNOWN')}")
            print(f"  Overall Health: {health.get('overall_health', 0):.1f}%")
            print(f"  Exchange Connected: {health.get('exchange_connected', False)}")
            print(f"  Database Connected: {health.get('database_connected', False)}")
        else:
            print(f"Error fetching health: {response.status_code}")
    except Exception as e:
        print(f"Health check error: {e}")

def main():
    print("CHECKING AI TRADING SYSTEM MONEY-MAKING CAPABILITY")
    print("=" * 55)
    
    check_ai_signals()
    check_portfolio_performance()
    check_trading_history()
    check_system_health()
    
    print(f"\nCheck completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()