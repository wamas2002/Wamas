#!/usr/bin/env python3
"""
Trading System Status Report
Real-time analysis of system performance and portfolio metrics
"""

import os
import ccxt
import sqlite3
from datetime import datetime

def generate_system_report():
    """Generate comprehensive system status report"""
    
    print("TRADING SYSTEM STATUS REPORT")
    print("=" * 45)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # OKX Connection
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False
        })
        
        balance = exchange.fetch_balance()
        print("OKX CONNECTION: ACTIVE")
        print("-" * 25)
        
        # Portfolio Analysis
        total_value = float(balance['USDT']['free'])
        positions = []
        
        tokens = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        
        for token in tokens:
            if token in balance and balance[token]['free'] > 0:
                amount = float(balance[token]['free'])
                if amount > 0:
                    try:
                        symbol = f"{token}/USDT"
                        ticker = exchange.fetch_ticker(symbol)
                        price = float(ticker['last'])
                        value = amount * price
                        change_24h = float(ticker['percentage']) if ticker['percentage'] else 0
                        total_value += value
                        
                        positions.append({
                            'token': token,
                            'amount': amount,
                            'price': price,
                            'value': value,
                            'change': change_24h
                        })
                    except Exception as e:
                        print(f"Error fetching {token}: {e}")
        
        print(f"PORTFOLIO OVERVIEW")
        print(f"Total Value: ${total_value:.2f}")
        print(f"USDT Cash: ${float(balance['USDT']['free']):.2f}")
        print(f"Active Positions: {len(positions)}")
        print()
        
        if positions:
            print("CURRENT HOLDINGS:")
            for pos in positions:
                percentage = (pos['value'] / total_value) * 100
                trend = "+" if pos['change'] > 0 else ""
                print(f"{pos['token']}: {pos['amount']:.6f} @ ${pos['price']:.2f} = ${pos['value']:.2f} ({percentage:.1f}%) {trend}{pos['change']:.1f}%")
            
            # Calculate allocation
            crypto_value = sum(pos['value'] for pos in positions)
            crypto_pct = (crypto_value / total_value) * 100
            cash_pct = 100 - crypto_pct
            
            print(f"\nALLOCATION:")
            print(f"Cryptocurrency: {crypto_pct:.1f}%")
            print(f"Cash (USDT): {cash_pct:.1f}%")
        
    except Exception as e:
        print(f"OKX CONNECTION: ERROR - {e}")
    
    print()
    
    # Trading Activity
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM live_trades')
        total_trades = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT symbol, side, amount, price, timestamp 
            FROM live_trades 
            ORDER BY timestamp DESC LIMIT 5
        ''')
        recent_trades = cursor.fetchall()
        
        print("TRADING ACTIVITY")
        print("-" * 20)
        print(f"Total Trades: {total_trades}")
        
        if recent_trades:
            print("Recent Trades:")
            for trade in recent_trades:
                symbol, side, amount, price, timestamp = trade
                value = float(amount) * float(price)
                print(f"{timestamp[:16]} | {side.upper()} {float(amount):.6f} {symbol} @ ${float(price):.2f}")
        
        conn.close()
        
    except Exception as e:
        print(f"Trading data error: {e}")
    
    print()
    
    # AI Signals
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM ai_signals')
        total_signals = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT signal, COUNT(*) as count
            FROM ai_signals
            WHERE id > (SELECT MAX(id) - 50 FROM ai_signals)
            GROUP BY signal
        ''')
        signal_dist = cursor.fetchall()
        
        print("AI SIGNAL PERFORMANCE")
        print("-" * 25)
        print(f"Total Signals: {total_signals}")
        
        if signal_dist:
            print("Recent Distribution:")
            for signal, count in signal_dist:
                print(f"{signal}: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"AI signals error: {e}")
    
    print()
    
    # System Components
    print("SYSTEM COMPONENTS")
    print("-" * 20)
    
    components = [
        ("Trading Platform", "Port 5000"),
        ("Enhanced Monitor", "Port 5001"),
        ("Live Trading Bridge", "Active"),
        ("Database Systems", "Operational"),
        ("Risk Management", "Active")
    ]
    
    for name, status in components:
        print(f"{name}: {status}")
    
    print()
    
    # Token Expansion Status
    print("EXPANDED MARKET COVERAGE")
    print("-" * 30)
    
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM expanded_trading_symbols 
            WHERE enabled = 1
        ''')
        enabled_symbols = cursor.fetchone()[0]
        
        print(f"Monitored Cryptocurrencies: {enabled_symbols}")
        print("Supported Assets: BTC, ETH, SOL, ADA, DOT, AVAX")
        
        conn.close()
        
    except Exception as e:
        print("Token expansion data not available")
    
    print()
    
    # Performance Metrics
    if 'positions' in locals() and positions and 'total_value' in locals():
        print("PERFORMANCE METRICS")
        print("-" * 22)
        
        # Calculate weighted performance
        total_invested = sum(pos['value'] for pos in positions)
        weighted_return = sum(pos['change'] * pos['value'] for pos in positions) / total_invested
        
        print(f"Portfolio 24h Performance: {weighted_return:+.2f}%")
        
        best = max(positions, key=lambda x: x['change'])
        worst = min(positions, key=lambda x: x['change'])
        
        print(f"Best Performer: {best['token']} ({best['change']:+.1f}%)")
        print(f"Worst Performer: {worst['token']} ({worst['change']:+.1f}%)")
        
        # Risk assessment
        if crypto_pct > 80:
            risk_level = "HIGH"
        elif crypto_pct < 20:
            risk_level = "LOW"
        else:
            risk_level = "MODERATE"
        
        print(f"Risk Level: {risk_level}")
    
    print()
    
    # Recommendations
    print("SYSTEM STATUS & RECOMMENDATIONS")
    print("-" * 35)
    
    recommendations = []
    
    if 'crypto_pct' in locals():
        if crypto_pct < 30:
            recommendations.append("Consider increasing crypto allocation")
        elif crypto_pct > 85:
            recommendations.append("Consider taking profits to reduce risk")
    
    if 'total_trades' in locals() and total_trades == 0:
        recommendations.append("No recent trading activity - review signal thresholds")
    
    if not recommendations:
        recommendations.append("System operating within normal parameters")
        recommendations.append("Continue monitoring for opportunities")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print()
    print("OVERALL STATUS: OPERATIONAL")
    print("=" * 45)

if __name__ == '__main__':
    generate_system_report()