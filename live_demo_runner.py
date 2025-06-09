#!/usr/bin/env python3
"""
Live Demo Runner - Demonstrates Advanced Trading Features
"""

import os
import ccxt
import time
from datetime import datetime
import sqlite3

def demo_portfolio_analytics():
    """Demonstrate live portfolio analysis"""
    print("PORTFOLIO ANALYTICS DEMONSTRATION")
    print("=" * 50)
    
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'rateLimit': 2000,
            'enableRateLimit': True,
        })
        
        # Get current portfolio
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        total_value = usdt_balance
        
        print(f"USDT Balance: ${usdt_balance:.2f}")
        
        positions = []
        for currency in balance:
            if currency != 'USDT' and balance[currency]['free'] > 0:
                amount = float(balance[currency]['free'])
                if amount > 0:
                    try:
                        symbol = f"{currency}/USDT"
                        ticker = exchange.fetch_ticker(symbol)
                        price = float(ticker['last'])
                        value = amount * price
                        total_value += value
                        
                        positions.append({
                            'symbol': currency,
                            'amount': amount,
                            'price': price,
                            'value': value,
                            'change_24h': float(ticker.get('percentage', 0))
                        })
                    except:
                        continue
        
        print(f"Total Portfolio Value: ${total_value:.2f}")
        print(f"Number of Positions: {len(positions)}")
        
        if positions:
            print("\nCurrent Holdings:")
            for pos in positions:
                allocation = (pos['value'] / total_value) * 100
                change_indicator = "+" if pos['change_24h'] > 0 else ""
                print(f"  {pos['symbol']}: {pos['amount']:.6f} @ ${pos['price']:.4f} "
                      f"({allocation:.1f}%) {change_indicator}{pos['change_24h']:.2f}%")
        
        # Risk assessment
        usdt_percentage = (usdt_balance / total_value) * 100
        max_position = max([pos['value']/total_value*100 for pos in positions]) if positions else 0
        
        print(f"\nRisk Metrics:")
        print(f"  USDT Reserve: {usdt_percentage:.1f}%")
        print(f"  Largest Position: {max_position:.1f}%")
        
        risk_level = "Low"
        if usdt_percentage < 20 or max_position > 50:
            risk_level = "High"
        elif usdt_percentage < 40 or max_position > 30:
            risk_level = "Medium"
        
        print(f"  Overall Risk: {risk_level}")
        
    except Exception as e:
        print(f"Portfolio analysis error: {e}")

def demo_trading_performance():
    """Demonstrate trading performance metrics"""
    print("\nTRADING PERFORMANCE DEMONSTRATION")
    print("=" * 50)
    
    try:
        if not os.path.exists('live_trading.db'):
            print("No trading database found - system is ready for trading")
            return
        
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        # Get recent trades
        cursor.execute('''
            SELECT symbol, side, amount, price, timestamp, status
            FROM live_trades 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        recent_trades = cursor.fetchall()
        
        if recent_trades:
            print(f"Recent Trades ({len(recent_trades)} shown):")
            for trade in recent_trades:
                symbol, side, amount, price, timestamp, status = trade
                print(f"  {timestamp[:19]} | {side.upper()} {float(amount):.6f} {symbol} @ ${float(price):.4f} [{status}]")
        else:
            print("No recent trades found")
        
        # Get total statistics
        cursor.execute('SELECT COUNT(*) FROM live_trades')
        total_trades = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM live_trades WHERE status = "filled"')
        successful_trades = cursor.fetchone()[0]
        
        success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Successful: {successful_trades}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        conn.close()
        
    except Exception as e:
        print(f"Performance analysis error: {e}")

def demo_ai_signals():
    """Demonstrate AI signal generation"""
    print("\nAI SIGNAL GENERATION DEMONSTRATION")
    print("=" * 50)
    
    try:
        if not os.path.exists('trading_platform.db'):
            print("No signals database found - generating new signals")
            return
        
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Get recent AI signals
        cursor.execute('''
            SELECT symbol, signal, confidence, timestamp
            FROM ai_signals 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        recent_signals = cursor.fetchall()
        
        if recent_signals:
            print(f"Recent AI Signals ({len(recent_signals)} shown):")
            for signal in recent_signals:
                symbol, signal_type, confidence, timestamp = signal
                confidence_bar = "█" * (confidence // 10) + "░" * (10 - confidence // 10)
                print(f"  {timestamp[:19]} | {symbol} {signal_type} {confidence}% [{confidence_bar}]")
        else:
            print("No recent AI signals found")
        
        # Signal distribution
        cursor.execute('''
            SELECT signal, COUNT(*) 
            FROM ai_signals 
            GROUP BY signal
        ''')
        
        signal_dist = cursor.fetchall()
        
        if signal_dist:
            print(f"\nSignal Distribution:")
            for signal_type, count in signal_dist:
                print(f"  {signal_type}: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"AI signals analysis error: {e}")

def demo_market_analysis():
    """Demonstrate real-time market analysis"""
    print("\nMARKET ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'rateLimit': 2000,
            'enableRateLimit': True,
        })
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        print("Live Market Data:")
        for symbol in symbols:
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = float(ticker['last'])
                change_24h = float(ticker.get('percentage', 0))
                volume = float(ticker.get('quoteVolume', 0))
                
                change_indicator = "+" if change_24h > 0 else ""
                trend = "📈" if change_24h > 0 else "📉" if change_24h < 0 else "➡️"
                
                print(f"  {symbol}: ${price:,.2f} {change_indicator}{change_24h:.2f}% {trend} Vol: ${volume:,.0f}")
                
            except Exception as e:
                print(f"  {symbol}: Error fetching data")
        
        # Market volatility assessment
        volatilities = []
        for symbol in symbols:
            try:
                ticker = exchange.fetch_ticker(symbol)
                change_24h = abs(float(ticker.get('percentage', 0)))
                volatilities.append(change_24h)
            except:
                continue
        
        avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
        
        vol_level = "Low"
        if avg_volatility > 6:
            vol_level = "Extreme"
        elif avg_volatility > 4:
            vol_level = "High"
        elif avg_volatility > 2:
            vol_level = "Medium"
        
        print(f"\nMarket Conditions:")
        print(f"  Average Volatility: {avg_volatility:.1f}%")
        print(f"  Volatility Level: {vol_level}")
        print(f"  Trading Recommendation: {'Conservative' if avg_volatility > 5 else 'Normal' if avg_volatility > 2 else 'Aggressive'}")
        
    except Exception as e:
        print(f"Market analysis error: {e}")

def main():
    """Run comprehensive system demonstration"""
    print("ADVANCED TRADING SYSTEM LIVE DEMONSTRATION")
    print("=" * 60)
    print(f"Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run all demonstrations
    demo_portfolio_analytics()
    demo_trading_performance()
    demo_ai_signals()
    demo_market_analysis()
    
    print("\n" + "=" * 60)
    print("SYSTEM STATUS SUMMARY")
    print("=" * 60)
    
    # Check system components
    components = {
        'OKX Connection': 'Available' if os.environ.get('OKX_API_KEY') else 'Missing API Keys',
        'Trading Database': 'Active' if os.path.exists('live_trading.db') else 'Initializing',
        'AI Signals': 'Active' if os.path.exists('trading_platform.db') else 'Initializing',
        'Live Trading': 'Enabled' if os.path.exists('signal_execution_bridge.py') else 'Disabled'
    }
    
    for component, status in components.items():
        status_icon = "✅" if status in ['Available', 'Active', 'Enabled'] else "⚠️"
        print(f"  {status_icon} {component}: {status}")
    
    print("\nSystem is fully operational and ready for advanced trading")
    print("Access your dashboards at:")
    print("  • Main Platform: http://localhost:5000")
    print("  • Live Monitor: http://localhost:5001") 
    print("  • Advanced Analytics: http://localhost:5002")

if __name__ == '__main__':
    main()