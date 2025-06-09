#!/usr/bin/env python3
"""
Live Trading Performance Summary
Comprehensive analysis of trading system performance and optimization status
"""

import sqlite3
import os
import ccxt
from datetime import datetime, timedelta
import json

def get_system_status():
    """Get comprehensive system status"""
    try:
        # OKX Connection Status
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'rateLimit': 2000,
        })
        
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        
        total_value = usdt_balance
        active_positions = 0
        position_details = []
        
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
                        active_positions += 1
                        position_details.append({
                            'symbol': currency,
                            'amount': amount,
                            'price': price,
                            'value': value
                        })
                    except:
                        continue
        
        return {
            'okx_connected': True,
            'usdt_balance': usdt_balance,
            'total_portfolio_value': total_value,
            'active_positions': active_positions,
            'position_details': position_details
        }
    except Exception as e:
        return {
            'okx_connected': False,
            'error': str(e)
        }

def get_recent_trading_activity():
    """Analyze recent trading activity"""
    try:
        if not os.path.exists('live_trading.db'):
            return {'trades_found': False, 'total_trades': 0}
        
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        # Get all trades
        cursor.execute('''
            SELECT symbol, side, amount, price, status, timestamp
            FROM live_trades 
            ORDER BY timestamp DESC
        ''')
        all_trades = cursor.fetchall()
        
        # Get recent trades (last 24 hours)
        cursor.execute('''
            SELECT symbol, side, amount, price, status, timestamp
            FROM live_trades 
            WHERE datetime(timestamp) >= datetime('now', '-24 hours')
            ORDER BY timestamp DESC
        ''')
        recent_trades = cursor.fetchall()
        
        conn.close()
        
        # Calculate metrics
        total_volume = sum([float(trade[2]) * float(trade[3]) for trade in all_trades])
        recent_volume = sum([float(trade[2]) * float(trade[3]) for trade in recent_trades])
        
        buy_trades = [t for t in all_trades if t[1] == 'buy']
        sell_trades = [t for t in all_trades if t[1] == 'sell']
        
        return {
            'trades_found': True,
            'total_trades': len(all_trades),
            'recent_trades_24h': len(recent_trades),
            'total_volume': total_volume,
            'recent_volume_24h': recent_volume,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'latest_trades': all_trades[:5]
        }
    except Exception as e:
        return {
            'trades_found': False,
            'error': str(e)
        }

def get_ai_signal_performance():
    """Analyze AI signal generation performance"""
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Get recent signals
        cursor.execute('''
            SELECT symbol, signal, confidence, timestamp 
            FROM ai_signals 
            WHERE datetime(timestamp) >= datetime('now', '-24 hours')
            ORDER BY id DESC
        ''')
        recent_signals = cursor.fetchall()
        
        # Get high confidence signals
        cursor.execute('''
            SELECT symbol, signal, confidence, timestamp 
            FROM ai_signals 
            WHERE confidence >= 60 
            AND datetime(timestamp) >= datetime('now', '-24 hours')
            ORDER BY id DESC
        ''')
        executable_signals = cursor.fetchall()
        
        conn.close()
        
        return {
            'signals_generated_24h': len(recent_signals),
            'executable_signals_24h': len(executable_signals),
            'signal_quality_ratio': len(executable_signals) / len(recent_signals) * 100 if recent_signals else 0,
            'latest_signals': recent_signals[:5]
        }
    except Exception as e:
        return {
            'signals_generated_24h': 0,
            'error': str(e)
        }

def analyze_trading_efficiency():
    """Calculate trading efficiency metrics"""
    trading_data = get_recent_trading_activity()
    signal_data = get_ai_signal_performance()
    
    if not trading_data.get('trades_found') or trading_data['total_trades'] == 0:
        return {
            'execution_rate': 0,
            'signal_to_trade_ratio': 0,
            'efficiency_score': 'Insufficient data'
        }
    
    executable_signals = signal_data.get('executable_signals_24h', 0)
    executed_trades = trading_data.get('recent_trades_24h', 0)
    
    execution_rate = (executed_trades / executable_signals * 100) if executable_signals > 0 else 0
    
    # Calculate efficiency score
    if execution_rate >= 80:
        efficiency = 'Excellent'
    elif execution_rate >= 60:
        efficiency = 'Good'
    elif execution_rate >= 40:
        efficiency = 'Fair'
    else:
        efficiency = 'Poor'
    
    return {
        'execution_rate': execution_rate,
        'executable_signals_24h': executable_signals,
        'executed_trades_24h': executed_trades,
        'efficiency_score': efficiency
    }

def generate_performance_report():
    """Generate comprehensive performance report"""
    print("LIVE TRADING PERFORMANCE OPTIMIZATION REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # System Status
    system = get_system_status()
    print("SYSTEM STATUS")
    print("-" * 30)
    if system.get('okx_connected'):
        print(f"✅ OKX Exchange: CONNECTED")
        print(f"💰 USDT Balance: ${system['usdt_balance']:.2f}")
        print(f"📈 Total Portfolio: ${system['total_portfolio_value']:.2f}")
        print(f"🎯 Active Positions: {system['active_positions']}")
        
        if system['position_details']:
            print("   Current Holdings:")
            for pos in system['position_details']:
                print(f"     {pos['symbol']}: {pos['amount']:.6f} @ ${pos['price']:.4f} = ${pos['value']:.2f}")
    else:
        print(f"❌ OKX Exchange: ERROR - {system.get('error', 'Unknown')}")
    print()
    
    # Trading Activity
    trading = get_recent_trading_activity()
    print("TRADING ACTIVITY")
    print("-" * 30)
    if trading.get('trades_found'):
        print(f"📊 Total Trades: {trading['total_trades']}")
        print(f"📈 Recent Trades (24h): {trading['recent_trades_24h']}")
        print(f"💹 Total Volume: ${trading['total_volume']:.2f}")
        print(f"🔄 Recent Volume (24h): ${trading['recent_volume_24h']:.2f}")
        print(f"📊 Buy/Sell Ratio: {trading['buy_trades']}/{trading['sell_trades']}")
        
        if trading['latest_trades']:
            print("   Recent Executions:")
            for trade in trading['latest_trades']:
                value = float(trade[2]) * float(trade[3])
                timestamp = trade[5][:16] if len(trade[5]) > 16 else trade[5]
                print(f"     {trade[1].upper()} {trade[0]} ${value:.2f} - {timestamp}")
    else:
        print("📱 No trading activity recorded")
    print()
    
    # AI Signal Performance
    signals = get_ai_signal_performance()
    print("AI SIGNAL PERFORMANCE")
    print("-" * 30)
    print(f"🤖 Signals Generated (24h): {signals['signals_generated_24h']}")
    print(f"🎯 Executable Signals (≥60%): {signals.get('executable_signals_24h', 0)}")
    print(f"📊 Signal Quality: {signals.get('signal_quality_ratio', 0):.1f}%")
    
    if signals.get('latest_signals'):
        print("   Recent Strong Signals:")
        for signal in signals['latest_signals']:
            timestamp = signal[3][:16] if len(signal[3]) > 16 else signal[3]
            print(f"     {signal[0]} {signal[1]} @ {signal[2]:.0f}% - {timestamp}")
    print()
    
    # Trading Efficiency
    efficiency = analyze_trading_efficiency()
    print("TRADING EFFICIENCY")
    print("-" * 30)
    print(f"⚡ Execution Rate: {efficiency['execution_rate']:.1f}%")
    print(f"📈 Efficiency Score: {efficiency['efficiency_score']}")
    print(f"🔄 Signal→Trade Conversion: {efficiency.get('executed_trades_24h', 0)}/{efficiency.get('executable_signals_24h', 0)}")
    print()
    
    # Optimization Recommendations
    print("OPTIMIZATION RECOMMENDATIONS")
    print("-" * 30)
    
    recommendations = []
    
    if efficiency['execution_rate'] < 50:
        recommendations.append("Consider lowering confidence threshold to increase trade frequency")
    
    if trading.get('recent_trades_24h', 0) == 0:
        recommendations.append("Check signal execution bridge - no recent trades detected")
    
    if signals.get('signal_quality_ratio', 0) < 30:
        recommendations.append("AI model may need retraining - low quality signal ratio")
    
    if system.get('usdt_balance', 0) < 100:
        recommendations.append("USDT balance low - consider depositing more funds")
    
    if len(recommendations) == 0:
        recommendations.append("System operating optimally - continue monitoring performance")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print()
    print("=" * 60)
    
    # Performance Summary
    if trading.get('total_trades', 0) > 0 and system.get('okx_connected'):
        print("🚀 SYSTEM STATUS: FULLY OPERATIONAL")
        print("   Autonomous trading active with confirmed executions")
    elif system.get('okx_connected') and signals.get('signals_generated_24h', 0) > 0:
        print("⚡ SYSTEM STATUS: ACTIVE MONITORING")
        print("   Signal generation active - awaiting execution opportunities")
    else:
        print("🔄 SYSTEM STATUS: INITIALIZING")
        print("   Components starting up or require attention")
    
    print("=" * 60)

if __name__ == '__main__':
    generate_performance_report()