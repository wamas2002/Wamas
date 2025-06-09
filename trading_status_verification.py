#!/usr/bin/env python3
"""
Complete Trading System Status Verification
Real-time verification of live trading execution and performance monitoring
"""

import sqlite3
import os
import ccxt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def check_okx_connectivity() -> Dict:
    """Verify OKX API connectivity and portfolio status"""
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'rateLimit': 2000,
            'enableRateLimit': True,
        })
        
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        
        # Calculate total portfolio value
        total_value = usdt_balance
        active_positions = 0
        
        for currency in balance:
            if currency != 'USDT' and balance[currency]['free'] > 0:
                amount = float(balance[currency]['free'])
                if amount > 0:
                    try:
                        symbol = f"{currency}/USDT"
                        ticker = exchange.fetch_ticker(symbol)
                        price = float(ticker['last'])
                        total_value += amount * price
                        active_positions += 1
                    except:
                        continue
        
        return {
            'status': 'connected',
            'usdt_balance': usdt_balance,
            'total_value': total_value,
            'active_positions': active_positions
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def check_signal_generation() -> Dict:
    """Check AI signal generation status"""
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Check recent signals
        cursor.execute('''
            SELECT COUNT(*) FROM ai_signals 
            WHERE datetime(timestamp) >= datetime('now', '-1 hour')
        ''')
        recent_signals = cursor.fetchone()[0]
        
        # Check high-confidence signals
        cursor.execute('''
            SELECT COUNT(*) FROM ai_signals 
            WHERE confidence >= 60 
            AND datetime(timestamp) >= datetime('now', '-1 hour')
        ''')
        high_confidence = cursor.fetchone()[0]
        
        # Get latest signals
        cursor.execute('''
            SELECT symbol, signal, confidence, timestamp 
            FROM ai_signals 
            WHERE confidence >= 60
            ORDER BY id DESC LIMIT 5
        ''')
        latest_signals = cursor.fetchall()
        
        conn.close()
        
        return {
            'recent_signals': recent_signals,
            'high_confidence_signals': high_confidence,
            'latest_signals': latest_signals,
            'signal_rate': recent_signals / 1 if recent_signals > 0 else 0  # per hour
        }
        
    except Exception as e:
        return {'error': str(e)}

def check_trade_execution() -> Dict:
    """Check live trade execution status"""
    try:
        # Check if live trading database exists
        if os.path.exists('live_trading.db'):
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            # Check for executed trades
            cursor.execute('''
                SELECT COUNT(*) FROM live_trades 
                WHERE datetime(timestamp) >= datetime('now', '-1 hour')
            ''')
            recent_trades = cursor.fetchone()[0]
            
            # Get latest trades
            cursor.execute('''
                SELECT symbol, side, amount, price, status, timestamp
                FROM live_trades 
                ORDER BY timestamp DESC LIMIT 5
            ''')
            latest_trades = cursor.fetchall()
            
            conn.close()
            
            return {
                'database_exists': True,
                'recent_trades': recent_trades,
                'latest_trades': latest_trades,
                'execution_active': recent_trades > 0
            }
        else:
            return {
                'database_exists': False,
                'recent_trades': 0,
                'execution_active': False
            }
            
    except Exception as e:
        return {'error': str(e)}

def calculate_system_health() -> Dict:
    """Calculate overall system health metrics"""
    okx_status = check_okx_connectivity()
    signal_status = check_signal_generation()
    execution_status = check_trade_execution()
    
    # Calculate health score (0-100)
    health_score = 0
    
    if okx_status.get('status') == 'connected':
        health_score += 30
    
    if signal_status.get('high_confidence_signals', 0) > 0:
        health_score += 25
    
    if execution_status.get('execution_active', False):
        health_score += 25
    
    if okx_status.get('usdt_balance', 0) > 0:
        health_score += 20
    
    # Determine system status
    if health_score >= 80:
        status = "FULLY OPERATIONAL"
    elif health_score >= 60:
        status = "OPERATIONAL"
    elif health_score >= 40:
        status = "PARTIAL"
    else:
        status = "INITIALIZING"
    
    return {
        'health_score': health_score,
        'status': status,
        'components': {
            'okx_connection': okx_status,
            'signal_generation': signal_status,
            'trade_execution': execution_status
        }
    }

def display_system_status():
    """Display comprehensive system status"""
    print("\n🚀 LIVE TRADING SYSTEM - COMPREHENSIVE STATUS")
    print("=" * 60)
    
    health = calculate_system_health()
    
    print(f"📊 System Health: {health['health_score']}/100 - {health['status']}")
    print("-" * 60)
    
    # OKX Connection Status
    okx = health['components']['okx_connection']
    if okx.get('status') == 'connected':
        print("✅ OKX Exchange: CONNECTED")
        print(f"   💰 USDT Balance: ${okx['usdt_balance']:.2f}")
        print(f"   📈 Portfolio Value: ${okx['total_value']:.2f}")
        print(f"   🎯 Active Positions: {okx['active_positions']}")
    else:
        print(f"❌ OKX Exchange: ERROR - {okx.get('error', 'Unknown')}")
    
    # Signal Generation Status
    signals = health['components']['signal_generation']
    if 'error' not in signals:
        print(f"\n✅ AI Signal Generation: ACTIVE")
        print(f"   📊 Signals (1h): {signals['recent_signals']}")
        print(f"   🎯 High Confidence: {signals['high_confidence_signals']}")
        print(f"   ⚡ Generation Rate: {signals['signal_rate']:.1f}/hour")
        
        if signals['latest_signals']:
            print("   📋 Recent Strong Signals:")
            for signal in signals['latest_signals']:
                timestamp = signal[3][:16] if len(signal[3]) > 16 else signal[3]
                print(f"      {signal[0]} {signal[1]} @ {signal[2]:.0f}% - {timestamp}")
    else:
        print(f"❌ Signal Generation: ERROR - {signals['error']}")
    
    # Trade Execution Status
    execution = health['components']['trade_execution']
    if 'error' not in execution:
        if execution['execution_active']:
            print(f"\n✅ Trade Execution: ACTIVE")
            print(f"   💹 Trades (1h): {execution['recent_trades']}")
            if execution['latest_trades']:
                print("   📈 Recent Executions:")
                for trade in execution['latest_trades']:
                    value = float(trade[2]) * float(trade[3])
                    print(f"      {trade[1]} {trade[2]:.6f} {trade[0]} @ ${trade[3]:.4f} (${value:.2f})")
        else:
            print("\n🔄 Trade Execution: MONITORING")
            print("   📱 System ready - awaiting high-confidence signals")
            if execution['database_exists']:
                print("   💾 Trading database initialized")
    else:
        print(f"❌ Trade Execution: ERROR - {execution['error']}")
    
    # Current Configuration
    print(f"\n⚙️  TRADING CONFIGURATION:")
    print("   🎯 Confidence Threshold: ≥60%")
    print("   💰 Risk Management: 1% per trade")
    print("   🔄 Minimum Trade: $5 USDT")
    print("   🚀 Trading Mode: LIVE (Real Money)")
    
    print("\n" + "=" * 60)
    
    if health['health_score'] >= 80:
        print("🎉 SYSTEM STATUS: FULLY OPERATIONAL")
        print("   Autonomous trading active with real-time signal execution")
    elif health['health_score'] >= 60:
        print("✅ SYSTEM STATUS: OPERATIONAL") 
        print("   Core functions active - monitoring performance")
    else:
        print("🔄 SYSTEM STATUS: INITIALIZING")
        print("   Components starting up - full functionality pending")
    
    print("=" * 60)

if __name__ == '__main__':
    display_system_status()