#!/usr/bin/env python3
"""
Live Trading Optimization - Direct Parameter Adjustment
Real-time optimization of trading parameters based on market conditions and performance
"""

import sqlite3
import os
import ccxt
import json
from datetime import datetime, timedelta

def analyze_market_volatility():
    """Analyze current market volatility using BTC as benchmark"""
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'rateLimit': 2000,
            'enableRateLimit': True,
        })
        
        # Get 24h data for volatility calculation
        ticker = exchange.fetch_ticker('BTC/USDT')
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=24)
        
        current_price = float(ticker['last'])
        high_24h = max([candle[2] for candle in ohlcv])
        low_24h = min([candle[3] for candle in ohlcv])
        
        volatility_pct = ((high_24h - low_24h) / current_price) * 100
        
        # Price trend analysis
        prices = [candle[4] for candle in ohlcv[-6:]]  # Last 6 hours
        trend_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
        
        return {
            'volatility_pct': volatility_pct,
            'trend_pct': trend_pct,
            'current_price': current_price,
            'high_24h': high_24h,
            'low_24h': low_24h
        }
        
    except Exception as e:
        print(f"Market analysis error: {e}")
        return None

def get_recent_trade_performance():
    """Analyze recent trade execution performance"""
    try:
        if not os.path.exists('live_trading.db'):
            return {'trades': 0, 'recent_trades': []}
            
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        # Get trades from last 2 hours
        cursor.execute('''
            SELECT symbol, side, amount, price, status, timestamp
            FROM live_trades 
            WHERE datetime(timestamp) >= datetime('now', '-2 hours')
            ORDER BY timestamp DESC
        ''')
        
        trades = cursor.fetchall()
        conn.close()
        
        return {
            'trades': len(trades),
            'recent_trades': trades[:10]  # Last 10 trades
        }
        
    except Exception as e:
        print(f"Performance analysis error: {e}")
        return {'trades': 0, 'recent_trades': []}

def calculate_optimal_confidence_threshold(market_data, performance_data):
    """Calculate optimal confidence threshold based on conditions"""
    base_threshold = 60.0
    
    if not market_data:
        return base_threshold
    
    # Adjust for volatility
    volatility = market_data['volatility_pct']
    if volatility > 4:  # High volatility
        base_threshold += 10
    elif volatility < 1.5:  # Low volatility
        base_threshold -= 5
    
    # Adjust for trend strength
    trend = abs(market_data['trend_pct'])
    if trend > 3:  # Strong trend
        base_threshold -= 5
    elif trend < 1:  # Weak trend
        base_threshold += 5
    
    # Adjust for recent trade frequency
    recent_trades = performance_data['trades']
    if recent_trades > 5:  # Too many trades
        base_threshold += 5
    elif recent_trades == 0:  # No recent trades
        base_threshold -= 3
    
    return max(55.0, min(75.0, base_threshold))

def update_signal_execution_parameters(confidence_threshold):
    """Update signal execution bridge with optimized parameters"""
    try:
        # Read current signal execution bridge file
        with open('signal_execution_bridge.py', 'r') as f:
            content = f.read()
        
        # Update confidence threshold
        if 'self.confidence_threshold =' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'self.confidence_threshold =' in line and 'def ' not in line:
                    lines[i] = f'        self.confidence_threshold = {confidence_threshold}'
                    break
            
            # Write updated content
            with open('signal_execution_bridge.py', 'w') as f:
                f.write('\n'.join(lines))
            
            return True
    except Exception as e:
        print(f"Parameter update error: {e}")
        return False
    
    return False

def log_optimization_action(market_data, performance_data, new_threshold):
    """Log optimization action to database"""
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market_volatility REAL,
                market_trend REAL,
                recent_trades INTEGER,
                old_threshold REAL,
                new_threshold REAL,
                reasoning TEXT
            )
        ''')
        
        reasoning = f"Volatility: {market_data['volatility_pct']:.1f}%, Trend: {market_data['trend_pct']:.1f}%, Recent trades: {performance_data['trades']}"
        
        cursor.execute('''
            INSERT INTO optimization_log 
            (timestamp, market_volatility, market_trend, recent_trades, old_threshold, new_threshold, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            market_data['volatility_pct'],
            market_data['trend_pct'],
            performance_data['trades'],
            60.0,  # Previous threshold
            new_threshold,
            reasoning
        ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Logging error: {e}")
        return False

def main():
    """Main optimization function"""
    print("LIVE TRADING OPTIMIZATION")
    print("=" * 40)
    
    # Analyze current market conditions
    market_data = analyze_market_volatility()
    performance_data = get_recent_trade_performance()
    
    if market_data:
        print(f"Market Analysis:")
        print(f"  BTC Price: ${market_data['current_price']:,.2f}")
        print(f"  24h Volatility: {market_data['volatility_pct']:.2f}%")
        print(f"  6h Trend: {market_data['trend_pct']:+.2f}%")
    else:
        print("Market analysis unavailable - using default parameters")
        return
    
    print(f"Performance Analysis:")
    print(f"  Recent trades (2h): {performance_data['trades']}")
    
    # Calculate optimal parameters
    optimal_threshold = calculate_optimal_confidence_threshold(market_data, performance_data)
    
    print(f"Optimization:")
    print(f"  Current threshold: 60.0%")
    print(f"  Optimal threshold: {optimal_threshold:.1f}%")
    
    # Apply optimization if significant change needed
    threshold_change = abs(optimal_threshold - 60.0)
    if threshold_change >= 2.0:
        if update_signal_execution_parameters(optimal_threshold):
            print(f"✅ Updated confidence threshold to {optimal_threshold:.1f}%")
            log_optimization_action(market_data, performance_data, optimal_threshold)
        else:
            print("❌ Failed to update parameters")
    else:
        print("✅ Current parameters are optimal")
    
    # Show recent trades if any
    if performance_data['recent_trades']:
        print(f"\nRecent Trade Activity:")
        for trade in performance_data['recent_trades'][:3]:
            value = float(trade[2]) * float(trade[3])
            print(f"  {trade[1].upper()} {trade[0]} ${value:.2f}")
    
    print("=" * 40)

if __name__ == '__main__':
    main()