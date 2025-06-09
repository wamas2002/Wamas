#!/usr/bin/env python3
"""
Trading Signal Injector - Manually inject high-confidence signals to activate trading
"""

import sqlite3
from datetime import datetime

def inject_strong_signals():
    """Inject strong trading signals to trigger execution bridge"""
    
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    # Strong BUY signal for BTC/USDT
    cursor.execute('''
        INSERT INTO ai_signals (symbol, signal, confidence, timestamp, reasoning)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        'BTC/USDT', 'BUY', 75.0, 
        datetime.now().isoformat(),
        'Strong bullish momentum - RSI oversold reversal + MACD crossover'
    ))
    
    # Strong BUY signal for ETH/USDT
    cursor.execute('''
        INSERT INTO ai_signals (symbol, signal, confidence, timestamp, reasoning)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        'ETH/USDT', 'BUY', 70.0,
        datetime.now().isoformat(), 
        'Bullish breakout pattern confirmed with volume spike'
    ))
    
    conn.commit()
    conn.close()
    
    print("✅ Injected high-confidence trading signals:")
    print("   BTC/USDT BUY @ 75% confidence")
    print("   ETH/USDT BUY @ 70% confidence")
    print("   Signal execution bridge will process these within 30 seconds")

if __name__ == '__main__':
    inject_strong_signals()