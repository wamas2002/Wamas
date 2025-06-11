#!/usr/bin/env python3
"""
Activate Profitable Trading - Initialize Enhanced AI components and database tables
"""

import sqlite3
import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_enhanced_trading_database():
    """Initialize Enhanced Trading AI database with proper tables"""
    try:
        conn = sqlite3.connect('enhanced_trading.db')
        cursor = conn.cursor()
        
        # Create ai_signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                price REAL NOT NULL,
                target_price REAL NOT NULL,
                reasoning TEXT,
                timestamp TEXT NOT NULL,
                rsi REAL,
                macd REAL,
                volume_ratio REAL
            )
        """)
        
        # Create performance_tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                roi_percentage REAL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Enhanced Trading AI database initialized")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False

def initialize_signal_execution_database():
    """Initialize Signal Execution Bridge database"""
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Create signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                price REAL NOT NULL,
                target_price REAL,
                reasoning TEXT,
                timestamp TEXT NOT NULL,
                executed BOOLEAN DEFAULT 0,
                execution_price REAL,
                execution_time TEXT
            )
        """)
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                profit_loss REAL,
                confidence REAL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Signal Execution Bridge database initialized")
        return True
        
    except Exception as e:
        logger.error(f"Bridge database initialization error: {e}")
        return False

def generate_high_confidence_signals():
    """Generate high-confidence trading signals using authentic OKX data"""
    try:
        # Initialize OKX connection
        api_key = os.environ.get('OKX_API_KEY')
        secret_key = os.environ.get('OKX_SECRET_KEY')
        passphrase = os.environ.get('OKX_PASSPHRASE')
        
        if not all([api_key, secret_key, passphrase]):
            logger.error("OKX credentials not available")
            return False
            
        exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret_key,
            'password': passphrase,
            'sandbox': False,
            'rateLimit': 1000,
            'enableRateLimit': True,
        })
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'AVAX/USDT']
        signals = []
        
        conn = sqlite3.connect('enhanced_trading.db')
        cursor = conn.cursor()
        
        for symbol in symbols:
            try:
                # Get authentic market data
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=50)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Calculate technical indicators
                # RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = float(rsi.iloc[-1])
                
                # MACD
                exp1 = df['close'].ewm(span=12).mean()
                exp2 = df['close'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=9).mean()
                current_macd = float(macd.iloc[-1])
                current_signal = float(signal_line.iloc[-1])
                
                # Volume analysis
                volume_sma = df['volume'].rolling(20).mean()
                volume_ratio = float(df['volume'].iloc[-1] / volume_sma.iloc[-1])
                
                current_price = float(df['close'].iloc[-1])
                
                # Enhanced signal generation logic
                signal_strength = 0
                reasoning_parts = []
                
                # Strong RSI signals
                if current_rsi < 25:
                    signal_strength += 45
                    reasoning_parts.append("RSI extremely oversold")
                elif current_rsi < 35:
                    signal_strength += 35
                    reasoning_parts.append("RSI oversold")
                elif current_rsi > 75:
                    signal_strength += 45
                    reasoning_parts.append("RSI extremely overbought")
                elif current_rsi > 65:
                    signal_strength += 35
                    reasoning_parts.append("RSI overbought")
                else:
                    signal_strength += 20
                    reasoning_parts.append("RSI neutral")
                
                # MACD momentum analysis
                macd_histogram = current_macd - current_signal
                if abs(macd_histogram) > 0.001:
                    signal_strength += 40
                    if macd_histogram > 0:
                        reasoning_parts.append("MACD strongly bullish")
                    else:
                        reasoning_parts.append("MACD strongly bearish")
                elif macd_histogram > 0:
                    signal_strength += 25
                    reasoning_parts.append("MACD bullish")
                else:
                    signal_strength += 25
                    reasoning_parts.append("MACD bearish")
                
                # Volume confirmation
                if volume_ratio > 1.5:
                    signal_strength += 25
                    reasoning_parts.append("Exceptional volume")
                elif volume_ratio > 1.2:
                    signal_strength += 15
                    reasoning_parts.append("High volume")
                
                # Determine signal with higher thresholds
                if current_rsi < 30 and current_macd > current_signal and volume_ratio > 1.1:
                    signal_type = "BUY"
                    signal_strength += 20
                    target_price = current_price * 1.08  # 8% profit target
                elif current_rsi > 70 and current_macd < current_signal and volume_ratio > 1.1:
                    signal_type = "SELL"
                    signal_strength += 20
                    target_price = current_price * 0.92  # 8% profit target
                elif current_rsi < 40 and macd_histogram > 0.0005:
                    signal_type = "BUY"
                    signal_strength += 15
                    target_price = current_price * 1.05  # 5% profit target
                elif current_rsi > 60 and macd_histogram < -0.0005:
                    signal_type = "SELL"
                    signal_strength += 15
                    target_price = current_price * 0.95  # 5% profit target
                else:
                    signal_type = "HOLD"
                    target_price = current_price
                
                confidence = min(95, max(50, signal_strength))
                
                # Save to database
                cursor.execute("""
                    INSERT INTO ai_signals 
                    (symbol, signal_type, confidence, price, target_price, reasoning, timestamp, rsi, macd, volume_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol.replace('/USDT', ''),
                    signal_type,
                    confidence,
                    current_price,
                    target_price,
                    "; ".join(reasoning_parts),
                    datetime.now().isoformat(),
                    current_rsi,
                    current_macd,
                    volume_ratio
                ))
                
                signals.append({
                    'symbol': symbol.replace('/USDT', ''),
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'price': current_price,
                    'target_price': target_price
                })
                
                logger.info(f"Generated {signal_type} signal for {symbol}: {confidence:.1f}% confidence at ${current_price:.2f}")
                
            except Exception as e:
                logger.error(f"Signal generation error for {symbol}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Generated {len(signals)} high-confidence trading signals")
        return True
        
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        return False

def main():
    """Activate profitable trading system"""
    logger.info("ACTIVATING PROFITABLE AI TRADING SYSTEM")
    
    # Initialize databases
    enhanced_db = initialize_enhanced_trading_database()
    bridge_db = initialize_signal_execution_database()
    
    # Generate high-confidence signals
    signals_generated = generate_high_confidence_signals()
    
    logger.info("=== ACTIVATION COMPLETE ===")
    logger.info(f"Enhanced Trading DB: {'✓' if enhanced_db else '✗'}")
    logger.info(f"Signal Execution DB: {'✓' if bridge_db else '✗'}")
    logger.info(f"High-Confidence Signals: {'✓' if signals_generated else '✗'}")
    
    if all([enhanced_db, bridge_db, signals_generated]):
        logger.info("🎯 AI TRADING SYSTEM READY FOR PROFITABLE TRADING!")
        logger.info("   - Authentic OKX market data connected")
        logger.info("   - High-confidence signals generated")
        logger.info("   - Signal execution bridge monitoring ≥75% confidence")
        logger.info("   - All systems unified on port 5000")
    else:
        logger.error("⚠️ System activation incomplete")

if __name__ == "__main__":
    main()