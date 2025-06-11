#!/usr/bin/env python3
"""
Comprehensive Trading Engine
Combines advanced AI, SELL signal generation, and continuous learning
"""

import sqlite3
import ccxt
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTradingEngine:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.exchange = None
        self.is_running = False
        self.initialize_exchange()
        
    def initialize_exchange(self):
        """Initialize OKX connection"""
        try:
            api_key = os.environ.get('OKX_API_KEY')
            secret_key = os.environ.get('OKX_SECRET_KEY')
            passphrase = os.environ.get('OKX_PASSPHRASE')
            
            if api_key and secret_key and passphrase:
                self.exchange = ccxt.okx({
                    'apiKey': api_key,
                    'secret': secret_key,
                    'password': passphrase,
                    'sandbox': False,
                    'rateLimit': 1000,
                    'enableRateLimit': True,
                })
                logger.info("Trading Engine connected to OKX")
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")

    def get_market_data(self, symbol, timeframe='1h', limit=100):
        """Get market data for analysis"""
        try:
            symbol_formatted = f"{symbol}/USDT" if not symbol.endswith('/USDT') else symbol
            ohlcv = self.exchange.fetch_ohlcv(symbol_formatted, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Market data error for {symbol}: {e}")
            return None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def generate_buy_signals(self, symbol, df):
        """Generate BUY signals"""
        if len(df) < 50:
            return None
            
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        buy_score = 0
        reasons = []
        
        # RSI oversold
        if current['rsi'] < 30:
            buy_score += 25
            reasons.append(f"RSI oversold ({current['rsi']:.1f})")
        elif current['rsi'] < 40:
            buy_score += 15
            reasons.append(f"RSI low ({current['rsi']:.1f})")
        
        # MACD bullish
        if current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            buy_score += 20
            reasons.append("MACD bullish crossover")
        elif current['macd_hist'] > 0 and current['macd_hist'] > prev['macd_hist']:
            buy_score += 10
            reasons.append("MACD histogram rising")
        
        # Moving average support
        if current['close'] > current['sma_20']:
            buy_score += 10
            reasons.append("Above SMA20")
        if current['sma_20'] > current['sma_50']:
            buy_score += 10
            reasons.append("SMA20 > SMA50")
        
        # Price momentum
        momentum_5 = (current['close'] / df['close'].iloc[-6] - 1) * 100
        if momentum_5 > 2:
            buy_score += 15
            reasons.append(f"Positive momentum ({momentum_5:.1f}%)")
        
        if buy_score >= 45:
            confidence = min(90, buy_score + 10)
            return {
                'symbol': symbol,
                'signal': 'BUY',
                'confidence': confidence,
                'reasoning': f"Technical BUY: {', '.join(reasons)}",
                'timestamp': datetime.now().isoformat()
            }
        
        return None

    def generate_sell_signals(self, symbol, df):
        """Generate SELL signals"""
        if len(df) < 50:
            return None
            
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        sell_score = 0
        reasons = []
        
        # RSI overbought
        if current['rsi'] > 70:
            sell_score += 25
            reasons.append(f"RSI overbought ({current['rsi']:.1f})")
        elif current['rsi'] > 65:
            sell_score += 15
            reasons.append(f"RSI high ({current['rsi']:.1f})")
        
        # MACD bearish
        if current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            sell_score += 20
            reasons.append("MACD bearish crossover")
        elif current['macd_hist'] < 0 and current['macd_hist'] < prev['macd_hist']:
            sell_score += 10
            reasons.append("MACD histogram declining")
        
        # Price below moving average
        if current['close'] < current['sma_20'] and prev['close'] >= prev['sma_20']:
            sell_score += 15
            reasons.append("Broke below SMA20")
        
        # Negative momentum
        momentum_5 = (current['close'] / df['close'].iloc[-6] - 1) * 100
        if momentum_5 < -2:
            sell_score += 15
            reasons.append(f"Negative momentum ({momentum_5:.1f}%)")
        
        # Check for profit-taking opportunity
        profit_signal = self.check_profit_taking(symbol)
        if profit_signal:
            sell_score += profit_signal['score']
            reasons.append(profit_signal['reason'])
        
        if sell_score >= 45:
            confidence = min(90, sell_score + 10)
            return {
                'symbol': symbol,
                'signal': 'SELL',
                'confidence': confidence,
                'reasoning': f"Technical SELL: {', '.join(reasons)}",
                'timestamp': datetime.now().isoformat()
            }
        
        return None

    def check_profit_taking(self, symbol):
        """Check for profit-taking opportunities"""
        try:
            # Get current position
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT quantity, current_price FROM portfolio WHERE symbol = ? AND quantity > 0", (symbol,))
            position = cursor.fetchone()
            
            if not position:
                conn.close()
                return None
            
            current_price = float(position[1])
            
            # Get recent buy trades
            conn_live = sqlite3.connect('live_trading.db')
            cursor_live = conn_live.cursor()
            
            cursor_live.execute('''
                SELECT price, amount FROM live_trades 
                WHERE symbol = ? AND side = 'BUY' 
                ORDER BY timestamp DESC LIMIT 5
            ''', (f"{symbol}/USDT",))
            
            buy_trades = cursor_live.fetchall()
            
            if buy_trades:
                total_cost = sum(float(price) * float(amount) for price, amount in buy_trades)
                total_amount = sum(float(amount) for price, amount in buy_trades)
                avg_buy_price = total_cost / total_amount if total_amount > 0 else current_price
                
                profit_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100
                
                if profit_pct > 5:  # 5%+ profit
                    score = min(25, int(profit_pct * 3))
                    return {
                        'score': score,
                        'reason': f"Profit taking: {profit_pct:.1f}% gain"
                    }
            
            conn.close()
            conn_live.close()
            return None
            
        except Exception as e:
            logger.error(f"Profit taking check error: {e}")
            return None

    def save_signals(self, signals):
        """Save signals to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for signal in signals:
                cursor.execute('''
                    INSERT INTO ai_signals (symbol, signal, confidence, reasoning, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    signal['symbol'],
                    signal['signal'],
                    signal['confidence'],
                    signal['reasoning'],
                    signal['timestamp']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving signals: {e}")

    def update_performance_metrics(self):
        """Update AI performance based on trade results"""
        try:
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            # Get recent completed trades
            cursor.execute('''
                SELECT symbol, side, ai_confidence, pnl, timestamp
                FROM live_trades 
                WHERE timestamp > datetime('now', '-24 hours')
                AND pnl IS NOT NULL
                ORDER BY timestamp DESC
            ''')
            
            trades = cursor.fetchall()
            
            if len(trades) >= 3:
                profitable_trades = [t for t in trades if t[3] > 0]
                win_rate = len(profitable_trades) / len(trades)
                avg_pnl = sum(t[3] for t in trades) / len(trades)
                
                logger.info(f"Performance Update: {len(trades)} trades, {win_rate:.2f} win rate, ${avg_pnl:.2f} avg PnL")
                
                # Store performance metrics
                conn_main = sqlite3.connect(self.db_path)
                cursor_main = conn_main.cursor()
                
                cursor_main.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        total_trades INTEGER,
                        win_rate REAL,
                        avg_pnl REAL,
                        best_trade REAL,
                        worst_trade REAL
                    )
                ''')
                
                best_trade = max(t[3] for t in trades) if trades else 0
                worst_trade = min(t[3] for t in trades) if trades else 0
                
                cursor_main.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, total_trades, win_rate, avg_pnl, best_trade, worst_trade)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    len(trades),
                    win_rate,
                    avg_pnl,
                    best_trade,
                    worst_trade
                ))
                
                conn_main.commit()
                conn_main.close()
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")

    def run_trading_cycle(self):
        """Run comprehensive trading cycle"""
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        all_signals = []
        
        logger.info("Running Comprehensive Trading Analysis")
        
        for symbol in crypto_symbols:
            try:
                df = self.get_market_data(symbol)
                if df is None:
                    continue
                
                # Generate both BUY and SELL signals
                buy_signal = self.generate_buy_signals(symbol, df)
                if buy_signal and buy_signal['confidence'] >= 50:
                    all_signals.append(buy_signal)
                
                sell_signal = self.generate_sell_signals(symbol, df)
                if sell_signal and sell_signal['confidence'] >= 50:
                    all_signals.append(sell_signal)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Analysis error for {symbol}: {e}")
        
        # Save generated signals
        if all_signals:
            self.save_signals(all_signals)
            
            buy_signals = [s for s in all_signals if s['signal'] == 'BUY']
            sell_signals = [s for s in all_signals if s['signal'] == 'SELL']
            
            logger.info(f"Generated {len(buy_signals)} BUY and {len(sell_signals)} SELL signals")
            
            for signal in all_signals:
                logger.info(f"{signal['symbol']}: {signal['signal']} ({signal['confidence']:.1f}%)")
        
        # Update performance metrics
        self.update_performance_metrics()
        
        return all_signals

    def start_continuous_trading(self):
        """Start continuous trading engine"""
        self.is_running = True
        logger.info("Advanced Trading Engine Started - Generating BUY and SELL signals")
        
        while self.is_running:
            try:
                # Run trading cycle every 10 minutes
                signals = self.run_trading_cycle()
                
                # Sleep for 10 minutes
                for i in range(600):  # 10 minutes = 600 seconds
                    if not self.is_running:
                        break
                    time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Trading engine stopped")
                break
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                time.sleep(60)  # Wait 1 minute before retry

def main():
    """Main function"""
    engine = ComprehensiveTradingEngine()
    engine.start_continuous_trading()

if __name__ == "__main__":
    main()