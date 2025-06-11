#!/usr/bin/env python3
"""
Smart SELL Signal Generator
Advanced ML-driven system to generate profitable SELL signals for active trading
"""

import sqlite3
import ccxt
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartSellSignalGenerator:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.exchange = None
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
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")

    def get_current_positions(self):
        """Get current cryptocurrency positions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT symbol, quantity, current_price FROM portfolio WHERE symbol != 'USDT' AND quantity > 0")
            positions = cursor.fetchall()
            conn.close()
            
            return [(symbol, float(quantity), float(price)) for symbol, quantity, price in positions]
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

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

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for SELL signal analysis"""
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
        df['momentum_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=14).std() / df['close'].rolling(window=14).mean()
        
        return df

    def detect_sell_signals(self, symbol, df):
        """Detect SELL signals based on technical analysis"""
        
        if len(df) < 50:
            return None
            
        df = self.calculate_technical_indicators(df)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        sell_score = 0
        reasons = []
        
        # RSI Overbought (Strong SELL signal)
        if current['rsi'] > 75:
            sell_score += 25
            reasons.append(f"RSI overbought ({current['rsi']:.1f})")
        elif current['rsi'] > 70:
            sell_score += 15
            reasons.append(f"RSI high ({current['rsi']:.1f})")
        
        # MACD Bearish signals
        if current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            sell_score += 20
            reasons.append("MACD bearish crossover")
        elif current['macd_histogram'] < 0 and current['macd_histogram'] < prev['macd_histogram']:
            sell_score += 10
            reasons.append("MACD histogram declining")
        
        # Price vs Bollinger Bands
        if current['close'] > current['bb_upper']:
            sell_score += 15
            reasons.append("Price above upper Bollinger Band")
        elif current['close'] > current['bb_middle'] * 1.05:
            sell_score += 8
            reasons.append("Price extended above BB middle")
        
        # Moving Average signals
        if current['close'] < current['sma_20'] and prev['close'] >= prev['sma_20']:
            sell_score += 15
            reasons.append("Price broke below SMA20")
        elif current['sma_20'] < current['sma_50']:
            sell_score += 10
            reasons.append("SMA20 below SMA50 (bearish)")
        
        # Volume confirmation
        if current['volume_spike'] > 1.5:
            sell_score += 8
            reasons.append("High volume confirmation")
        
        # Momentum analysis
        if current['momentum_5'] < -3:
            sell_score += 12
            reasons.append(f"Negative 5-period momentum ({current['momentum_5']:.1f}%)")
        if current['momentum_10'] < -5:
            sell_score += 15
            reasons.append(f"Strong negative 10-period momentum ({current['momentum_10']:.1f}%)")
        
        # Recent high analysis (resistance)
        recent_high = df['high'].tail(20).max()
        if current['close'] >= recent_high * 0.98:
            sell_score += 10
            reasons.append("Near recent high (resistance)")
        
        # Volatility spike (risk management)
        if current['volatility'] > df['volatility'].tail(20).mean() * 1.5:
            sell_score += 8
            reasons.append("High volatility spike")
        
        # Generate signal if score is sufficient
        if sell_score >= 50:
            confidence = min(95, sell_score + 10)
            
            return {
                'symbol': symbol,
                'signal': 'SELL',
                'confidence': confidence,
                'reasoning': f"Technical SELL analysis: {', '.join(reasons)}",
                'sell_score': sell_score,
                'timestamp': datetime.now().isoformat()
            }
        
        return None

    def analyze_profit_taking_opportunities(self):
        """Analyze current positions for profit-taking SELL signals"""
        positions = self.get_current_positions()
        sell_signals = []
        
        for symbol, quantity, current_price in positions:
            try:
                # Get recent trades for this symbol to calculate average buy price
                conn = sqlite3.connect('live_trading.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT price, amount FROM live_trades 
                    WHERE symbol = ? AND side = 'BUY' 
                    ORDER BY timestamp DESC LIMIT 10
                ''', (f"{symbol}/USDT",))
                
                buy_trades = cursor.fetchall()
                conn.close()
                
                if buy_trades:
                    # Calculate weighted average buy price
                    total_cost = sum(float(price) * float(amount) for price, amount in buy_trades)
                    total_amount = sum(float(amount) for price, amount in buy_trades)
                    avg_buy_price = total_cost / total_amount if total_amount > 0 else current_price
                    
                    profit_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100
                    
                    # Profit-taking logic
                    if profit_pct > 8:  # 8%+ profit
                        confidence = min(85, 60 + profit_pct)
                        sell_signals.append({
                            'symbol': symbol,
                            'signal': 'SELL',
                            'confidence': confidence,
                            'reasoning': f"Profit taking: {profit_pct:.1f}% gain (bought avg ${avg_buy_price:.2f}, now ${current_price:.2f})",
                            'profit_pct': profit_pct,
                            'timestamp': datetime.now().isoformat()
                        })
                    elif profit_pct > 5:  # 5%+ profit with technical confirmation
                        # Get technical analysis
                        df = self.get_market_data(symbol)
                        if df is not None:
                            tech_signal = self.detect_sell_signals(symbol, df)
                            if tech_signal and tech_signal['confidence'] > 60:
                                sell_signals.append({
                                    'symbol': symbol,
                                    'signal': 'SELL',
                                    'confidence': min(80, tech_signal['confidence']),
                                    'reasoning': f"Profit taking + technical: {profit_pct:.1f}% gain, {tech_signal['reasoning']}",
                                    'profit_pct': profit_pct,
                                    'timestamp': datetime.now().isoformat()
                                })
                
            except Exception as e:
                logger.error(f"Error analyzing profit taking for {symbol}: {e}")
        
        return sell_signals

    def generate_technical_sell_signals(self):
        """Generate SELL signals based on technical analysis for all tracked symbols"""
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        sell_signals = []
        
        for symbol in crypto_symbols:
            try:
                df = self.get_market_data(symbol)
                if df is not None:
                    signal = self.detect_sell_signals(symbol, df)
                    if signal and signal['confidence'] >= 55:
                        sell_signals.append(signal)
                        
            except Exception as e:
                logger.error(f"Error generating technical SELL signal for {symbol}: {e}")
        
        return sell_signals

    def save_sell_signals(self, signals):
        """Save SELL signals to database"""
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
            
            logger.info(f"Saved {len(signals)} SELL signals")
            
        except Exception as e:
            logger.error(f"Error saving SELL signals: {e}")

    def run_sell_signal_generation(self):
        """Main function to generate comprehensive SELL signals"""
        all_sell_signals = []
        
        logger.info("🔴 Generating Smart SELL Signals")
        
        # Profit-taking analysis for current positions
        profit_signals = self.analyze_profit_taking_opportunities()
        all_sell_signals.extend(profit_signals)
        
        # Technical analysis SELL signals
        technical_signals = self.generate_technical_sell_signals()
        all_sell_signals.extend(technical_signals)
        
        # Save all signals
        if all_sell_signals:
            self.save_sell_signals(all_sell_signals)
            
            logger.info(f"Generated {len(all_sell_signals)} SELL signals:")
            for signal in all_sell_signals:
                logger.info(f"{signal['symbol']}: SELL ({signal['confidence']:.1f}%)")
        else:
            logger.info("No SELL signals generated at this time")
        
        return all_sell_signals

def run_smart_sell_generator():
    """Run the smart SELL signal generator"""
    generator = SmartSellSignalGenerator()
    return generator.run_sell_signal_generation()

if __name__ == "__main__":
    run_smart_sell_generator()