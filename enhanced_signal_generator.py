#!/usr/bin/env python3
"""
Enhanced Signal Generator with Optimized Parameters
Uses the new trading parameters to generate more active BUY signals
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

class EnhancedSignalGenerator:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.exchange = None
        self.initialize_exchange()
        self.load_optimized_parameters()
        
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
                    'rateLimit': 1200,
                    'enableRateLimit': True,
                })
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
    
    def load_optimized_parameters(self):
        """Load optimized trading parameters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM trading_parameters ORDER BY timestamp DESC LIMIT 1")
            params = cursor.fetchone()
            
            if params:
                self.confidence_threshold = params[1]
                self.position_size_pct = params[2]
                self.max_positions = params[5]
            else:
                self.confidence_threshold = 45.0
                self.position_size_pct = 3.5
                self.max_positions = 6
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            self.confidence_threshold = 45.0
            self.position_size_pct = 3.5
            self.max_positions = 6
    
    def get_portfolio_allocation(self):
        """Get current portfolio allocation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT symbol, quantity, current_price FROM portfolio")
            holdings = cursor.fetchall()
            conn.close()
            
            total_value = 0
            allocations = {}
            
            for symbol, quantity, price in holdings:
                if price and price > 0:
                    value = quantity * price
                    total_value += value
                    allocations[symbol] = value
            
            # Calculate percentages
            for symbol in allocations:
                allocations[symbol] = (allocations[symbol] / total_value) * 100 if total_value > 0 else 0
            
            return allocations, total_value
            
        except Exception as e:
            logger.error(f"Portfolio allocation error: {e}")
            return {}, 0
    
    def get_target_allocations(self):
        """Get target allocation percentages"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT symbol, target_percentage FROM allocation_targets WHERE active = 1")
            targets = cursor.fetchall()
            conn.close()
            
            return {symbol: target for symbol, target in targets}
            
        except Exception as e:
            logger.error(f"Target allocation error: {e}")
            return {'BTC': 25, 'ETH': 20, 'SOL': 12, 'ADA': 8, 'DOT': 6, 'AVAX': 4, 'USDT': 25}
    
    def calculate_rebalancing_signals(self):
        """Generate BUY signals based on allocation gaps"""
        current_allocations, total_value = self.get_portfolio_allocation()
        target_allocations = self.get_target_allocations()
        
        signals = []
        crypto_tokens = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        
        for token in crypto_tokens:
            current_pct = current_allocations.get(token, 0)
            target_pct = target_allocations.get(token, 0)
            
            allocation_gap = target_pct - current_pct
            
            # Generate BUY signal if significantly underallocated
            if allocation_gap > 5.0:  # More than 5% underallocated
                confidence = min(85, 50 + (allocation_gap * 2))  # Higher confidence for larger gaps
                
                signal = {
                    'symbol': token,
                    'signal': 'BUY',
                    'confidence': confidence,
                    'reasoning': f'Portfolio rebalancing: {current_pct:.1f}% current vs {target_pct}% target (gap: {allocation_gap:.1f}%)',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'rebalancing'
                }
                
                signals.append(signal)
        
        return signals
    
    def generate_technical_signals(self):
        """Generate technical analysis signals"""
        signals = []
        crypto_tokens = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        
        for token in crypto_tokens:
            try:
                symbol = f"{token}/USDT"
                
                # Get recent price data
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Calculate technical indicators
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['rsi'] = self.calculate_rsi(df['close'])
                
                current_price = df['close'].iloc[-1]
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                rsi = df['rsi'].iloc[-1]
                
                # Signal logic with lower thresholds
                signal_strength = 0
                reasoning_parts = []
                
                # Price above moving averages (bullish)
                if current_price > sma_20:
                    signal_strength += 15
                    reasoning_parts.append("price above SMA20")
                
                if current_price > sma_50:
                    signal_strength += 10
                    reasoning_parts.append("price above SMA50")
                
                # SMA crossover
                if sma_20 > sma_50:
                    signal_strength += 15
                    reasoning_parts.append("SMA20 > SMA50")
                
                # RSI conditions (more lenient)
                if 30 < rsi < 70:
                    signal_strength += 10
                    reasoning_parts.append(f"RSI {rsi:.1f} in healthy range")
                elif rsi < 40:
                    signal_strength += 20
                    reasoning_parts.append(f"RSI {rsi:.1f} oversold")
                
                # Volume confirmation
                recent_volume = df['volume'].tail(5).mean()
                avg_volume = df['volume'].mean()
                if recent_volume > avg_volume * 1.1:
                    signal_strength += 10
                    reasoning_parts.append("increased volume")
                
                # Generate signal if strength is sufficient
                if signal_strength >= 35:  # Lowered threshold
                    confidence = min(85, signal_strength + 10)
                    
                    signal = {
                        'symbol': token,
                        'signal': 'BUY',
                        'confidence': confidence,
                        'reasoning': f'Technical analysis: {", ".join(reasoning_parts)}',
                        'timestamp': datetime.now().isoformat(),
                        'type': 'technical'
                    }
                    
                    signals.append(signal)
                
            except Exception as e:
                logger.error(f"Technical analysis error for {token}: {e}")
        
        return signals
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def save_signals(self, signals):
        """Save generated signals to database"""
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
            
            logger.info(f"Saved {len(signals)} new signals")
            
        except Exception as e:
            logger.error(f"Error saving signals: {e}")
    
    def generate_enhanced_signals(self):
        """Generate enhanced signals using optimized parameters"""
        all_signals = []
        
        # Generate rebalancing signals
        rebalancing_signals = self.calculate_rebalancing_signals()
        all_signals.extend(rebalancing_signals)
        
        # Generate technical signals
        technical_signals = self.generate_technical_signals()
        all_signals.extend(technical_signals)
        
        # Filter by confidence threshold
        filtered_signals = [s for s in all_signals if s['confidence'] >= self.confidence_threshold]
        
        # Save to database
        if filtered_signals:
            self.save_signals(filtered_signals)
            
            logger.info(f"Generated {len(filtered_signals)} signals above {self.confidence_threshold}% confidence")
            for signal in filtered_signals:
                logger.info(f"{signal['symbol']}: {signal['signal']} ({signal['confidence']:.1f}%)")
        
        return filtered_signals

def run_enhanced_signal_generation():
    """Run enhanced signal generation"""
    generator = EnhancedSignalGenerator()
    signals = generator.generate_enhanced_signals()
    return signals

if __name__ == "__main__":
    run_enhanced_signal_generation()