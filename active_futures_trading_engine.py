"""
Active Futures Trading Engine
Optimized for frequent signal generation and execution
"""

import os
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas_ta as ta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActiveFuturesEngine:
    def __init__(self):
        self.exchange = None
        self.db_path = 'active_futures_trading.db'
        self.min_confidence = 60.0  # Lower threshold for more activity
        self.max_leverage = 3
        self.max_position_size = 0.08  # 8% max position
        self.symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 
            'ADA/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT', 'LINK/USDT:USDT',
            'LTC/USDT:USDT', 'DOT/USDT:USDT', 'AVAX/USDT:USDT', 'UNI/USDT:USDT'
        ]
        
    def initialize_exchange(self):
        """Initialize OKX futures connection"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            
            balance = self.exchange.fetch_balance()
            logger.info("Active Futures Engine connected to OKX")
            return True
            
        except Exception as e:
            logger.error(f"OKX connection failed: {e}")
            return False
    
    def setup_database(self):
        """Setup active futures database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_futures_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    leverage REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    rsi REAL,
                    macd_signal TEXT,
                    volume_signal TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_futures_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    leverage REAL NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    pnl_usd REAL DEFAULT 0,
                    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    exit_time TIMESTAMP,
                    exit_price REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Active futures database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
            df['macd_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
            df['macd_histogram'] = ta.macd(df['close'])['MACDh_12_26_9']
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['price_change_5'] = df['close'].pct_change(5) * 100
            df['price_change_10'] = df['close'].pct_change(10) * 100
            
            return df.fillna(0)
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def generate_active_signal(self, symbol: str) -> Optional[Dict]:
        """Generate active trading signal with lower thresholds"""
        try:
            df = self.get_market_data(symbol)
            if df is None or len(df) < 20:
                return None
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Initialize scoring
            bullish_score = 50
            bearish_score = 50
            
            # RSI Analysis - More sensitive
            rsi = latest['rsi']
            if rsi < 40:
                bullish_score += 15
            elif rsi > 60:
                bearish_score += 15
            elif rsi < 45:
                bullish_score += 8
            elif rsi > 55:
                bearish_score += 8
            
            # MACD Analysis
            macd_signal = "NEUTRAL"
            if latest['macd'] > latest['macd_signal']:
                if latest['macd_histogram'] > 0:
                    bullish_score += 12
                    macd_signal = "BULLISH"
                else:
                    bullish_score += 6
                    macd_signal = "WEAK_BULLISH"
            elif latest['macd'] < latest['macd_signal']:
                if latest['macd_histogram'] < 0:
                    bearish_score += 12
                    macd_signal = "BEARISH"
                else:
                    bearish_score += 6
                    macd_signal = "WEAK_BEARISH"
            
            # Bollinger Bands
            bb_pos = latest['bb_position']
            if bb_pos < 0.25:
                bullish_score += 10
            elif bb_pos > 0.75:
                bearish_score += 10
            elif bb_pos < 0.4:
                bullish_score += 5
            elif bb_pos > 0.6:
                bearish_score += 5
            
            # Volume Analysis
            volume_signal = "NORMAL"
            vol_ratio = latest['volume_ratio']
            if vol_ratio > 1.2:
                # High volume confirms trend
                volume_signal = "HIGH"
                if latest['close'] > prev['close']:
                    bullish_score += 8
                else:
                    bearish_score += 8
            elif vol_ratio > 1.0:
                volume_signal = "ELEVATED"
                if latest['close'] > prev['close']:
                    bullish_score += 4
                else:
                    bearish_score += 4
            
            # Price momentum
            momentum_5 = latest['price_change_5']
            momentum_10 = latest['price_change_10']
            
            if momentum_5 > 1:
                bullish_score += 8
            elif momentum_5 < -1:
                bearish_score += 8
            
            if momentum_10 > 2:
                bullish_score += 6
            elif momentum_10 < -2:
                bearish_score += 6
            
            # Determine signal
            confidence_diff = abs(bullish_score - bearish_score)
            
            if bullish_score > bearish_score and confidence_diff >= 10:
                signal = 'LONG'
                confidence = min(95, 50 + confidence_diff * 1.5)
            elif bearish_score > bullish_score and confidence_diff >= 10:
                signal = 'SHORT'
                confidence = min(95, 50 + confidence_diff * 1.5)
            else:
                return None  # No clear signal
            
            # Calculate targets
            current_price = latest['close']
            
            # Dynamic stop loss and take profit based on volatility
            price_std = df['close'].rolling(20).std().iloc[-1]
            volatility_factor = (price_std / current_price) * 100
            
            if signal == 'LONG':
                stop_loss = current_price * (1 - max(0.03, volatility_factor * 0.4))
                take_profit = current_price * (1 + max(0.06, volatility_factor * 0.8))
            else:  # SHORT
                stop_loss = current_price * (1 + max(0.03, volatility_factor * 0.4))
                take_profit = current_price * (1 - max(0.06, volatility_factor * 0.8))
            
            # Leverage based on confidence and volatility
            if confidence > 80 and volatility_factor < 2:
                leverage = 3
            elif confidence > 70:
                leverage = 2
            else:
                leverage = 1
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': round(confidence, 1),
                'price': current_price,
                'leverage': leverage,
                'stop_loss': round(stop_loss, 6),
                'take_profit': round(take_profit, 6),
                'rsi': round(rsi, 1),
                'macd_signal': macd_signal,
                'volume_signal': volume_signal,
                'volatility': round(volatility_factor, 2),
                'momentum_5': round(momentum_5, 2),
                'momentum_10': round(momentum_10, 2),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None
    
    def save_signal(self, signal: Dict):
        """Save signal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO active_futures_signals 
                (symbol, signal, confidence, price, leverage, stop_loss, take_profit,
                 rsi, macd_signal, volume_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'], signal['signal'], signal['confidence'],
                signal['price'], signal['leverage'], signal['stop_loss'],
                signal['take_profit'], signal['rsi'], signal['macd_signal'],
                signal['volume_signal']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
    
    def simulate_trade_execution(self, signal: Dict) -> bool:
        """Simulate trade execution for testing"""
        try:
            # Calculate position size
            balance = 1000  # Simulated USDT balance
            max_position_value = balance * self.max_position_size
            position_size = max_position_value / signal['price']
            
            # Save simulated trade
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO active_futures_trades 
                (symbol, side, size, entry_price, leverage, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'], signal['signal'], position_size,
                signal['price'], signal['leverage'], signal['confidence']
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"🚀 FUTURES TRADE: {signal['symbol']} {signal['signal']} "
                       f"(Conf: {signal['confidence']}%, Leverage: {signal['leverage']}x)")
            
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def scan_active_opportunities(self) -> List[Dict]:
        """Scan for active futures opportunities"""
        signals = []
        
        for symbol in self.symbols:
            try:
                signal = self.generate_active_signal(symbol)
                if signal and signal['confidence'] >= self.min_confidence:
                    signals.append(signal)
                    self.save_signal(signal)
                    
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Scan failed for {symbol}: {e}")
                continue
        
        return signals
    
    def run_active_futures_cycle(self):
        """Run active futures trading cycle"""
        try:
            logger.info("🔄 Starting active futures scan...")
            
            signals = self.scan_active_opportunities()
            executed_trades = 0
            
            for signal in signals:
                if self.simulate_trade_execution(signal):
                    executed_trades += 1
            
            logger.info(f"✅ Active futures scan complete: {len(signals)} signals, {executed_trades} trades executed")
            
        except Exception as e:
            logger.error(f"Active futures cycle failed: {e}")

def main():
    """Main active futures function"""
    try:
        engine = ActiveFuturesEngine()
        
        if not engine.initialize_exchange():
            logger.error("Failed to initialize exchange")
            return
        
        engine.setup_database()
        
        logger.info("🚀 Starting Active Futures Trading Engine")
        logger.info(f"Configuration: Min Confidence: {engine.min_confidence}%, Max Leverage: {engine.max_leverage}x")
        
        while True:
            engine.run_active_futures_cycle()
            logger.info("⏰ Next active futures scan in 180 seconds...")
            time.sleep(180)  # 3 minutes for more frequent scanning
            
    except KeyboardInterrupt:
        logger.info("Active futures engine stopped")
    except Exception as e:
        logger.error(f"Active futures engine failed: {e}")

if __name__ == "__main__":
    main()