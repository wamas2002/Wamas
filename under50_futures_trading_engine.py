"""
Under $50 Futures Trading Engine
Specialized engine for trading cryptocurrency tokens priced under $50
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

class Under50FuturesEngine:
    def __init__(self):
        self.exchange = None
        self.db_path = 'under50_futures_trading.db'
        self.min_confidence = 58.0  # Lower threshold for under $50 tokens
        self.max_leverage = 4  # Higher leverage for smaller price movements
        self.max_position_size = 0.12  # 12% max position for volatility
        self.price_threshold = 50.0  # Only tokens under $50
        
        # Initial token list - will be filtered by price
        self.candidate_symbols = [
            'ADA/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'DOT/USDT:USDT',
            'AVAX/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'NEAR/USDT:USDT',
            'TRX/USDT:USDT', 'ICP/USDT:USDT', 'ALGO/USDT:USDT', 'HBAR/USDT:USDT',
            'XLM/USDT:USDT', 'SAND/USDT:USDT', 'MANA/USDT:USDT', 'THETA/USDT:USDT',
            'AXS/USDT:USDT', 'FIL/USDT:USDT', 'ETC/USDT:USDT', 'EGLD/USDT:USDT',
            'FLOW/USDT:USDT', 'ENJ/USDT:USDT', 'CHZ/USDT:USDT', 'CRV/USDT:USDT',
            'LTC/USDT:USDT', 'LINK/USDT:USDT', 'MATIC/USDT:USDT', 'FTM/USDT:USDT',
            'AAVE/USDT:USDT', 'GRT/USDT:USDT', 'SUSHI/USDT:USDT', 'COMP/USDT:USDT',
            'MKR/USDT:USDT', 'SNX/USDT:USDT', 'YFI/USDT:USDT', 'RUNE/USDT:USDT',
            'KAVA/USDT:USDT', 'WAVES/USDT:USDT', 'ZEC/USDT:USDT', 'DASH/USDT:USDT',
            'XMR/USDT:USDT', 'NEO/USDT:USDT', 'VET/USDT:USDT', 'IOTA/USDT:USDT',
            'ZIL/USDT:USDT', 'ONT/USDT:USDT', 'BAT/USDT:USDT', 'QTUM/USDT:USDT'
        ]
        
        self.active_symbols = []
        
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
            logger.info("Under $50 Futures Engine connected to OKX")
            return True
            
        except Exception as e:
            logger.error(f"OKX connection failed: {e}")
            return False
    
    def filter_symbols_by_price(self):
        """Filter symbols to only include tokens under $50"""
        filtered_symbols = []
        
        for symbol in self.candidate_symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker['last']
                
                if price and price < self.price_threshold:
                    filtered_symbols.append(symbol)
                    logger.info(f"✅ {symbol}: ${price:.4f} - INCLUDED")
                else:
                    logger.info(f"❌ {symbol}: ${price:.4f} - EXCLUDED (over $50)")
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
                continue
        
        self.active_symbols = filtered_symbols
        logger.info(f"🎯 Active symbols under $50: {len(self.active_symbols)} tokens")
        return len(self.active_symbols)
    
    def setup_database(self):
        """Setup under $50 futures database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS under50_futures_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    price_tier TEXT NOT NULL,
                    leverage REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    rsi REAL,
                    volume_spike BOOLEAN,
                    price_momentum REAL,
                    volatility_score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS under50_futures_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    leverage REAL NOT NULL,
                    confidence REAL NOT NULL,
                    price_tier TEXT NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    pnl_usd REAL DEFAULT 0,
                    pnl_percentage REAL DEFAULT 0,
                    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    exit_time TIMESTAMP,
                    exit_price REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Under $50 futures database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_price_tier(self, price: float) -> str:
        """Categorize price into tiers for different strategies"""
        if price < 0.01:
            return "MICRO"  # Under 1 cent
        elif price < 0.1:
            return "PENNY"  # 1-10 cents
        elif price < 1.0:
            return "CENT"   # 10 cents - $1
        elif price < 10.0:
            return "SINGLE" # $1 - $10
        else:
            return "DOUBLE" # $10 - $50
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data optimized for under $50 tokens"""
        try:
            # Use shorter timeframes for more responsive signals
            ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Technical indicators optimized for lower-priced tokens
            df['rsi'] = ta.rsi(df['close'], length=10)  # Faster RSI
            df['rsi_smooth'] = ta.rsi(df['close'], length=20)  # Smoother RSI
            
            # MACD with faster settings
            macd_data = ta.macd(df['close'], fast=8, slow=21, signal=9)
            df['macd'] = macd_data['MACD_8_21_9']
            df['macd_signal'] = macd_data['MACDs_8_21_9']
            df['macd_histogram'] = macd_data['MACDh_8_21_9']
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=15, std=2)
            df['bb_upper'] = bb['BBU_15_2.0']
            df['bb_lower'] = bb['BBL_15_2.0']
            df['bb_middle'] = bb['BBM_15_2.0']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_spike'] = df['volume_ratio'] > 1.5
            
            # Price momentum indicators
            df['price_change_3'] = df['close'].pct_change(3) * 100
            df['price_change_7'] = df['close'].pct_change(7) * 100
            df['price_change_15'] = df['close'].pct_change(15) * 100
            
            # Volatility measures
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean() * 100
            df['high_low_range'] = (df['high'] - df['low']) / df['close'] * 100
            
            # Moving averages
            df['ema_8'] = ta.ema(df['close'], length=8)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_cross'] = np.where(df['ema_8'] > df['ema_21'], 1, -1)
            
            return df.fillna(0)
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def generate_under50_signal(self, symbol: str) -> Optional[Dict]:
        """Generate trading signal optimized for under $50 tokens"""
        try:
            df = self.get_market_data(symbol)
            if df is None or len(df) < 30:
                return None
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            current_price = latest['close']
            
            # Skip if price is above threshold
            if current_price >= self.price_threshold:
                return None
            
            price_tier = self.get_price_tier(current_price)
            
            # Initialize scoring with tier-specific adjustments
            bullish_score = 50
            bearish_score = 50
            
            # RSI Analysis - More sensitive for lower-priced tokens
            rsi = latest['rsi']
            rsi_smooth = latest['rsi_smooth']
            
            if rsi < 25:  # Extreme oversold
                bullish_score += 20
            elif rsi < 35:  # Oversold
                bullish_score += 12
            elif rsi < 45:  # Bearish lean
                bullish_score += 6
            elif rsi > 75:  # Extreme overbought
                bearish_score += 20
            elif rsi > 65:  # Overbought
                bearish_score += 12
            elif rsi > 55:  # Bullish lean
                bearish_score += 6
            
            # MACD Analysis
            if latest['macd'] > latest['macd_signal']:
                if latest['macd_histogram'] > prev['macd_histogram']:  # Increasing momentum
                    bullish_score += 15
                else:
                    bullish_score += 8
            elif latest['macd'] < latest['macd_signal']:
                if latest['macd_histogram'] < prev['macd_histogram']:  # Decreasing momentum
                    bearish_score += 15
                else:
                    bearish_score += 8
            
            # EMA Cross Analysis
            if latest['ema_cross'] == 1 and prev['ema_cross'] != 1:  # Bullish cross
                bullish_score += 12
            elif latest['ema_cross'] == -1 and prev['ema_cross'] != -1:  # Bearish cross
                bearish_score += 12
            elif latest['ema_cross'] == 1:  # Continues bullish
                bullish_score += 6
            elif latest['ema_cross'] == -1:  # Continues bearish
                bearish_score += 6
            
            # Bollinger Bands - Adjusted for volatility
            bb_pos = latest['bb_position']
            bb_width = latest['bb_width']
            
            if bb_pos < 0.15:  # Very oversold
                bullish_score += 15
            elif bb_pos < 0.3:  # Oversold
                bullish_score += 10
            elif bb_pos > 0.85:  # Very overbought
                bearish_score += 15
            elif bb_pos > 0.7:  # Overbought
                bearish_score += 10
            
            # Volume Analysis - Critical for under $50 tokens
            volume_spike = latest['volume_spike']
            volume_ratio = latest['volume_ratio']
            
            if volume_spike:
                if latest['close'] > prev['close']:
                    bullish_score += 12
                else:
                    bearish_score += 12
            elif volume_ratio > 1.2:
                if latest['close'] > prev['close']:
                    bullish_score += 8
                else:
                    bearish_score += 8
            elif volume_ratio < 0.6:  # Low volume warning
                bullish_score -= 5
                bearish_score -= 5
            
            # Price Momentum Analysis
            momentum_3 = latest['price_change_3']
            momentum_7 = latest['price_change_7']
            momentum_15 = latest['price_change_15']
            
            # Short-term momentum (most important for under $50 tokens)
            if momentum_3 > 3:
                bullish_score += 12
            elif momentum_3 > 1:
                bullish_score += 6
            elif momentum_3 < -3:
                bearish_score += 12
            elif momentum_3 < -1:
                bearish_score += 6
            
            # Medium-term momentum
            if momentum_7 > 5:
                bullish_score += 8
            elif momentum_7 < -5:
                bearish_score += 8
            
            # Volatility Scoring - Higher volatility = higher opportunity
            volatility = latest['volatility']
            if volatility > 5:  # High volatility
                if momentum_3 > 0:
                    bullish_score += 8
                else:
                    bearish_score += 8
            elif volatility < 2:  # Low volatility - reduce confidence
                bullish_score -= 3
                bearish_score -= 3
            
            # Price tier adjustments
            if price_tier in ['MICRO', 'PENNY']:
                # More aggressive for micro/penny tokens
                bullish_score += 5
                bearish_score += 5
            elif price_tier == 'DOUBLE':
                # More conservative for higher-priced tokens
                bullish_score -= 3
                bearish_score -= 3
            
            # Determine signal
            confidence_diff = abs(bullish_score - bearish_score)
            
            if bullish_score > bearish_score and confidence_diff >= 8:
                signal = 'LONG'
                confidence = min(95, 50 + confidence_diff * 1.8)
            elif bearish_score > bullish_score and confidence_diff >= 8:
                signal = 'SHORT'
                confidence = min(95, 50 + confidence_diff * 1.8)
            else:
                return None  # No clear signal
            
            # Dynamic targets based on price tier and volatility
            volatility_factor = max(0.02, volatility / 100)
            
            if price_tier in ['MICRO', 'PENNY']:
                # Wider stops for micro/penny tokens
                stop_factor = max(0.08, volatility_factor * 2)
                profit_factor = max(0.15, volatility_factor * 3)
            elif price_tier == 'CENT':
                stop_factor = max(0.06, volatility_factor * 1.5)
                profit_factor = max(0.12, volatility_factor * 2.5)
            elif price_tier == 'SINGLE':
                stop_factor = max(0.04, volatility_factor * 1.2)
                profit_factor = max(0.09, volatility_factor * 2)
            else:  # DOUBLE
                stop_factor = max(0.03, volatility_factor)
                profit_factor = max(0.06, volatility_factor * 1.5)
            
            if signal == 'LONG':
                stop_loss = current_price * (1 - stop_factor)
                take_profit = current_price * (1 + profit_factor)
            else:  # SHORT
                stop_loss = current_price * (1 + stop_factor)
                take_profit = current_price * (1 - profit_factor)
            
            # Leverage based on confidence, volatility, and price tier
            if confidence > 85 and volatility < 3:
                leverage = 4
            elif confidence > 75 and volatility < 5:
                leverage = 3
            elif confidence > 65:
                leverage = 2
            else:
                leverage = 1
            
            # Adjust leverage for price tiers
            if price_tier in ['MICRO', 'PENNY']:
                leverage = min(leverage, 2)  # Lower leverage for micro tokens
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': round(confidence, 1),
                'price': current_price,
                'price_tier': price_tier,
                'leverage': leverage,
                'stop_loss': round(stop_loss, 8),
                'take_profit': round(take_profit, 8),
                'rsi': round(rsi, 1),
                'volume_spike': bool(volume_spike),
                'price_momentum': round(momentum_3, 2),
                'volatility_score': round(volatility, 2),
                'bb_position': round(bb_pos, 3),
                'volume_ratio': round(volume_ratio, 2),
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
                INSERT INTO under50_futures_signals 
                (symbol, signal, confidence, price, price_tier, leverage, stop_loss, take_profit,
                 rsi, volume_spike, price_momentum, volatility_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'], signal['signal'], signal['confidence'],
                signal['price'], signal['price_tier'], signal['leverage'],
                signal['stop_loss'], signal['take_profit'], signal['rsi'],
                signal['volume_spike'], signal['price_momentum'], signal['volatility_score']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
    
    def simulate_trade_execution(self, signal: Dict) -> bool:
        """Simulate trade execution for under $50 tokens"""
        try:
            # Calculate position size based on price tier
            balance = 1000  # Simulated USDT balance
            
            # Adjust position size based on price tier
            if signal['price_tier'] in ['MICRO', 'PENNY']:
                max_position_value = balance * 0.08  # 8% for micro tokens
            elif signal['price_tier'] == 'CENT':
                max_position_value = balance * 0.10  # 10% for cent tokens
            elif signal['price_tier'] == 'SINGLE':
                max_position_value = balance * 0.12  # 12% for single-digit tokens
            else:  # DOUBLE
                max_position_value = balance * 0.15  # 15% for double-digit tokens
            
            position_size = max_position_value / signal['price']
            
            # Calculate potential P&L
            if signal['signal'] == 'LONG':
                potential_pnl = (signal['take_profit'] - signal['price']) / signal['price'] * 100
            else:
                potential_pnl = (signal['price'] - signal['take_profit']) / signal['price'] * 100
            
            leveraged_pnl = potential_pnl * signal['leverage']
            
            # Save simulated trade
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO under50_futures_trades 
                (symbol, side, size, entry_price, leverage, confidence, price_tier, pnl_percentage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'], signal['signal'], position_size,
                signal['price'], signal['leverage'], signal['confidence'],
                signal['price_tier'], leveraged_pnl
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"🚀 UNDER $50 FUTURES: {signal['symbol']} ({signal['price_tier']}) {signal['signal']} "
                       f"${signal['price']:.6f} (Conf: {signal['confidence']}%, "
                       f"Leverage: {signal['leverage']}x, Potential: {leveraged_pnl:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def scan_under50_opportunities(self) -> List[Dict]:
        """Scan for under $50 futures opportunities"""
        signals = []
        
        for symbol in self.active_symbols:
            try:
                signal = self.generate_under50_signal(symbol)
                if signal and signal['confidence'] >= self.min_confidence:
                    signals.append(signal)
                    self.save_signal(signal)
                    
                time.sleep(0.25)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Scan failed for {symbol}: {e}")
                continue
        
        return signals
    
    def run_under50_futures_cycle(self):
        """Run under $50 futures trading cycle"""
        try:
            logger.info("🔄 Starting under $50 futures scan...")
            
            # Refresh symbol list every 10 cycles
            if not hasattr(self, '_cycle_count'):
                self._cycle_count = 0
            
            if self._cycle_count % 10 == 0:
                self.filter_symbols_by_price()
            
            self._cycle_count += 1
            
            if not self.active_symbols:
                logger.warning("No active symbols under $50 found")
                return
            
            signals = self.scan_under50_opportunities()
            executed_trades = 0
            
            for signal in signals:
                if self.simulate_trade_execution(signal):
                    executed_trades += 1
            
            logger.info(f"✅ Under $50 futures scan complete: {len(signals)} signals, {executed_trades} trades executed")
            
            # Summary by price tier
            tier_summary = {}
            for signal in signals:
                tier = signal['price_tier']
                if tier not in tier_summary:
                    tier_summary[tier] = 0
                tier_summary[tier] += 1
            
            if tier_summary:
                tier_info = ", ".join([f"{tier}: {count}" for tier, count in tier_summary.items()])
                logger.info(f"📊 Signals by tier: {tier_info}")
            
        except Exception as e:
            logger.error(f"Under $50 futures cycle failed: {e}")

def main():
    """Main under $50 futures function"""
    try:
        engine = Under50FuturesEngine()
        
        if not engine.initialize_exchange():
            logger.error("Failed to initialize exchange")
            return
        
        engine.setup_database()
        
        logger.info("🚀 Starting Under $50 Futures Trading Engine")
        logger.info(f"Configuration: Price Threshold: ${engine.price_threshold}, Min Confidence: {engine.min_confidence}%")
        
        # Initial symbol filtering
        active_count = engine.filter_symbols_by_price()
        if active_count == 0:
            logger.error("No tokens under $50 found")
            return
        
        while True:
            engine.run_under50_futures_cycle()
            logger.info("⏰ Next under $50 futures scan in 240 seconds...")
            time.sleep(240)  # 4 minutes
            
    except KeyboardInterrupt:
        logger.info("Under $50 futures engine stopped")
    except Exception as e:
        logger.error(f"Under $50 futures engine failed: {e}")

if __name__ == "__main__":
    main()