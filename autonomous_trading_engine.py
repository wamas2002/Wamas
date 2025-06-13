"""
Autonomous Trading Engine
Continuously scans markets and executes trades based on AI signals and technical analysis
"""

import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import sqlite3
import logging
import time
import json
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import threading
from decimal import Decimal, ROUND_DOWN
from gpt_enhanced_trading_analyzer import GPTEnhancedTradingAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutonomousTradingEngine:
    def __init__(self):
        """Initialize Autonomous Trading Engine"""
        self.exchange = None
        self.running = False
        self.scan_interval = 300  # 5 minutes between scans
        self.min_confidence = 70  # Minimum confidence for automatic execution
        self.max_position_size = 0.25  # Maximum 25% allocation per trade
        self.stop_loss_pct = 0.08  # 8% stop loss
        self.take_profit_pct = 0.15  # 15% take profit
        
        # AI Models
        self.trend_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.momentum_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # GPT Enhancement
        self.gpt_analyzer = None
        self.gpt_enhancement_enabled = True
        
        # Active positions tracking
        self.active_positions = {}
        self.trade_history = []
        
        # Trading pairs (top liquid pairs)
        self.trading_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'LINK/USDT', 'LTC/USDT', 'DOT/USDT',
            'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT', 'FTM/USDT'
        ]
        
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            logger.info("Autonomous trading engine connected to OKX")
            return True
            
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
            return False
    
    def setup_database(self):
        """Setup autonomous trading database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Autonomous trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS autonomous_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        amount REAL NOT NULL,
                        price REAL NOT NULL,
                        confidence REAL NOT NULL,
                        ai_score REAL NOT NULL,
                        technical_score REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        status TEXT DEFAULT 'PENDING',
                        order_id TEXT,
                        timestamp TEXT NOT NULL,
                        exit_price REAL,
                        exit_timestamp TEXT,
                        pnl REAL DEFAULT 0
                    )
                ''')
                
                # Position monitoring table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS active_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL UNIQUE,
                        side TEXT NOT NULL,
                        amount REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        current_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        confidence REAL,
                        entry_timestamp TEXT,
                        last_update TEXT,
                        pnl REAL DEFAULT 0
                    )
                ''')
                
                conn.commit()
                logger.info("Autonomous trading database initialized")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_market_data(self, symbol, timeframe='1h', limit=100):
        """Get market data with technical indicators"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 50:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate indicators
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                df['macd'] = macd.iloc[:, 0]
                df['macd_signal'] = macd.iloc[:, 1]
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20)
            if bb is not None and not bb.empty:
                df['bb_upper'] = bb.iloc[:, 0]
                df['bb_middle'] = bb.iloc[:, 1]
                df['bb_lower'] = bb.iloc[:, 2]
            
            # Volume analysis
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_ai_score(self, df):
        """Calculate AI-enhanced signal score"""
        try:
            if len(df) < 20:
                return 50
                
            # Price momentum
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            momentum_score = min(50, max(-50, price_change * 1000))
            
            # Volume momentum
            volume_trend = df['volume_ratio'].tail(3).mean()
            volume_score = min(25, volume_trend * 10)
            
            # Trend consistency
            ema_trend = 0
            if df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1]:
                ema_trend += 1
            if df['ema_21'].iloc[-1] > df['ema_50'].iloc[-1]:
                ema_trend += 1
            
            trend_score = ema_trend * 12.5
            
            return max(0, min(100, momentum_score + volume_score + trend_score + 50))
            
        except Exception as e:
            logger.error(f"AI score calculation failed: {e}")
            return 50
    
    def analyze_trading_signal(self, symbol):
        """Analyze symbol for trading signals"""
        try:
            df = self.get_market_data(symbol, '1h', 100)
            if df is None or len(df) < 50:
                return None
                
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Technical analysis scoring
            technical_score = 0
            signal = 'HOLD'
            entry_reasons = []
            
            # RSI analysis
            rsi = latest.get('rsi', 50)
            if rsi < 25:  # Strong oversold
                technical_score += 30
                signal = 'BUY'
                entry_reasons.append('RSI_OVERSOLD')
            elif rsi > 75:  # Strong overbought
                technical_score += 30
                signal = 'SELL'
                entry_reasons.append('RSI_OVERBOUGHT')
            elif rsi < 35 and signal != 'SELL':
                technical_score += 15
                signal = 'BUY'
                entry_reasons.append('RSI_BULLISH')
            elif rsi > 65 and signal != 'BUY':
                technical_score += 15
                signal = 'SELL'
                entry_reasons.append('RSI_BEARISH')
            
            # EMA trend analysis
            ema_9 = latest.get('ema_9', latest['close'])
            ema_21 = latest.get('ema_21', latest['close'])
            ema_50 = latest.get('ema_50', latest['close'])
            
            # Strong uptrend
            if ema_9 > ema_21 > ema_50 and latest['close'] > ema_9:
                technical_score += 25
                if signal != 'SELL':
                    signal = 'BUY'
                entry_reasons.append('STRONG_UPTREND')
            # Strong downtrend
            elif ema_9 < ema_21 < ema_50 and latest['close'] < ema_9:
                technical_score += 25
                if signal != 'BUY':
                    signal = 'SELL'
                entry_reasons.append('STRONG_DOWNTREND')
            
            # MACD crossover
            macd = latest.get('macd', 0)
            macd_signal_val = latest.get('macd_signal', 0)
            prev_macd = prev.get('macd', 0)
            prev_macd_signal = prev.get('macd_signal', 0)
            
            # Bullish MACD crossover
            if macd > macd_signal_val and prev_macd <= prev_macd_signal:
                technical_score += 20
                if signal != 'SELL':
                    signal = 'BUY'
                entry_reasons.append('MACD_BULLISH_CROSS')
            # Bearish MACD crossover
            elif macd < macd_signal_val and prev_macd >= prev_macd_signal:
                technical_score += 20
                if signal != 'BUY':
                    signal = 'SELL'
                entry_reasons.append('MACD_BEARISH_CROSS')
            
            # Volume confirmation
            volume_ratio = latest.get('volume_ratio', 1)
            if volume_ratio > 2.0:  # High volume confirmation
                technical_score += 15
                entry_reasons.append('HIGH_VOLUME')
            elif volume_ratio > 1.5:
                technical_score += 10
                entry_reasons.append('ELEVATED_VOLUME')
            
            # Bollinger Bands analysis
            bb_upper = latest.get('bb_upper')
            bb_lower = latest.get('bb_lower')
            if bb_upper and bb_lower:
                if latest['close'] < bb_lower:  # Below lower band
                    technical_score += 15
                    if signal != 'SELL':
                        signal = 'BUY'
                    entry_reasons.append('BB_OVERSOLD')
                elif latest['close'] > bb_upper:  # Above upper band
                    technical_score += 15
                    if signal != 'BUY':
                        signal = 'SELL'
                    entry_reasons.append('BB_OVERBOUGHT')
            
            # Calculate AI score
            ai_score = self.calculate_ai_score(df)
            
            # Combined confidence
            base_confidence = (technical_score * 0.6) + (ai_score * 0.4)
            base_confidence = min(100, base_confidence)
            
            # GPT Enhancement
            if self.gpt_enhancement_enabled and self.gpt_analyzer and signal != 'HOLD':
                try:
                    gpt_analysis = self.gpt_analyzer.analyze_signal_with_gpt(
                        symbol=symbol,
                        technical_confidence=base_confidence,
                        ai_score=ai_score,
                        signal_type=signal
                    )
                    
                    confidence = gpt_analysis['enhanced_confidence']
                    entry_reasons.append(f"GPT_{gpt_analysis['risk_level'].upper()}")
                    
                    logger.info(f"🧠 GPT Enhanced {symbol}: {base_confidence:.1f}% → {confidence:.1f}% "
                              f"({gpt_analysis['confidence_adjustment']:+.1f} pts, Risk: {gpt_analysis['risk_level']})")
                    
                except Exception as e:
                    logger.warning(f"GPT enhancement failed for {symbol}: {e}")
                    confidence = base_confidence
            else:
                confidence = base_confidence
            
            # Risk management calculations
            current_price = latest['close']
            if signal == 'BUY':
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
            elif signal == 'SELL':
                stop_loss = current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 - self.take_profit_pct)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': round(confidence, 2),
                'technical_score': round(technical_score, 2),
                'ai_score': round(ai_score, 2),
                'current_price': current_price,
                'stop_loss': round(stop_loss, 6),
                'take_profit': round(take_profit, 6),
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'entry_reasons': entry_reasons,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal analysis failed for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol, signal, confidence):
        """Calculate optimal position size based on confidence and risk management"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance < 10:  # Minimum $10 for trading
                return 0
            
            # Base position size (confidence-weighted)
            confidence_multiplier = confidence / 100
            base_allocation = self.max_position_size * confidence_multiplier
            
            # Reduce allocation if we have many active positions
            active_count = len(self.active_positions)
            if active_count > 3:
                base_allocation *= 0.5
            elif active_count > 1:
                base_allocation *= 0.75
            
            # Calculate USDT amount to trade
            position_value = usdt_balance * base_allocation
            position_value = max(10, min(position_value, usdt_balance * 0.9))  # Min $10, max 90% of balance
            
            return round(position_value, 2)
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0
    
    def execute_trade(self, signal_data):
        """Execute autonomous trade based on signal"""
        try:
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Skip if signal is HOLD or confidence too low
            if signal == 'HOLD' or confidence < self.min_confidence:
                return False
            
            # Skip if we already have a position in this symbol
            if symbol in self.active_positions:
                logger.info(f"Skipping {symbol} - already have active position")
                return False
            
            # Calculate position size
            position_value = self.calculate_position_size(symbol, signal, confidence)
            if position_value < 10:
                logger.info(f"Skipping {symbol} - position size too small: ${position_value}")
                return False
            
            # Get current market price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate trade amount
            if signal == 'BUY':
                amount = position_value / current_price
                side = 'buy'
            else:
                # For SELL signals, we need to have the base currency
                balance = self.exchange.fetch_balance()
                base_currency = symbol.split('/')[0]
                available = balance.get(base_currency, {}).get('free', 0)
                
                if available < 0.001:  # Minimum amount check
                    logger.info(f"Skipping {symbol} SELL - insufficient {base_currency} balance")
                    return False
                
                amount = min(available * 0.9, position_value / current_price)  # Use 90% of available
                side = 'sell'
            
            # Round amount to exchange precision
            amount = float(Decimal(str(amount)).quantize(Decimal('0.000001'), rounding=ROUND_DOWN))
            
            if amount < 0.001:  # Minimum trade amount
                logger.info(f"Skipping {symbol} - trade amount too small: {amount}")
                return False
            
            # Execute market order
            order = self.exchange.create_market_order(symbol, side, amount)
            
            if order and order.get('id'):
                # Record the trade
                trade_record = {
                    'symbol': symbol,
                    'side': signal,
                    'amount': amount,
                    'price': current_price,
                    'confidence': confidence,
                    'ai_score': signal_data['ai_score'],
                    'technical_score': signal_data['technical_score'],
                    'stop_loss': signal_data['stop_loss'],
                    'take_profit': signal_data['take_profit'],
                    'order_id': order['id'],
                    'timestamp': datetime.now().isoformat(),
                    'entry_reasons': signal_data.get('entry_reasons', [])
                }
                
                # Save to database
                self.save_trade_record(trade_record)
                
                # Add to active positions
                self.active_positions[symbol] = trade_record
                
                # Save active position to database
                self.save_active_position(trade_record)
                
                logger.info(f"🚀 EXECUTED {signal} {symbol}: {amount:.6f} @ ${current_price:.6f} "
                          f"(Confidence: {confidence}%, Reasons: {', '.join(signal_data.get('entry_reasons', []))})")
                
                return True
            
        except Exception as e:
            logger.error(f"Trade execution failed for {signal_data['symbol']}: {e}")
            return False
    
    def save_trade_record(self, trade_record):
        """Save trade record to database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO autonomous_trades (
                        symbol, side, amount, price, confidence, ai_score, technical_score,
                        stop_loss, take_profit, order_id, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_record['symbol'], trade_record['side'], trade_record['amount'],
                    trade_record['price'], trade_record['confidence'], trade_record['ai_score'],
                    trade_record['technical_score'], trade_record['stop_loss'], trade_record['take_profit'],
                    trade_record['order_id'], trade_record['timestamp']
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Trade record save failed: {e}")
    
    def save_active_position(self, position_data):
        """Save active position to database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO active_positions (
                        symbol, side, amount, entry_price, stop_loss, take_profit,
                        confidence, entry_timestamp, last_update
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position_data['symbol'], position_data['side'], position_data['amount'],
                    position_data['price'], position_data['stop_loss'], position_data['take_profit'],
                    position_data['confidence'], position_data['timestamp'], datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Active position save failed: {e}")
    
    def monitor_positions(self):
        """Monitor active positions for stop loss and take profit"""
        try:
            for symbol in list(self.active_positions.keys()):
                position = self.active_positions[symbol]
                
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                entry_price = position['price']
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']
                side = position['side']
                
                # Calculate P&L
                if side == 'BUY':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    # Check exit conditions
                    should_exit = current_price <= stop_loss or current_price >= take_profit
                else:  # SELL
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    # Check exit conditions
                    should_exit = current_price >= stop_loss or current_price <= take_profit
                
                # Update position
                position['current_price'] = current_price
                position['pnl'] = pnl_pct
                
                if should_exit:
                    self.close_position(symbol, current_price, pnl_pct)
                else:
                    # Update position in database
                    self.update_position_status(symbol, current_price, pnl_pct)
                
        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")
    
    def close_position(self, symbol, exit_price, pnl_pct):
        """Close an active position"""
        try:
            position = self.active_positions.get(symbol)
            if not position:
                return
            
            side = 'sell' if position['side'] == 'BUY' else 'buy'
            amount = position['amount']
            
            # Execute closing order
            order = self.exchange.create_market_order(symbol, side, amount)
            
            if order and order.get('id'):
                # Update trade record
                self.update_trade_exit(position['order_id'], exit_price, pnl_pct)
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                # Remove from database
                self.remove_active_position(symbol)
                
                logger.info(f"🎯 CLOSED {symbol}: Exit @ ${exit_price:.6f} "
                          f"(P&L: {pnl_pct:+.2f}%)")
                
        except Exception as e:
            logger.error(f"Position close failed for {symbol}: {e}")
    
    def update_trade_exit(self, order_id, exit_price, pnl):
        """Update trade record with exit data"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE autonomous_trades 
                    SET exit_price = ?, exit_timestamp = ?, pnl = ?, status = 'CLOSED'
                    WHERE order_id = ?
                ''', (exit_price, datetime.now().isoformat(), pnl, order_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Trade exit update failed: {e}")
    
    def update_position_status(self, symbol, current_price, pnl):
        """Update active position status"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE active_positions 
                    SET current_price = ?, pnl = ?, last_update = ?
                    WHERE symbol = ?
                ''', (current_price, pnl, datetime.now().isoformat(), symbol))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Position status update failed: {e}")
    
    def remove_active_position(self, symbol):
        """Remove active position from database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM active_positions WHERE symbol = ?', (symbol,))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Position removal failed: {e}")
    
    def scan_and_trade(self):
        """Main scanning and trading loop"""
        logger.info("🔄 Starting market scan for autonomous trading opportunities...")
        
        signals_found = 0
        trades_executed = 0
        
        for symbol in self.trading_pairs:
            try:
                # Analyze signal
                signal_data = self.analyze_trading_signal(symbol)
                
                if signal_data and signal_data['signal'] != 'HOLD':
                    signals_found += 1
                    logger.info(f"📊 {symbol}: {signal_data['signal']} "
                              f"(Confidence: {signal_data['confidence']}%, "
                              f"AI: {signal_data['ai_score']}%, "
                              f"Technical: {signal_data['technical_score']}%)")
                    
                    # Execute trade if confidence meets threshold
                    if signal_data['confidence'] >= self.min_confidence:
                        if self.execute_trade(signal_data):
                            trades_executed += 1
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Scan failed for {symbol}: {e}")
                continue
        
        logger.info(f"✅ Scan complete: {signals_found} signals found, {trades_executed} trades executed")
    
    def run_autonomous_trading(self):
        """Run continuous autonomous trading"""
        logger.info("🚀 Starting Autonomous Trading Engine")
        logger.info(f"Configuration: Min Confidence: {self.min_confidence}%, "
                   f"Max Position: {self.max_position_size*100}%, "
                   f"Stop Loss: {self.stop_loss_pct*100}%, "
                   f"Take Profit: {self.take_profit_pct*100}%")
        
        self.running = True
        
        while self.running:
            try:
                # Monitor existing positions
                if self.active_positions:
                    logger.info(f"📈 Monitoring {len(self.active_positions)} active positions...")
                    self.monitor_positions()
                
                # Scan for new opportunities
                self.scan_and_trade()
                
                # Wait for next scan
                logger.info(f"⏰ Next scan in {self.scan_interval} seconds...")
                time.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Autonomous trading stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Autonomous trading error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def stop_trading(self):
        """Stop autonomous trading"""
        self.running = False
        logger.info("🛑 Autonomous trading engine stopped")

def main():
    """Main autonomous trading function"""
    try:
        engine = AutonomousTradingEngine()
        
        # Initialize exchange
        if not engine.initialize_exchange():
            logger.error("Failed to initialize exchange")
            return
        
        # Setup database
        engine.setup_database()
        
        # Initialize GPT enhancement
        try:
            engine.gpt_analyzer = GPTEnhancedTradingAnalyzer()
            logger.info("🧠 GPT-enhanced analysis enabled")
        except Exception as e:
            logger.warning(f"GPT enhancement disabled: {e}")
            engine.gpt_enhancement_enabled = False
        
        # Start autonomous trading
        engine.run_autonomous_trading()
        
    except Exception as e:
        logger.error(f"Autonomous trading engine failed: {e}")

if __name__ == "__main__":
    main()