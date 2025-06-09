#!/usr/bin/env python3
"""
Live Trading Engine - Autonomous AI-Powered Trading
Executes real trades on OKX with risk management and portfolio protection
"""

import os
import time
import sqlite3
import logging
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from threading import Thread, Event
import signal
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTradingEngine:
    """Autonomous live trading engine with AI signal integration"""
    
    def __init__(self):
        self.db_path = 'live_trading.db'
        self.exchange = None
        self.is_running = False
        self.stop_event = Event()
        self.last_signal_time = None
        
        # Trading parameters
        self.max_position_size_pct = 0.01  # 1% per trade
        self.min_trade_amount = 10  # Minimum $10 USDT
        self.trading_pairs = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT']
        self.signal_cooldown = 60  # 60 seconds between signals
        
        # Risk management
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.max_open_positions = 3
        
        self.initialize_exchange()
        self.initialize_database()
        
    def initialize_exchange(self):
        """Initialize OKX exchange for live trading"""
        try:
            api_key = os.getenv('OKX_API_KEY')
            secret_key = os.getenv('OKX_SECRET_KEY')
            passphrase = os.getenv('OKX_PASSPHRASE')
            
            if not all([api_key, secret_key, passphrase]):
                raise Exception("OKX API credentials required for live trading")
            
            self.exchange = ccxt.okx({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase,
                'sandbox': False,  # LIVE TRADING MODE
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # Spot trading
                }
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            logger.info(f"Live trading engine connected to OKX")
            logger.info(f"Available USDT balance: {balance['USDT']['free']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OKX exchange: {e}")
            raise
    
    def initialize_database(self):
        """Initialize live trading database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Live trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    order_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    ai_confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    entry_time TEXT,
                    exit_time TEXT,
                    pnl REAL DEFAULT 0,
                    fees REAL DEFAULT 0
                )
            ''')
            
            # Active positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    strategy TEXT NOT NULL
                )
            ''')
            
            # Trading signals log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    executed BOOLEAN DEFAULT FALSE,
                    ai_model TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Live trading database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def get_portfolio_balance(self) -> Dict:
        """Get current portfolio balance"""
        try:
            balance = self.exchange.fetch_balance()
            total_value = 0
            
            for currency, amount in balance['total'].items():
                if amount > 0:
                    if currency == 'USDT':
                        total_value += amount
                    else:
                        try:
                            ticker = self.exchange.fetch_ticker(f"{currency}/USDT")
                            total_value += amount * ticker['last']
                        except:
                            pass
            
            return {
                'total_value': total_value,
                'usdt_balance': balance['USDT']['free'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio balance: {e}")
            return {'total_value': 0, 'usdt_balance': 0}
    
    def generate_ai_signal(self, symbol: str) -> Dict:
        """Generate AI trading signal for symbol"""
        try:
            # Get market data
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            if len(ohlcv) < 50:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['rsi'] = self.calculate_rsi(df['close'])
            df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # AI signal logic (simplified but effective)
            signals = []
            confidence = 0
            
            # Enhanced trend following signals with more sensitivity
            price_above_sma20 = current['close'] > current['sma_20']
            sma20_above_sma50 = current['sma_20'] > current['sma_50']
            
            if price_above_sma20 and sma20_above_sma50:
                trend_strength = (current['close'] - current['sma_50']) / current['sma_50']
                signals.append(('BUY', min(0.5 + trend_strength * 2, 0.8), 'Strong bullish trend'))
            elif not price_above_sma20 and not sma20_above_sma50:
                trend_strength = (current['sma_50'] - current['close']) / current['sma_50']
                signals.append(('SELL', min(0.5 + trend_strength * 2, 0.8), 'Strong bearish trend'))
            
            # Enhanced RSI signals with more aggressive thresholds
            if current['rsi'] < 35:  # Lowered from 30
                rsi_strength = (35 - current['rsi']) / 35
                signals.append(('BUY', min(0.6 + rsi_strength, 0.85), 'RSI oversold opportunity'))
            elif current['rsi'] > 65:  # Lowered from 70
                rsi_strength = (current['rsi'] - 65) / 35
                signals.append(('SELL', min(0.6 + rsi_strength, 0.85), 'RSI overbought opportunity'))
            
            # Enhanced MACD signals
            if current['macd'] > current['macd_signal']:
                macd_strength = abs(current['macd'] - current['macd_signal']) / abs(current['macd_signal']) if current['macd_signal'] != 0 else 0.3
                signals.append(('BUY', min(0.4 + macd_strength, 0.8), 'MACD bullish momentum'))
            elif current['macd'] < current['macd_signal']:
                macd_strength = abs(current['macd_signal'] - current['macd']) / abs(current['macd_signal']) if current['macd_signal'] != 0 else 0.3
                signals.append(('SELL', min(0.4 + macd_strength, 0.8), 'MACD bearish momentum'))
            
            # Price momentum signals
            price_change_1h = (current['close'] - prev['close']) / prev['close']
            if abs(price_change_1h) > 0.01:  # 1% price movement
                momentum_strength = min(abs(price_change_1h) * 20, 0.4)
                if price_change_1h > 0:
                    signals.append(('BUY', 0.3 + momentum_strength, 'Strong upward momentum'))
                else:
                    signals.append(('SELL', 0.3 + momentum_strength, 'Strong downward momentum'))
            
            # Volume confirmation
            volume_sma = df['volume'].rolling(20).mean().iloc[-1]
            if current['volume'] > volume_sma * 1.5:
                # Increase confidence for volume confirmation
                signals = [(side, conf * 1.2, reason) for side, conf, reason in signals]
            
            # Aggregate signals
            buy_signals = [s for s in signals if s[0] == 'BUY']
            sell_signals = [s for s in signals if s[0] == 'SELL']
            
            if buy_signals:
                total_confidence = sum(s[1] for s in buy_signals)
                reasons = [s[2] for s in buy_signals]
                if total_confidence >= 0.45:  # Lowered from 60% to 45% for more aggressive trading
                    return {
                        'symbol': symbol,
                        'signal': 'BUY',
                        'confidence': min(total_confidence, 0.95),
                        'price': current['close'],
                        'reasons': reasons,
                        'model': 'AI_Technical_Ensemble'
                    }
            
            elif sell_signals:
                total_confidence = sum(s[1] for s in sell_signals)
                reasons = [s[2] for s in sell_signals]
                if total_confidence >= 0.45:  # Lowered from 60% to 45% for more aggressive trading
                    return {
                        'symbol': symbol,
                        'signal': 'SELL',
                        'confidence': min(total_confidence, 0.95),
                        'price': current['close'],
                        'reasons': reasons,
                        'model': 'AI_Technical_Ensemble'
                    }
            
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0.5,
                'price': current['close'],
                'reasons': ['Insufficient signal strength'],
                'model': 'AI_Technical_Ensemble'
            }
            
        except Exception as e:
            logger.error(f"Error generating AI signal for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate position size based on portfolio value and confidence"""
        try:
            portfolio = self.get_portfolio_balance()
            max_amount = portfolio['total_value'] * self.max_position_size_pct
            
            # Adjust by confidence (50% confidence = 50% of max position)
            confidence_adjusted = max_amount * confidence
            
            # Ensure minimum trade amount
            if confidence_adjusted < self.min_trade_amount:
                return 0
            
            return min(confidence_adjusted, portfolio['usdt_balance'] * 0.9)  # Keep 10% cash
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def execute_trade(self, signal: Dict) -> Optional[Dict]:
        """Execute live trade based on AI signal"""
        try:
            symbol = signal['symbol']
            side = signal['signal'].lower()
            confidence = signal['confidence']
            
            if side == 'hold':
                return None
            
            # Check for existing position
            if self.has_active_position(symbol):
                logger.info(f"Skipping {symbol} - position already exists")
                return None
            
            # Check maximum open positions
            if self.get_active_position_count() >= self.max_open_positions:
                logger.info(f"Skipping {symbol} - maximum positions reached")
                return None
            
            # Calculate position size
            position_value = self.calculate_position_size(symbol, confidence)
            if position_value < self.min_trade_amount:
                logger.info(f"Skipping {symbol} - position size too small: ${position_value}")
                return None
            
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate amount to trade
            if side == 'buy':
                amount = position_value / current_price
                amount = self.exchange.amount_to_precision(symbol, amount)
                
                if amount * current_price < self.min_trade_amount:
                    logger.info(f"Skipping {symbol} - amount too small after precision: {amount}")
                    return None
                
                # Execute market buy order
                order = self.exchange.create_market_buy_order(symbol, amount)
                
            elif side == 'sell':
                # Check if we have the asset to sell
                balance = self.exchange.fetch_balance()
                base_currency = symbol.split('/')[0]
                available = balance[base_currency]['free']
                
                if available < amount:
                    logger.info(f"Skipping {symbol} sell - insufficient balance: {available}")
                    return None
                
                # Execute market sell order
                order = self.exchange.create_market_sell_order(symbol, amount)
            
            # Log the trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side.upper(),
                'amount': amount,
                'price': current_price,
                'order_id': order['id'],
                'strategy': 'AI_Live_Trading',
                'ai_confidence': confidence,
                'status': 'EXECUTED'
            }
            
            self.log_trade(trade_record)
            
            # Add to active positions if buy
            if side == 'buy':
                self.add_active_position({
                    'symbol': symbol,
                    'side': 'LONG',
                    'amount': amount,
                    'entry_price': current_price,
                    'entry_time': datetime.now().isoformat(),
                    'stop_loss': current_price * (1 - self.stop_loss_pct),
                    'take_profit': current_price * (1 + self.take_profit_pct),
                    'strategy': 'AI_Live_Trading'
                })
            
            logger.info(f"LIVE TRADE EXECUTED: {side.upper()} {amount} {symbol} at ${current_price}")
            logger.info(f"Order ID: {order['id']}, Confidence: {confidence:.1%}")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Trade execution failed for {signal['symbol']}: {e}")
            return None
    
    def log_trade(self, trade: Dict):
        """Log trade to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO live_trades 
                (timestamp, symbol, side, amount, price, order_id, strategy, ai_confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['timestamp'], trade['symbol'], trade['side'],
                trade['amount'], trade['price'], trade['order_id'],
                trade['strategy'], trade['ai_confidence'], trade['status']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def add_active_position(self, position: Dict):
        """Add position to active positions tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO active_positions 
                (symbol, side, amount, entry_price, entry_time, stop_loss, take_profit, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position['symbol'], position['side'], position['amount'],
                position['entry_price'], position['entry_time'],
                position['stop_loss'], position['take_profit'], position['strategy']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error adding active position: {e}")
    
    def has_active_position(self, symbol: str) -> bool:
        """Check if there's an active position for symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM active_positions WHERE symbol = ?', (symbol,))
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking active position: {e}")
            return False
    
    def get_active_position_count(self) -> int:
        """Get number of active positions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM active_positions')
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting position count: {e}")
            return 0
    
    def check_position_exits(self):
        """Check active positions for stop loss or take profit triggers"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM active_positions')
            positions = cursor.fetchall()
            
            for pos in positions:
                symbol = pos[1]
                entry_price = pos[4]
                stop_loss = pos[6]
                take_profit = pos[7]
                
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    should_exit = False
                    exit_reason = ""
                    
                    if current_price <= stop_loss:
                        should_exit = True
                        exit_reason = "STOP_LOSS"
                    elif current_price >= take_profit:
                        should_exit = True
                        exit_reason = "TAKE_PROFIT"
                    
                    if should_exit:
                        # Execute exit trade
                        balance = self.exchange.fetch_balance()
                        base_currency = symbol.split('/')[0]
                        amount = balance[base_currency]['free']
                        
                        if amount > 0:
                            order = self.exchange.create_market_sell_order(symbol, amount)
                            
                            # Calculate PnL
                            pnl = (current_price - entry_price) / entry_price * 100
                            
                            # Log exit trade
                            exit_trade = {
                                'timestamp': datetime.now().isoformat(),
                                'symbol': symbol,
                                'side': 'SELL',
                                'amount': amount,
                                'price': current_price,
                                'order_id': order['id'],
                                'strategy': f'EXIT_{exit_reason}',
                                'ai_confidence': 1.0,
                                'status': 'EXECUTED'
                            }
                            
                            self.log_trade(exit_trade)
                            
                            # Remove from active positions
                            cursor.execute('DELETE FROM active_positions WHERE symbol = ?', (symbol,))
                            
                            logger.info(f"POSITION CLOSED: {symbol} at ${current_price} ({exit_reason}) PnL: {pnl:.2f}%")
                
                except Exception as e:
                    logger.error(f"Error checking position {symbol}: {e}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error checking position exits: {e}")
    
    def trading_loop(self):
        """Main autonomous trading loop"""
        logger.info("🚀 STARTING LIVE AUTONOMOUS TRADING")
        logger.info("=" * 50)
        logger.info("Trading Mode: LIVE")
        logger.info("Risk Limit: 1% per trade")
        logger.info("AI Autonomy: ENABLED")
        logger.info("=" * 50)
        
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Rate limiting - check signals every minute
                if (self.last_signal_time and 
                    (current_time - self.last_signal_time).seconds < self.signal_cooldown):
                    time.sleep(10)
                    continue
                
                # Check position exits first
                self.check_position_exits()
                
                # Generate and evaluate signals for each trading pair
                for symbol in self.trading_pairs:
                    try:
                        signal = self.generate_ai_signal(symbol)
                        
                        if signal and signal['signal'] != 'HOLD':
                            # Log signal
                            self.log_signal(signal)
                            
                            # Execute trade if signal is strong enough
                            if signal['confidence'] >= 0.6:  # Lowered to 60% minimum confidence for execution
                                trade_result = self.execute_trade(signal)
                                
                                if trade_result:
                                    # Add delay after successful trade
                                    time.sleep(30)
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                self.last_signal_time = current_time
                
                # Portfolio status update
                portfolio = self.get_portfolio_balance()
                active_positions = self.get_active_position_count()
                
                logger.info(f"Portfolio: ${portfolio['total_value']:.2f}, "
                           f"USDT: ${portfolio['usdt_balance']:.2f}, "
                           f"Positions: {active_positions}")
                
                # Sleep before next iteration
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping trading...")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(30)  # Wait before retry
    
    def log_signal(self, signal: Dict):
        """Log AI signal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trading_signals 
                (timestamp, symbol, signal, confidence, price, ai_model)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(), signal['symbol'], signal['signal'],
                signal['confidence'], signal['price'], signal['model']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging signal: {e}")
    
    def start(self):
        """Start the live trading engine"""
        if self.is_running:
            logger.warning("Trading engine already running")
            return
        
        self.is_running = True
        self.trading_thread = Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()
        
        logger.info("Live trading engine started")
    
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        self.stop_event.set()
        
        logger.info("Live trading engine stopped")
    
    def get_trading_status(self) -> Dict:
        """Get current trading status"""
        portfolio = self.get_portfolio_balance()
        active_positions = self.get_active_position_count()
        
        return {
            'system_status': 'LIVE_TRADING',
            'portfolio_mode': 'REAL',
            'is_running': self.is_running,
            'portfolio_value': portfolio['total_value'],
            'usdt_balance': portfolio['usdt_balance'],
            'active_positions': active_positions,
            'risk_limit': '1% per trade',
            'ai_autonomy': True,
            'timestamp': datetime.now().isoformat()
        }

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    if 'engine' in globals():
        engine.stop()
    sys.exit(0)

def main():
    """Main function to start live trading"""
    global engine
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        engine = LiveTradingEngine()
        engine.start()
        
        # Keep main thread alive
        while engine.is_running:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Failed to start live trading engine: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()