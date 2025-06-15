"""
Live Under $50 Futures Trading Engine
Real money trading with OKX futures for tokens under $50
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

class LiveUnder50FuturesEngine:
    def __init__(self):
        self.exchange = None
        self.db_path = 'live_under50_futures.db'
        self.min_confidence = 75.0  # Higher threshold for live trading
        self.max_leverage = 3  # Conservative leverage for live trading
        self.max_position_size = 0.03  # 3% max position for live trading
        self.price_threshold = 50.0
        self.min_usdt_balance = 50  # Minimum USDT balance required
        
        # Active symbols under $50
        self.active_symbols = []
        
    def initialize_exchange(self):
        """Initialize OKX futures connection for live trading"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'},
                'rateLimit': 500
            })
            
            # Test connection and get balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance < self.min_usdt_balance:
                logger.error(f"Insufficient USDT balance: ${usdt_balance:.2f} (minimum: ${self.min_usdt_balance})")
                return False
            
            logger.info(f"Live trading engine connected - USDT Balance: ${usdt_balance:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"OKX connection failed: {e}")
            return False
    
    def setup_database(self):
        """Setup live trading database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_futures_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    leverage REAL NOT NULL,
                    confidence REAL NOT NULL,
                    price_tier TEXT NOT NULL,
                    order_id TEXT,
                    stop_loss_id TEXT,
                    take_profit_id TEXT,
                    status TEXT DEFAULT 'OPEN',
                    pnl_usd REAL DEFAULT 0,
                    pnl_percentage REAL DEFAULT 0,
                    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    exit_time TIMESTAMP,
                    exit_price REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_trading_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    symbol TEXT,
                    details TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Live trading database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_available_balance(self) -> float:
        """Get available USDT balance for trading"""
        try:
            balance = self.exchange.fetch_balance()
            return balance.get('USDT', {}).get('free', 0)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0
    
    def filter_symbols_by_price(self) -> List[str]:
        """Filter and return symbols under $50"""
        candidate_symbols = [
            'ADA/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'DOT/USDT:USDT',
            'AVAX/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'NEAR/USDT:USDT',
            'TRX/USDT:USDT', 'ICP/USDT:USDT', 'ALGO/USDT:USDT', 'HBAR/USDT:USDT',
            'XLM/USDT:USDT', 'SAND/USDT:USDT', 'MANA/USDT:USDT', 'THETA/USDT:USDT',
            'AXS/USDT:USDT', 'FIL/USDT:USDT', 'ETC/USDT:USDT', 'EGLD/USDT:USDT',
            'FLOW/USDT:USDT', 'ENJ/USDT:USDT', 'CHZ/USDT:USDT', 'CRV/USDT:USDT',
            'LINK/USDT:USDT', 'GRT/USDT:USDT', 'SUSHI/USDT:USDT', 'SNX/USDT:USDT',
            'NEO/USDT:USDT', 'IOTA/USDT:USDT', 'ZIL/USDT:USDT', 'ONT/USDT:USDT',
            'BAT/USDT:USDT', 'QTUM/USDT:USDT'
        ]
        
        filtered_symbols = []
        for symbol in candidate_symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker['last']
                
                if price and price < self.price_threshold:
                    filtered_symbols.append(symbol)
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Failed to check price for {symbol}: {e}")
                continue
        
        self.active_symbols = filtered_symbols
        logger.info(f"Active symbols under $50: {len(self.active_symbols)} tokens")
        return filtered_symbols
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Technical indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd_data = ta.macd(df['close'])
            df['macd'] = macd_data['MACD_12_26_9']
            df['macd_signal'] = macd_data['MACDs_12_26_9']
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df.fillna(0)
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def generate_live_signal(self, symbol: str) -> Optional[Dict]:
        """Generate live trading signal with higher confidence threshold"""
        try:
            df = self.get_market_data(symbol)
            if df is None or len(df) < 20:
                return None
            
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Skip if price is above threshold
            if current_price >= self.price_threshold:
                return None
            
            # Initialize scoring (higher threshold for live trading)
            bullish_score = 50
            bearish_score = 50
            
            # RSI Analysis
            rsi = latest['rsi']
            if rsi < 30:
                bullish_score += 20
            elif rsi < 40:
                bullish_score += 10
            elif rsi > 70:
                bearish_score += 20
            elif rsi > 60:
                bearish_score += 10
            
            # MACD Analysis
            if latest['macd'] > latest['macd_signal']:
                bullish_score += 15
            else:
                bearish_score += 15
            
            # Bollinger Bands
            bb_pos = latest['bb_position']
            if bb_pos < 0.2:
                bullish_score += 15
            elif bb_pos > 0.8:
                bearish_score += 15
            
            # Volume confirmation
            if latest['volume_ratio'] > 1.3:
                if latest['close'] > df.iloc[-2]['close']:
                    bullish_score += 10
                else:
                    bearish_score += 10
            
            # Determine signal
            confidence_diff = abs(bullish_score - bearish_score)
            
            if bullish_score > bearish_score and confidence_diff >= 15:
                signal = 'LONG'
                confidence = min(95, 50 + confidence_diff * 1.5)
            elif bearish_score > bullish_score and confidence_diff >= 15:
                signal = 'SHORT'
                confidence = min(95, 50 + confidence_diff * 1.5)
            else:
                return None
            
            # Only proceed if confidence meets live trading threshold
            if confidence < self.min_confidence:
                return None
            
            # Price tier classification
            if current_price < 0.1:
                price_tier = "PENNY"
            elif current_price < 1.0:
                price_tier = "CENT"
            elif current_price < 10.0:
                price_tier = "SINGLE"
            else:
                price_tier = "DOUBLE"
            
            # Conservative targets for live trading
            volatility = df['close'].rolling(20).std().iloc[-1] / current_price
            
            if signal == 'LONG':
                stop_loss = current_price * (1 - max(0.03, volatility * 0.5))
                take_profit = current_price * (1 + max(0.06, volatility))
            else:
                stop_loss = current_price * (1 + max(0.03, volatility * 0.5))
                take_profit = current_price * (1 - max(0.06, volatility))
            
            # Conservative leverage for live trading
            if confidence > 85:
                leverage = 3
            elif confidence > 80:
                leverage = 2
            else:
                leverage = 1
            
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
                'volatility': round(volatility, 4),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None
    
    def execute_live_trade(self, signal: Dict) -> bool:
        """Execute actual live trade on OKX"""
        try:
            # Get current balance
            balance = self.get_available_balance()
            if balance < self.min_usdt_balance:
                logger.error(f"Insufficient balance for trade: ${balance:.2f}")
                return False
            
            # Calculate position size (conservative for live trading)
            max_position_value = balance * self.max_position_size
            
            # Further reduce position size for margin requirements
            # Use only 1% of balance for micro positions
            conservative_position_value = min(max_position_value, balance * 0.01)
            position_size = conservative_position_value / signal['price']
            
            # Ensure minimum position size requirements
            if position_size < 1:
                position_size = 1
            
            # Set leverage
            try:
                self.exchange.set_leverage(signal['leverage'], signal['symbol'])
                logger.info(f"Set leverage {signal['leverage']}x for {signal['symbol']}")
            except Exception as e:
                logger.warning(f"Failed to set leverage: {e}")
            
            # Place market order
            side = 'buy' if signal['signal'] == 'LONG' else 'sell'
            
            order = self.exchange.create_market_order(
                symbol=signal['symbol'],
                side=side,
                amount=position_size
            )
            
            order_id = order['id']
            entry_price = order['average'] or signal['price']
            
            logger.info(f"🚀 LIVE TRADE EXECUTED: {signal['symbol']} {signal['signal']} "
                       f"Size: {position_size:.4f} Price: ${entry_price:.6f} "
                       f"Leverage: {signal['leverage']}x (Order ID: {order_id})")
            
            # Set stop loss and take profit
            self.set_stop_loss_take_profit(signal, order_id, entry_price)
            
            # Save to database
            self.save_live_position(signal, order_id, entry_price, position_size)
            
            # Log action
            self.log_action("TRADE_EXECUTED", signal['symbol'], 
                          f"{signal['signal']} {position_size:.4f} @ ${entry_price:.6f}", True)
            
            return True
            
        except Exception as e:
            logger.error(f"Live trade execution failed: {e}")
            self.log_action("TRADE_FAILED", signal.get('symbol', 'UNKNOWN'), str(e), False, str(e))
            return False
    
    def set_stop_loss_take_profit(self, signal: Dict, order_id: str, entry_price: float):
        """Set stop loss and take profit orders"""
        try:
            symbol = signal['symbol']
            position_size = self.get_position_size(symbol)
            
            if signal['signal'] == 'LONG':
                # Stop loss (sell)
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side='sell',
                    amount=position_size,
                    price=signal['stop_loss']
                )
                
                # Take profit (sell)
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side='sell',
                    amount=position_size,
                    price=signal['take_profit']
                )
            else:
                # Stop loss (buy)
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side='buy',
                    amount=position_size,
                    price=signal['stop_loss']
                )
                
                # Take profit (buy)
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side='buy',
                    amount=position_size,
                    price=signal['take_profit']
                )
            
            logger.info(f"Set SL: {signal['stop_loss']:.6f} TP: {signal['take_profit']:.6f} for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to set SL/TP for {symbol}: {e}")
    
    def get_position_size(self, symbol: str) -> float:
        """Get current position size for symbol"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            for position in positions:
                if position['symbol'] == symbol and position['size'] > 0:
                    return position['size']
            return 0
        except Exception as e:
            logger.error(f"Failed to get position size for {symbol}: {e}")
            return 0
    
    def save_live_position(self, signal: Dict, order_id: str, entry_price: float, position_size: float):
        """Save live position to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO live_futures_positions 
                (symbol, side, size, entry_price, leverage, confidence, price_tier, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'], signal['signal'], position_size, entry_price,
                signal['leverage'], signal['confidence'], signal['price_tier'], order_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
    
    def log_action(self, action: str, symbol: str, details: str, success: bool, error_message: str = None):
        """Log trading action to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO live_trading_log (action, symbol, details, success, error_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (action, symbol, details, success, error_message))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log action: {e}")
    
    def scan_and_trade(self):
        """Scan for signals and execute live trades"""
        try:
            logger.info("🔄 Starting live under $50 futures scan...")
            
            # Refresh symbols list periodically
            self.filter_symbols_by_price()
            
            trades_executed = 0
            signals_found = 0
            
            for symbol in self.active_symbols:
                try:
                    signal = self.generate_live_signal(symbol)
                    if signal:
                        signals_found += 1
                        if self.execute_live_trade(signal):
                            trades_executed += 1
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            logger.info(f"✅ Live scan complete: {signals_found} signals, {trades_executed} live trades executed")
            
        except Exception as e:
            logger.error(f"Live trading cycle failed: {e}")

def main():
    """Main live trading function"""
    try:
        engine = LiveUnder50FuturesEngine()
        
        if not engine.initialize_exchange():
            logger.error("Failed to initialize exchange")
            return
        
        engine.setup_database()
        
        logger.info("🚀 Starting LIVE Under $50 Futures Trading Engine")
        logger.info(f"LIVE TRADING ENABLED - Min Confidence: {engine.min_confidence}%")
        logger.info(f"Max Position Size: {engine.max_position_size*100}%, Max Leverage: {engine.max_leverage}x")
        
        while True:
            engine.scan_and_trade()
            logger.info("⏰ Next live scan in 600 seconds (10 minutes)...")
            time.sleep(600)  # 10 minutes for live trading
            
    except KeyboardInterrupt:
        logger.info("Live trading engine stopped")
    except Exception as e:
        logger.error(f"Live trading engine failed: {e}")

if __name__ == "__main__":
    main()