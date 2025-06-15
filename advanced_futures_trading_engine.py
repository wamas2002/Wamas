"""
Advanced Futures Trading Engine
Implements long and short position trading with sophisticated risk management
"""

import os
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFuturesEngine:
    def __init__(self):
        self.exchange = None
        self.db_path = 'futures_trading.db'
        self.min_confidence = 65.0  # Lowered threshold for more signals
        self.max_leverage = 3  # Conservative leverage
        self.max_position_size = 0.10  # 10% max position size
        self.stop_loss_pct = 0.06  # 6% stop loss
        self.take_profit_pct = 0.12  # 12% take profit
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT', 'LINK/USDT:USDT', 'LTC/USDT:USDT', 'DOT/USDT:USDT', 'AVAX/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'NEAR/USDT:USDT', 'TRX/USDT:USDT', 'ICP/USDT:USDT', 'ALGO/USDT:USDT', 'HBAR/USDT:USDT', 'XLM/USDT:USDT', 'SAND/USDT:USDT', 'MANA/USDT:USDT', 'THETA/USDT:USDT', 'AXS/USDT:USDT', 'FIL/USDT:USDT', 'ETC/USDT:USDT', 'EGLD/USDT:USDT', 'FLOW/USDT:USDT', 'ENJ/USDT:USDT', 'CHZ/USDT:USDT', 'CRV/USDT:USDT']
        self.models = {}
        self.scalers = {}
        
    def initialize_exchange(self):
        """Initialize OKX exchange for futures trading"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap'  # Enable futures trading
                }
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            logger.info("Advanced Futures Engine connected to OKX")
            return True
            
        except Exception as e:
            logger.error(f"OKX connection failed: {e}")
            return False
    
    def setup_database(self):
        """Setup futures trading database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Futures positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS futures_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    leverage REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    confidence REAL NOT NULL,
                    pnl_usd REAL DEFAULT 0,
                    status TEXT DEFAULT 'OPEN',
                    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    entry_reasons TEXT,
                    market_conditions TEXT
                )
            ''')
            
            # Futures signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS futures_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    technical_score REAL,
                    ai_score REAL,
                    momentum_score REAL,
                    volatility_score REAL,
                    current_price REAL,
                    rsi REAL,
                    macd REAL,
                    bb_position REAL,
                    volume_ratio REAL,
                    trend_strength REAL,
                    support_resistance TEXT,
                    recommended_leverage REAL,
                    risk_level TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Risk management table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS futures_risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_exposure REAL,
                    used_margin REAL,
                    free_margin REAL,
                    margin_ratio REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    daily_pnl REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Futures trading database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """Get comprehensive market data for futures analysis"""
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate comprehensive technical indicators
            df = self.calculate_advanced_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators for futures trading"""
        try:
            # Basic indicators
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['ema_12'] = ta.ema(df['close'], length=12)
            df['ema_26'] = ta.ema(df['close'], length=26)
            
            # Momentum indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                df['macd'] = macd.iloc[:, 0] if len(macd.columns) > 0 else 0
                df['macd_signal'] = macd.iloc[:, 1] if len(macd.columns) > 1 else 0
                df['macd_histogram'] = macd.iloc[:, 2] if len(macd.columns) > 2 else 0
            
            # Volatility indicators
            bb = ta.bbands(df['close'], length=20)
            if bb is not None and not bb.empty:
                df['bb_upper'] = bb.iloc[:, 0] if len(bb.columns) > 0 else df['close']
                df['bb_middle'] = bb.iloc[:, 1] if len(bb.columns) > 1 else df['close']
                df['bb_lower'] = bb.iloc[:, 2] if len(bb.columns) > 2 else df['close']
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Trend indicators
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Support and Resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            # Trend strength
            df['trend_strength'] = abs(df['close'] - df['sma_20']) / df['atr']
            
            # Price momentum
            df['momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
            
            # Futures-specific indicators
            df['price_change_1h'] = df['close'].pct_change(1) * 100
            df['price_change_4h'] = df['close'].pct_change(4) * 100
            df['price_change_24h'] = df['close'].pct_change(24) * 100
            
            return df.fillna(0)
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            return df
    
    def train_futures_models(self, df: pd.DataFrame, symbol: str):
        """Train ML models specifically for futures trading"""
        try:
            if len(df) < 50:
                return
            
            # Prepare features for futures trading
            features = [
                'rsi', 'macd', 'macd_histogram', 'bb_position', 'bb_width',
                'atr', 'adx', 'volume_ratio', 'trend_strength', 'momentum',
                'price_change_1h', 'price_change_4h', 'price_change_24h'
            ]
            
            # Create target variable (future price movement)
            df['future_return'] = df['close'].shift(-5) / df['close'] - 1
            df['target'] = np.where(df['future_return'] > 0.02, 2,  # Strong bullish
                                  np.where(df['future_return'] > 0.005, 1,  # Bullish
                                         np.where(df['future_return'] < -0.02, -2,  # Strong bearish
                                                np.where(df['future_return'] < -0.005, -1, 0))))  # Bearish, Neutral
            
            # Prepare training data
            X = df[features].fillna(0).iloc[:-5]
            y = df['target'].fillna(0).iloc[:-5]
            
            if len(X) < 30:
                return
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train models
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
            
            rf_model.fit(X_scaled, y)
            gb_model.fit(X_scaled, y)
            
            # Store models
            self.models[symbol] = {
                'rf': rf_model,
                'gb': gb_model,
                'features': features
            }
            self.scalers[symbol] = scaler
            
            logger.info(f"Futures models trained for {symbol}")
            
        except Exception as e:
            logger.error(f"Model training failed for {symbol}: {e}")
    
    def generate_futures_signal(self, symbol: str) -> Optional[Dict]:
        """Generate sophisticated futures trading signal"""
        try:
            df = self.get_market_data(symbol)
            if df is None or len(df) < 50:
                return None
            
            # Train models if not exists
            if symbol not in self.models:
                self.train_futures_models(df, symbol)
            
            if symbol not in self.models:
                return None
            
            latest = df.iloc[-1]
            
            # Technical Analysis Score (40%) - More sensitive thresholds
            technical_signals = []
            
            # RSI signals with broader ranges
            if latest['rsi'] < 35:
                technical_signals.append(('RSI_OVERSOLD', 15))
            elif latest['rsi'] > 65:
                technical_signals.append(('RSI_OVERBOUGHT', -15))
            elif 40 <= latest['rsi'] <= 60:
                technical_signals.append(('RSI_NEUTRAL', 5))
            elif latest['rsi'] < 45:
                technical_signals.append(('RSI_BEARISH_LEAN', -8))
            elif latest['rsi'] > 55:
                technical_signals.append(('RSI_BULLISH_LEAN', 8))
            
            # MACD signals - more sensitive
            if latest['macd'] > latest['macd_signal']:
                if latest['macd_histogram'] > 0:
                    technical_signals.append(('MACD_BULLISH', 12))
                else:
                    technical_signals.append(('MACD_BULLISH_WEAK', 6))
            elif latest['macd'] < latest['macd_signal']:
                if latest['macd_histogram'] < 0:
                    technical_signals.append(('MACD_BEARISH', -12))
                else:
                    technical_signals.append(('MACD_BEARISH_WEAK', -6))
            elif latest['macd'] < latest['macd_signal'] and latest['macd_histogram'] < 0:
                technical_signals.append(('MACD_BEARISH', -12))
            
            # Bollinger Bands
            if latest['bb_position'] < 0.2:
                technical_signals.append(('BB_OVERSOLD', 10))
            elif latest['bb_position'] > 0.8:
                technical_signals.append(('BB_OVERBOUGHT', -10))
            
            # Trend analysis
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                technical_signals.append(('UPTREND', 8))
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                technical_signals.append(('DOWNTREND', -8))
            
            # Volume confirmation
            if latest['volume_ratio'] > 1.5:
                technical_signals.append(('HIGH_VOLUME', 5))
            
            # Calculate technical score
            technical_score = sum([signal[1] for signal in technical_signals])
            technical_score = max(-50, min(50, technical_score)) + 50  # Normalize to 0-100
            
            # AI Model Predictions (40%)
            features = [latest[f] for f in self.models[symbol]['features']]
            features_scaled = self.scalers[symbol].transform([features])
            
            rf_pred = self.models[symbol]['rf'].predict_proba(features_scaled)[0]
            gb_pred = self.models[symbol]['gb'].predict_proba(features_scaled)[0]
            
            # Combine predictions
            avg_pred = (rf_pred + gb_pred) / 2
            
            # Convert to signal
            if len(avg_pred) >= 5:  # 5 classes: -2, -1, 0, 1, 2
                strong_bearish = avg_pred[0] if len(avg_pred) > 0 else 0
                bearish = avg_pred[1] if len(avg_pred) > 1 else 0
                neutral = avg_pred[2] if len(avg_pred) > 2 else 0
                bullish = avg_pred[3] if len(avg_pred) > 3 else 0
                strong_bullish = avg_pred[4] if len(avg_pred) > 4 else 0
                
                ai_score = (strong_bullish * 100 + bullish * 70 + neutral * 50 + 
                           bearish * 30 + strong_bearish * 0)
            else:
                ai_score = 50  # Neutral if prediction fails
            
            # Momentum Analysis (20%)
            momentum_score = 50
            if latest['momentum'] > 2:
                momentum_score += 20
            elif latest['momentum'] < -2:
                momentum_score -= 20
            
            if latest['trend_strength'] > 2:
                momentum_score += 15
            
            momentum_score = max(0, min(100, momentum_score))
            
            # Combined confidence
            confidence = (technical_score * 0.4) + (ai_score * 0.4) + (momentum_score * 0.2)
            confidence = max(0, min(100, confidence))
            
            # Determine signal with lower thresholds for more activity
            if confidence >= 65:
                signal = 'LONG'
            elif confidence <= 35:
                signal = 'SHORT'
            else:
                signal = 'HOLD'
            
            # Risk assessment
            volatility = latest['atr'] / latest['close'] * 100
            risk_level = 'HIGH' if volatility > 3 else 'MEDIUM' if volatility > 1.5 else 'LOW'
            
            # Recommended leverage
            if risk_level == 'LOW' and confidence >= 80:
                recommended_leverage = min(self.max_leverage, 5)
            elif risk_level == 'MEDIUM' and confidence >= 75:
                recommended_leverage = min(self.max_leverage, 3)
            else:
                recommended_leverage = min(self.max_leverage, 2)
            
            # Calculate targets
            current_price = latest['close']
            atr_value = latest['atr']
            
            if signal == 'LONG':
                stop_loss = current_price - (atr_value * 2)
                take_profit = current_price + (atr_value * 3)
            elif signal == 'SHORT':
                stop_loss = current_price + (atr_value * 2)
                take_profit = current_price - (atr_value * 3)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': round(confidence, 2),
                'technical_score': round(technical_score, 2),
                'ai_score': round(ai_score, 2),
                'momentum_score': round(momentum_score, 2),
                'current_price': current_price,
                'stop_loss': round(stop_loss, 6),
                'take_profit': round(take_profit, 6),
                'recommended_leverage': recommended_leverage,
                'risk_level': risk_level,
                'volatility': round(volatility, 2),
                'rsi': round(latest['rsi'], 2),
                'volume_ratio': round(latest['volume_ratio'], 2),
                'trend_strength': round(latest['trend_strength'], 2),
                'entry_reasons': [signal[0] for signal in technical_signals],
                'market_type': 'futures',
                'trade_direction': signal.lower(),
                'source_engine': 'advanced_futures_trading_engine',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None
    
    def execute_futures_trade(self, signal: Dict) -> bool:
        """Execute futures trade with proper risk management"""
        try:
            if signal['confidence'] < self.min_confidence:
                return False
            
            symbol = signal['symbol']
            side = signal['signal'].lower()  # 'long' or 'short'
            
            if side == 'hold':
                return False
            
            # Get account balance
            balance = self.exchange.fetch_balance()
            available_balance = balance['USDT']['free']
            
            # Calculate position size
            leverage = signal['recommended_leverage']
            risk_amount = available_balance * self.max_position_size
            position_value = risk_amount * leverage
            
            # Calculate position size in base currency
            current_price = signal['current_price']
            position_size = position_value / current_price
            
            # Set leverage
            self.exchange.set_leverage(leverage, symbol)
            
            # Place order
            if side == 'long':
                order = self.exchange.create_market_buy_order(symbol, position_size)
            else:  # short
                order = self.exchange.create_market_sell_order(symbol, position_size)
            
            if order:
                # Save position to database
                self.save_futures_position(signal, order, leverage, position_size)
                
                # Set stop loss and take profit
                self.set_stop_loss_take_profit(signal, order['id'], side)
                
                logger.info(f"🚀 EXECUTED {side.upper()} {symbol}: {position_size:.6f} @ ${current_price} "
                          f"(Confidence: {signal['confidence']}%, Leverage: {leverage}x)")
                return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def save_futures_position(self, signal: Dict, order: Dict, leverage: float, size: float):
        """Save futures position to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO futures_positions 
                (symbol, side, size, entry_price, leverage, stop_loss, take_profit, 
                 confidence, entry_reasons, market_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'],
                signal['signal'],
                size,
                signal['current_price'],
                leverage,
                signal['stop_loss'],
                signal['take_profit'],
                signal['confidence'],
                json.dumps(signal['entry_reasons']),
                json.dumps({
                    'volatility': signal['volatility'],
                    'risk_level': signal['risk_level'],
                    'trend_strength': signal['trend_strength']
                })
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
    
    def set_stop_loss_take_profit(self, signal: Dict, order_id: str, side: str):
        """Set stop loss and take profit orders"""
        try:
            symbol = signal['symbol']
            position_size = signal.get('position_size', 0)
            
            if side == 'long':
                # Stop loss (sell)
                self.exchange.create_stop_loss_order(
                    symbol, 'sell', position_size, signal['stop_loss']
                )
                # Take profit (sell)
                self.exchange.create_take_profit_order(
                    symbol, 'sell', position_size, signal['take_profit']
                )
            else:  # short
                # Stop loss (buy)
                self.exchange.create_stop_loss_order(
                    symbol, 'buy', position_size, signal['stop_loss']
                )
                # Take profit (buy)
                self.exchange.create_take_profit_order(
                    symbol, 'buy', position_size, signal['take_profit']
                )
                
        except Exception as e:
            logger.error(f"Failed to set stop/take profit: {e}")
    
    def scan_futures_opportunities(self) -> List[Dict]:
        """Scan all symbols for futures trading opportunities"""
        signals = []
        
        for symbol in self.symbols:
            try:
                signal = self.generate_futures_signal(symbol)
                if signal and signal['signal'] != 'HOLD':
                    signals.append(signal)
                    
                    # Save signal to database
                    self.save_futures_signal(signal)
                    
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")
                continue
        
        return signals
    
    def save_futures_signal(self, signal: Dict):
        """Save futures signal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO futures_signals 
                (symbol, signal, confidence, technical_score, ai_score, momentum_score,
                 current_price, rsi, volume_ratio, trend_strength, recommended_leverage, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'], signal['signal'], signal['confidence'],
                signal['technical_score'], signal['ai_score'], signal['momentum_score'],
                signal['current_price'], signal['rsi'], signal['volume_ratio'],
                signal['trend_strength'], signal['recommended_leverage'], signal['risk_level']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
    
    def run_futures_trading_cycle(self):
        """Run complete futures trading cycle"""
        try:
            logger.info("🔄 Starting futures trading scan...")
            
            # Scan for opportunities
            signals = self.scan_futures_opportunities()
            
            # Execute high-confidence trades
            executed_trades = 0
            for signal in signals:
                if signal['confidence'] >= self.min_confidence:
                    if self.execute_futures_trade(signal):
                        executed_trades += 1
            
            logger.info(f"✅ Futures scan complete: {len(signals)} signals found, {executed_trades} trades executed")
            
        except Exception as e:
            logger.error(f"Futures trading cycle failed: {e}")

def main():
    """Main futures trading function"""
    try:
        engine = AdvancedFuturesEngine()
        
        # Initialize exchange
        if not engine.initialize_exchange():
            logger.error("Failed to initialize exchange")
            return
        
        # Setup database
        engine.setup_database()
        
        logger.info("🚀 Starting Advanced Futures Trading Engine")
        logger.info(f"Configuration: Min Confidence: {engine.min_confidence}%, Max Leverage: {engine.max_leverage}x, Max Position: {engine.max_position_size*100}%")
        
        # Run continuous trading
        while True:
            engine.run_futures_trading_cycle()
            logger.info("⏰ Next futures scan in 300 seconds...")
            time.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        logger.info("Futures trading engine stopped by user")
    except Exception as e:
        logger.error(f"Futures trading engine failed: {e}")

if __name__ == "__main__":
    main()