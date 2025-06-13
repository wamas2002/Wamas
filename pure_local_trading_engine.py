"""
Pure Local Trading Engine
Autonomous trading system using only local analysis - no external APIs
"""
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PureLocalTradingEngine:
    def __init__(self):
        self.exchange = None
        self.min_confidence = 70.0
        self.max_position_size = 0.25  # 25% max position
        self.stop_loss_pct = 8.0
        self.take_profit_pct = 15.0
        self.scan_interval = 300  # 5 minutes
        
        # Valid symbols only
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'LINK/USDT', 'LTC/USDT', 'DOT/USDT',
            'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT', 'TRX/USDT',
            'ICP/USDT', 'ALGO/USDT', 'HBAR/USDT', 'XLM/USDT', 'SAND/USDT',
            'MANA/USDT', 'THETA/USDT', 'AXS/USDT', 'FIL/USDT', 'ETC/USDT',
            'EGLD/USDT', 'FLOW/USDT', 'ENJ/USDT', 'CHZ/USDT', 'CRV/USDT'
        ]
        
        self.initialize_exchange()
        self.setup_database()
    
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            config = {
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            }
            
            if config['apiKey']:
                self.exchange = ccxt.okx(config)
                self.exchange.load_markets()
                logger.info("Pure local trading engine connected to OKX")
            else:
                logger.warning("OKX credentials not configured")
                
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_database(self):
        """Setup trading database"""
        try:
            with sqlite3.connect('pure_local_trading.db') as conn:
                cursor = conn.cursor()
                
                # Local signals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS local_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        
                        -- Technical indicators
                        rsi REAL,
                        macd REAL,
                        bb_position REAL,
                        volume_ratio REAL,
                        
                        -- Price data
                        price REAL,
                        target_price REAL,
                        stop_loss REAL,
                        
                        -- Local analysis scores
                        trend_score REAL,
                        momentum_score REAL,
                        volume_score REAL,
                        volatility_score REAL,
                        
                        -- Risk assessment
                        risk_level TEXT,
                        position_size REAL,
                        
                        executed BOOLEAN DEFAULT FALSE,
                        execution_price REAL,
                        execution_time DATETIME
                    )
                ''')
                
                # Performance tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS local_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        entry_price REAL,
                        exit_price REAL,
                        profit_loss REAL,
                        profit_pct REAL,
                        hold_time_hours REAL,
                        confidence REAL,
                        actual_outcome TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("Pure local trading database initialized")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """Get market data for analysis"""
        try:
            if not self.exchange:
                return None
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with bulletproof error handling"""
        try:
            result_df = df.copy()
            
            # RSI
            delta = result_df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            rs = avg_gain / avg_loss.replace(0, 0.01)
            result_df['RSI'] = 100 - (100 / (1 + rs))
            result_df['RSI'] = result_df['RSI'].fillna(50.0)
            
            # MACD
            ema_12 = result_df['close'].ewm(span=12, min_periods=1).mean()
            ema_26 = result_df['close'].ewm(span=26, min_periods=1).mean()
            result_df['MACD'] = ema_12 - ema_26
            result_df['MACD_signal'] = result_df['MACD'].ewm(span=9, min_periods=1).mean()
            result_df['MACD_histogram'] = result_df['MACD'] - result_df['MACD_signal']
            
            # Bollinger Bands
            sma_20 = result_df['close'].rolling(window=20, min_periods=1).mean()
            std_20 = result_df['close'].rolling(window=20, min_periods=1).std()
            result_df['BB_upper'] = sma_20 + (std_20 * 2)
            result_df['BB_middle'] = sma_20
            result_df['BB_lower'] = sma_20 - (std_20 * 2)
            
            # BB position (0 = at lower band, 1 = at upper band)
            bb_range = result_df['BB_upper'] - result_df['BB_lower']
            bb_range = bb_range.replace(0, 0.01)  # Avoid division by zero
            result_df['BB_position'] = (result_df['close'] - result_df['BB_lower']) / bb_range
            result_df['BB_position'] = result_df['BB_position'].fillna(0.5)
            
            # Volume analysis
            result_df['volume_sma'] = result_df['volume'].rolling(window=20, min_periods=1).mean()
            result_df['volume_ratio'] = result_df['volume'] / result_df['volume_sma']
            result_df['volume_ratio'] = result_df['volume_ratio'].fillna(1.0)
            
            # EMAs for trend
            result_df['EMA_9'] = result_df['close'].ewm(span=9, min_periods=1).mean()
            result_df['EMA_21'] = result_df['close'].ewm(span=21, min_periods=1).mean()
            result_df['EMA_50'] = result_df['close'].ewm(span=50, min_periods=1).mean()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            return df
    
    def analyze_local_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Generate trading signal using pure local analysis"""
        try:
            latest = df.iloc[-1]
            
            # Initialize scores
            trend_score = 50.0
            momentum_score = 50.0
            volume_score = 50.0
            volatility_score = 50.0
            
            # Trend Analysis
            current_price = latest['close']
            ema_9 = latest['EMA_9']
            ema_21 = latest['EMA_21']
            ema_50 = latest['EMA_50']
            
            # EMA alignment scoring
            if current_price > ema_9 > ema_21 > ema_50:
                trend_score += 25  # Strong uptrend
            elif current_price > ema_9 > ema_21:
                trend_score += 20  # Moderate uptrend
            elif current_price > ema_9:
                trend_score += 15  # Weak uptrend
            
            # Price momentum
            price_change_5 = (current_price - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100
            if price_change_5 > 2:
                trend_score += 10
            elif price_change_5 > 1:
                trend_score += 5
            
            # Momentum Analysis
            rsi = latest['RSI']
            macd = latest['MACD']
            macd_signal = latest['MACD_signal']
            
            # RSI scoring
            if 30 < rsi < 70:
                momentum_score += 15  # Healthy range
            elif rsi < 30:
                momentum_score += 20  # Oversold - potential bounce
            elif rsi > 70:
                momentum_score += 8   # Overbought but momentum can continue
            
            # MACD scoring
            if macd > macd_signal:
                momentum_score += 15
            if latest['MACD_histogram'] > df['MACD_histogram'].iloc[-2]:
                momentum_score += 10  # Improving momentum
            
            # Volume Analysis
            volume_ratio = latest['volume_ratio']
            if volume_ratio > 1.5:
                volume_score += 20  # High volume
            elif volume_ratio > 1.2:
                volume_score += 15
            elif volume_ratio > 1.0:
                volume_score += 10
            
            # Volatility Analysis (Bollinger Bands)
            bb_position = latest['BB_position']
            if 0.2 < bb_position < 0.8:
                volatility_score += 15  # Good range
            elif bb_position < 0.2:
                volatility_score += 20  # Near lower band - potential bounce
            elif bb_position > 0.8:
                volatility_score += 10  # Near upper band
            
            # Calculate composite confidence
            weights = {
                'trend': 0.35,
                'momentum': 0.3,
                'volume': 0.2,
                'volatility': 0.15
            }
            
            composite_score = (
                trend_score * weights['trend'] +
                momentum_score * weights['momentum'] +
                volume_score * weights['volume'] +
                volatility_score * weights['volatility']
            )
            
            # Apply local enhancement boost
            enhanced_confidence = min(composite_score * 1.15, 94.0)  # 15% local boost
            
            # Determine signal
            if enhanced_confidence >= 70:
                signal_type = 'BUY'
            elif enhanced_confidence <= 30:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            # Calculate targets
            target_price = current_price * (1 + self.take_profit_pct / 100)
            stop_loss = current_price * (1 - self.stop_loss_pct / 100)
            
            # Risk assessment
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24 * 365)
            risk_level = "high" if volatility > 2.0 else "medium" if volatility > 1.0 else "low"
            
            # Position sizing based on confidence and risk
            base_position = 0.15  # 15% base
            confidence_multiplier = enhanced_confidence / 100
            risk_multiplier = 0.5 if risk_level == "high" else 0.75 if risk_level == "medium" else 1.0
            position_size = min(base_position * confidence_multiplier * risk_multiplier, self.max_position_size)
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': round(enhanced_confidence, 2),
                'trend_score': round(trend_score, 2),
                'momentum_score': round(momentum_score, 2),
                'volume_score': round(volume_score, 2),
                'volatility_score': round(volatility_score, 2),
                'price': current_price,
                'target_price': round(target_price, 4),
                'stop_loss': round(stop_loss, 4),
                'position_size': round(position_size, 3),
                'risk_level': risk_level,
                'rsi': round(rsi, 2),
                'macd': round(macd, 4),
                'bb_position': round(bb_position, 2),
                'volume_ratio': round(volume_ratio, 2)
            }
            
        except Exception as e:
            logger.error(f"Local signal analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal_type': 'HOLD',
                'confidence': 50.0,
                'risk_level': 'medium'
            }
    
    def save_signal(self, signal: Dict):
        """Save signal to database"""
        try:
            with sqlite3.connect('pure_local_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO local_signals (
                        symbol, signal_type, confidence, trend_score, momentum_score,
                        volume_score, volatility_score, price, target_price, stop_loss,
                        position_size, risk_level, rsi, macd, bb_position, volume_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal['symbol'], signal['signal_type'], signal['confidence'],
                    signal.get('trend_score', 0), signal.get('momentum_score', 0),
                    signal.get('volume_score', 0), signal.get('volatility_score', 0),
                    signal['price'], signal.get('target_price', 0), signal.get('stop_loss', 0),
                    signal.get('position_size', 0), signal['risk_level'],
                    signal.get('rsi', 0), signal.get('macd', 0),
                    signal.get('bb_position', 0), signal.get('volume_ratio', 0)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
    
    def scan_markets(self):
        """Scan all markets for trading opportunities"""
        logger.info("🔄 Starting pure local market scan...")
        
        signals_found = 0
        high_confidence_signals = []
        
        for symbol in self.symbols:
            try:
                # Get market data
                df = self.get_market_data(symbol)
                if df is None or len(df) < 50:
                    continue
                
                # Calculate indicators
                df = self.calculate_technical_indicators(df)
                
                # Generate signal
                signal = self.analyze_local_signal(df, symbol)
                
                # Save all signals
                self.save_signal(signal)
                
                # Log signal
                if signal['confidence'] >= self.min_confidence:
                    high_confidence_signals.append(signal)
                    logger.info(f"📊 {symbol}: {signal['signal_type']} (Confidence: {signal['confidence']}%, Risk: {signal['risk_level']})")
                    signals_found += 1
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Scan failed for {symbol}: {e}")
                continue
        
        logger.info(f"✅ Local scan complete: {signals_found} high-confidence signals found")
        return high_confidence_signals
    
    def run_trading_cycle(self):
        """Run continuous trading cycle"""
        logger.info("🚀 Starting Pure Local Trading Engine")
        logger.info(f"Configuration: Min Confidence: {self.min_confidence}%, Max Position: {self.max_position_size*100}%")
        logger.info(f"Stop Loss: {self.stop_loss_pct}%, Take Profit: {self.take_profit_pct}%")
        
        while True:
            try:
                # Scan markets
                signals = self.scan_markets()
                
                # Execute high-confidence signals (if exchange connected)
                if self.exchange and signals:
                    logger.info(f"Found {len(signals)} signals ready for execution")
                    # In demo mode, we just log signals
                    # In live mode, would execute actual trades here
                
                # Wait for next scan
                logger.info(f"⏰ Next scan in {self.scan_interval} seconds...")
                time.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("Trading engine stopped by user")
                break
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                time.sleep(60)  # Wait 1 minute before retry

def main():
    """Main function"""
    engine = PureLocalTradingEngine()
    engine.run_trading_cycle()

if __name__ == "__main__":
    main()