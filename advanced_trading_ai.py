#!/usr/bin/env python3
"""
Advanced Trading AI System
Sophisticated ML-driven trading with continuous learning, adaptive strategies, and active BUY/SELL signals
"""

import sqlite3
import ccxt
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTradingAI:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.exchange = None
        self.models = {}
        self.performance_history = {}
        self.market_regime = "neutral"
        self.confidence_threshold = 55.0  # Dynamic threshold
        self.initialize_exchange()
        self.initialize_ai_models()
        
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
                logger.info("Trading AI connected to OKX")
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")

    def initialize_ai_models(self):
        """Initialize AI models for different market conditions"""
        self.models = {
            'trend_following': RandomForestClassifier(n_estimators=100, random_state=42),
            'mean_reversion': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'momentum': RandomForestClassifier(n_estimators=80, random_state=42),
            'volatility': GradientBoostingClassifier(n_estimators=80, random_state=42)
        }
        logger.info("AI models initialized")

    def get_market_data(self, symbol, timeframe='1h', limit=200):
        """Get comprehensive market data for analysis"""
        try:
            symbol_formatted = f"{symbol}/USDT" if not symbol.endswith('/USDT') else symbol
            ohlcv = self.exchange.fetch_ohlcv(symbol_formatted, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate advanced technical indicators
            df = self.calculate_advanced_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Market data error for {symbol}: {e}")
            return None

    def calculate_advanced_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        
        # Price-based indicators
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Support/Resistance levels
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        return df

    def detect_market_regime(self, df):
        """Detect current market regime (trending, ranging, volatile)"""
        try:
            current_volatility = df['volatility'].iloc[-1]
            price_momentum = df['momentum_20'].iloc[-1]
            volume_trend = df['volume_ratio'].iloc[-5:].mean()
            
            # Regime classification
            if current_volatility > 0.05 and volume_trend > 1.2:
                regime = "high_volatility"
            elif abs(price_momentum) > 0.15:
                regime = "trending"
            elif current_volatility < 0.02:
                regime = "ranging"
            else:
                regime = "neutral"
                
            self.market_regime = regime
            return regime
            
        except Exception as e:
            logger.error(f"Market regime detection error: {e}")
            return "neutral"

    def generate_features(self, df):
        """Generate ML features for prediction"""
        features = []
        
        # Technical indicator features
        feature_columns = [
            'rsi', 'macd', 'macd_histogram', 'bb_position', 'volume_ratio',
            'volatility', 'momentum_5', 'momentum_10', 'momentum_20', 'price_position'
        ]
        
        for col in feature_columns:
            if col in df.columns:
                features.extend([
                    df[col].iloc[-1],  # Current value
                    df[col].rolling(5).mean().iloc[-1],  # 5-period average
                    df[col].diff().iloc[-1]  # Rate of change
                ])
        
        # Price action features
        features.extend([
            (df['close'].iloc[-1] - df['sma_20'].iloc[-1]) / df['sma_20'].iloc[-1],  # Distance from SMA
            (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1],  # Current candle body
            df['high'].iloc[-1] / df['close'].iloc[-1] - 1,  # Upper wick ratio
            df['close'].iloc[-1] / df['low'].iloc[-1] - 1   # Lower wick ratio
        ])
        
        return np.array(features).reshape(1, -1)

    def create_training_labels(self, df):
        """Create training labels based on future price movements"""
        labels = []
        
        for i in range(len(df) - 5):  # Look 5 periods ahead
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + 5]
            price_change = (future_price - current_price) / current_price
            
            if price_change > 0.02:  # 2% gain threshold
                labels.append(1)  # BUY
            elif price_change < -0.02:  # 2% loss threshold
                labels.append(-1)  # SELL
            else:
                labels.append(0)  # HOLD
                
        return labels

    def train_models(self, symbol):
        """Train AI models with historical data"""
        try:
            df = self.get_market_data(symbol, limit=500)
            if df is None or len(df) < 100:
                return False
                
            # Prepare training data
            features_list = []
            labels = self.create_training_labels(df)
            
            for i in range(30, len(df) - 5):  # Need enough data for indicators
                df_slice = df.iloc[:i+1]
                features = self.generate_features(df_slice)
                if features is not None and features.shape[1] > 0:
                    features_list.append(features.flatten())
            
            if len(features_list) < 50:
                return False
                
            X = np.array(features_list)
            y = np.array(labels[:len(features_list)])
            
            # Train models
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            for model_name, model in self.models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store performance
                if symbol not in self.performance_history:
                    self.performance_history[symbol] = {}
                self.performance_history[symbol][model_name] = accuracy
                
                logger.info(f"Model {model_name} for {symbol}: {accuracy:.3f} accuracy")
            
            return True
            
        except Exception as e:
            logger.error(f"Model training error for {symbol}: {e}")
            return False

    def generate_smart_signals(self, symbol):
        """Generate intelligent trading signals using ensemble of models"""
        try:
            df = self.get_market_data(symbol)
            if df is None or len(df) < 50:
                return None
                
            # Detect market regime
            regime = self.detect_market_regime(df)
            
            # Generate features
            features = self.generate_features(df)
            if features is None:
                return None
                
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    # Get prediction and probability
                    pred = model.predict(features)[0]
                    proba = model.predict_proba(features)[0]
                    confidence = np.max(proba)
                    
                    predictions[model_name] = pred
                    confidences[model_name] = confidence
                    
                except Exception as e:
                    logger.error(f"Prediction error for {model_name}: {e}")
                    continue
            
            if not predictions:
                return None
                
            # Ensemble decision with regime weighting
            regime_weights = {
                'trending': {'trend_following': 1.5, 'momentum': 1.3, 'mean_reversion': 0.7, 'volatility': 1.0},
                'ranging': {'mean_reversion': 1.5, 'volatility': 1.2, 'trend_following': 0.6, 'momentum': 0.8},
                'high_volatility': {'volatility': 1.4, 'momentum': 1.2, 'trend_following': 0.9, 'mean_reversion': 0.8},
                'neutral': {'trend_following': 1.0, 'momentum': 1.0, 'mean_reversion': 1.0, 'volatility': 1.0}
            }
            
            weights = regime_weights.get(regime, regime_weights['neutral'])
            
            weighted_signal = 0
            total_weight = 0
            total_confidence = 0
            
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 1.0)
                weighted_signal += pred * weight * confidences[model_name]
                total_weight += weight
                total_confidence += confidences[model_name]
            
            if total_weight == 0:
                return None
                
            final_signal = weighted_signal / total_weight
            avg_confidence = (total_confidence / len(predictions)) * 100
            
            # Determine action
            if final_signal > 0.3:
                action = 'BUY'
            elif final_signal < -0.3:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # Generate reasoning
            reasoning = f"AI Ensemble ({regime} market): {len(predictions)} models, "
            reasoning += f"weighted signal: {final_signal:.2f}, "
            reasoning += f"top models: {sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:2]}"
            
            return {
                'symbol': symbol,
                'signal': action,
                'confidence': min(95, max(45, avg_confidence)),
                'reasoning': reasoning,
                'market_regime': regime,
                'model_consensus': len([p for p in predictions.values() if p == np.sign(final_signal)]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None

    def adaptive_learning(self):
        """Continuously adapt and improve models based on performance"""
        try:
            # Adjust confidence threshold based on recent performance
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent trades performance
            cursor.execute('''
                SELECT symbol, side, ai_confidence, pnl
                FROM live_trades 
                WHERE timestamp > datetime('now', '-24 hours')
                AND pnl IS NOT NULL
            ''')
            
            recent_trades = cursor.fetchall()
            conn.close()
            
            if len(recent_trades) >= 5:
                # Calculate win rate and adjust threshold
                profitable_trades = [t for t in recent_trades if t[3] > 0]
                win_rate = len(profitable_trades) / len(recent_trades)
                
                if win_rate > 0.7:
                    # High win rate, can be more aggressive
                    self.confidence_threshold = max(45, self.confidence_threshold - 2)
                elif win_rate < 0.4:
                    # Low win rate, be more conservative
                    self.confidence_threshold = min(75, self.confidence_threshold + 3)
                
                logger.info(f"Adaptive learning: Win rate {win_rate:.2f}, threshold adjusted to {self.confidence_threshold}")
            
        except Exception as e:
            logger.error(f"Adaptive learning error: {e}")

    def run_ai_trading_cycle(self):
        """Execute complete AI trading cycle"""
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        generated_signals = []
        
        logger.info("🤖 Advanced AI Trading Cycle Starting")
        
        for symbol in crypto_symbols:
            # Train models periodically
            if np.random.random() < 0.3:  # 30% chance to retrain
                logger.info(f"Retraining models for {symbol}")
                self.train_models(symbol)
            
            # Generate smart signals
            signal = self.generate_smart_signals(symbol)
            if signal and signal['confidence'] >= self.confidence_threshold:
                generated_signals.append(signal)
        
        # Save signals to database
        if generated_signals:
            self.save_ai_signals(generated_signals)
            logger.info(f"Generated {len(generated_signals)} high-confidence AI signals")
            
            for signal in generated_signals:
                logger.info(f"{signal['symbol']}: {signal['signal']} ({signal['confidence']:.1f}%) - {signal['market_regime']}")
        
        # Adaptive learning
        self.adaptive_learning()
        
        return generated_signals

    def save_ai_signals(self, signals):
        """Save AI-generated signals to database"""
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
            logger.error(f"Error saving AI signals: {e}")

def run_advanced_trading_ai():
    """Main function to run advanced trading AI"""
    ai_trader = AdvancedTradingAI()
    
    # Initial training for all symbols
    crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
    logger.info("🎯 Initial AI model training...")
    
    for symbol in crypto_symbols:
        ai_trader.train_models(symbol)
        time.sleep(1)  # Rate limiting
    
    # Run continuous AI trading
    while True:
        try:
            signals = ai_trader.run_ai_trading_cycle()
            time.sleep(300)  # Run every 5 minutes
            
        except KeyboardInterrupt:
            logger.info("AI trading stopped")
            break
        except Exception as e:
            logger.error(f"AI trading cycle error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_advanced_trading_ai()