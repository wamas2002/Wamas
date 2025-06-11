#!/usr/bin/env python3
"""
Advanced ML Optimizer
Machine learning-based signal optimization and performance enhancement
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMLOptimizer:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.exchange = None
        self.models = {}
        self.scalers = {}
        self.performance_tracker = {}
        self.initialize_exchange()
        self.setup_ml_database()
        
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
                    'rateLimit': 800,
                    'enableRateLimit': True,
                })
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")

    def setup_ml_database(self):
        """Setup ML-specific database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ML model performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_model_performance (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    model_name TEXT,
                    symbol TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    training_samples INTEGER,
                    feature_importance TEXT
                )
            ''')
            
            # Signal quality predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_quality_predictions (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    original_confidence REAL,
                    ml_adjusted_confidence REAL,
                    predicted_success_probability REAL,
                    feature_vector TEXT
                )
            ''')
            
            # Market regime classification
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_regime_classification (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    regime_type TEXT,
                    confidence REAL,
                    volatility_level REAL,
                    trend_strength REAL,
                    volume_profile TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"ML database setup error: {e}")

    def collect_training_data(self, symbol, days=30):
        """Collect comprehensive training data"""
        try:
            # Get market data
            ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", '1h', limit=days*24)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate comprehensive features
            features = self.calculate_advanced_features(df)
            
            # Get historical signals and outcomes
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.confidence, s.timestamp, t.pnl
                FROM ai_signals s
                LEFT JOIN (
                    SELECT symbol, pnl, timestamp
                    FROM live_trades
                    WHERE symbol = ?
                ) t ON REPLACE(t.symbol, '/USDT', '') = s.symbol
                WHERE s.symbol = ? AND s.timestamp > datetime('now', '-30 days')
                ORDER BY s.timestamp
            ''', (f"{symbol}/USDT", symbol))
            
            signal_data = cursor.fetchall()
            conn.close()
            
            return features, signal_data
            
        except Exception as e:
            logger.error(f"Training data collection error for {symbol}: {e}")
            return None, None

    def calculate_advanced_features(self, df):
        """Calculate advanced technical features for ML"""
        features = []
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Moving averages and ratios
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
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
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price patterns
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Compile feature vectors
        feature_columns = [
            'rsi', 'macd', 'macd_histogram', 'bb_position', 'volume_ratio',
            'volatility', 'price_sma_5_ratio', 'price_sma_20_ratio', 
            'momentum_5', 'momentum_10', 'momentum_20', 'high_low_ratio', 'close_position'
        ]
        
        for i in range(30, len(df)):  # Need enough data for indicators
            feature_vector = []
            for col in feature_columns:
                if col in df.columns and not pd.isna(df[col].iloc[i]):
                    feature_vector.append(df[col].iloc[i])
                else:
                    feature_vector.append(0)
            
            features.append({
                'timestamp': df['timestamp'].iloc[i],
                'features': feature_vector
            })
        
        return features

    def train_signal_quality_model(self, symbol):
        """Train ML model to predict signal quality"""
        try:
            features_data, signal_data = self.collect_training_data(symbol)
            
            if not features_data or not signal_data or len(signal_data) < 10:
                return None
            
            # Prepare training data
            X = []
            y = []
            
            for signal in signal_data:
                confidence, timestamp, pnl = signal
                
                # Find corresponding features
                signal_time = datetime.fromisoformat(timestamp)
                matching_features = None
                
                for feature_data in features_data:
                    if abs((feature_data['timestamp'] - signal_time).total_seconds()) < 3600:  # Within 1 hour
                        matching_features = feature_data['features']
                        break
                
                if matching_features and pnl is not None:
                    # Add confidence to features
                    feature_vector = matching_features + [float(confidence)]
                    X.append(feature_vector)
                    
                    # Label: 1 if profitable, 0 if not
                    y.append(1 if float(pnl) > 0 else 0)
            
            if len(X) < 5:
                return None
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            
            # Train ensemble of models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = 0
            
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                
                # Save model performance
                self.save_model_performance(model_name, symbol, accuracy, precision, recall, len(X_train))
                
                logger.info(f"{symbol} {model_name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
            
            # Save best model and scaler
            self.models[symbol] = best_model
            self.scalers[symbol] = scaler
            
            # Save model files
            model_path = f'ml_models/{symbol}_quality_model.pkl'
            scaler_path = f'ml_models/{symbol}_scaler.pkl'
            
            os.makedirs('ml_models', exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            return best_model
            
        except Exception as e:
            logger.error(f"Model training error for {symbol}: {e}")
            return None

    def predict_signal_quality(self, symbol, confidence, current_features):
        """Predict signal quality using trained ML model"""
        try:
            if symbol not in self.models or symbol not in self.scalers:
                return confidence  # Return original confidence if no model
            
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            # Prepare feature vector
            feature_vector = current_features + [confidence]
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Predict success probability
            success_probability = model.predict_proba(feature_vector_scaled)[0][1]
            
            # Adjust confidence based on ML prediction
            adjustment_factor = success_probability / 0.5  # 0.5 is neutral
            adjusted_confidence = confidence * adjustment_factor
            
            # Cap confidence between 20-95
            adjusted_confidence = max(20, min(95, adjusted_confidence))
            
            # Save prediction
            self.save_signal_prediction(symbol, confidence, adjusted_confidence, success_probability)
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Signal quality prediction error for {symbol}: {e}")
            return confidence

    def classify_market_regime(self):
        """Classify current market regime using ML"""
        try:
            # Get market data for major symbols
            btc_data = self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
            eth_data = self.exchange.fetch_ohlcv('ETH/USDT', '1h', limit=100)
            
            btc_df = pd.DataFrame(btc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            eth_df = pd.DataFrame(eth_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate market features
            btc_returns = btc_df['close'].pct_change()
            eth_returns = eth_df['close'].pct_change()
            
            # Volatility
            btc_vol = btc_returns.rolling(window=24).std()
            eth_vol = eth_returns.rolling(window=24).std()
            avg_volatility = (btc_vol.iloc[-1] + eth_vol.iloc[-1]) / 2
            
            # Trend strength
            btc_trend = (btc_df['close'].iloc[-1] / btc_df['close'].iloc[-24] - 1)
            eth_trend = (eth_df['close'].iloc[-1] / eth_df['close'].iloc[-24] - 1)
            avg_trend_strength = abs((btc_trend + eth_trend) / 2)
            
            # Volume profile
            btc_vol_ratio = btc_df['volume'].iloc[-24:].mean() / btc_df['volume'].iloc[-48:-24].mean()
            eth_vol_ratio = eth_df['volume'].iloc[-24:].mean() / eth_df['volume'].iloc[-48:-24].mean()
            avg_volume_ratio = (btc_vol_ratio + eth_vol_ratio) / 2
            
            # Classify regime
            if avg_volatility > 0.05 and avg_volume_ratio > 1.2:
                regime = "HIGH_VOLATILITY"
                confidence = min(90, 60 + avg_volatility * 500)
            elif avg_trend_strength > 0.1 and avg_volume_ratio > 1.1:
                regime = "TRENDING"
                confidence = min(90, 60 + avg_trend_strength * 200)
            elif avg_volatility < 0.02 and avg_volume_ratio < 0.9:
                regime = "LOW_VOLATILITY"
                confidence = min(90, 70 - avg_volatility * 1000)
            else:
                regime = "NEUTRAL"
                confidence = 60
            
            # Save regime classification
            self.save_market_regime(regime, confidence, avg_volatility, avg_trend_strength)
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Market regime classification error: {e}")
            return "NEUTRAL", 60

    def optimize_trading_parameters(self):
        """Optimize trading parameters based on ML insights"""
        try:
            # Get recent performance data
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, side, ai_confidence, pnl, timestamp
                FROM live_trades 
                WHERE timestamp > datetime('now', '-7 days')
                AND pnl IS NOT NULL
                ORDER BY timestamp DESC
            ''')
            
            trades = cursor.fetchall()
            conn.close()
            
            if len(trades) < 5:
                return None
            
            # Analyze performance by confidence levels
            confidence_performance = {}
            
            for trade in trades:
                symbol, side, confidence, pnl, timestamp = trade
                confidence_bucket = int(float(confidence) / 10) * 10  # Round to nearest 10
                
                if confidence_bucket not in confidence_performance:
                    confidence_performance[confidence_bucket] = []
                
                confidence_performance[confidence_bucket].append(float(pnl))
            
            # Find optimal confidence threshold
            optimal_threshold = 60
            best_performance = -float('inf')
            
            for threshold in range(50, 90, 10):
                eligible_trades = [pnl for conf, pnls in confidence_performance.items() 
                                if conf >= threshold for pnl in pnls]
                
                if len(eligible_trades) >= 3:
                    avg_pnl = np.mean(eligible_trades)
                    win_rate = len([p for p in eligible_trades if p > 0]) / len(eligible_trades)
                    performance_score = avg_pnl * win_rate
                    
                    if performance_score > best_performance:
                        best_performance = performance_score
                        optimal_threshold = threshold
            
            # Update trading parameters
            self.update_system_parameters(optimal_threshold)
            
            logger.info(f"ML Optimization: Optimal confidence threshold = {optimal_threshold}")
            
            return optimal_threshold
            
        except Exception as e:
            logger.error(f"Parameter optimization error: {e}")
            return None

    def save_model_performance(self, model_name, symbol, accuracy, precision, recall, samples):
        """Save ML model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ml_model_performance 
                (timestamp, model_name, symbol, accuracy, precision_score, recall_score, training_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                model_name,
                symbol,
                accuracy,
                precision,
                recall,
                samples
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving model performance: {e}")

    def save_signal_prediction(self, symbol, original_conf, adjusted_conf, success_prob):
        """Save signal quality prediction"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signal_quality_predictions 
                (timestamp, symbol, original_confidence, ml_adjusted_confidence, predicted_success_probability)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                original_conf,
                adjusted_conf,
                success_prob
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving signal prediction: {e}")

    def save_market_regime(self, regime, confidence, volatility, trend_strength):
        """Save market regime classification"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_regime_classification 
                (timestamp, regime_type, confidence, volatility_level, trend_strength)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                regime,
                confidence,
                volatility,
                trend_strength
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving market regime: {e}")

    def update_system_parameters(self, optimal_threshold):
        """Update system trading parameters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update or insert optimized parameters
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_optimized_parameters (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    optimal_confidence_threshold REAL,
                    optimization_method TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO ml_optimized_parameters 
                (timestamp, optimal_confidence_threshold, optimization_method)
                VALUES (?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                optimal_threshold,
                'ML_PERFORMANCE_ANALYSIS'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating system parameters: {e}")

    def run_ml_optimization_cycle(self):
        """Run complete ML optimization cycle"""
        logger.info("Running Advanced ML Optimization Cycle")
        
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        
        # Train models for each symbol
        for symbol in crypto_symbols:
            logger.info(f"Training ML model for {symbol}")
            self.train_signal_quality_model(symbol)
            time.sleep(1)  # Rate limiting
        
        # Classify market regime
        regime, confidence = self.classify_market_regime()
        logger.info(f"Market Regime: {regime} ({confidence:.1f}% confidence)")
        
        # Optimize trading parameters
        optimal_threshold = self.optimize_trading_parameters()
        
        logger.info("ML Optimization Cycle Complete")
        
        return {
            'models_trained': len(self.models),
            'market_regime': regime,
            'optimal_threshold': optimal_threshold
        }

def main():
    """Main ML optimization function"""
    optimizer = AdvancedMLOptimizer()
    optimizer.run_ml_optimization_cycle()

if __name__ == "__main__":
    main()