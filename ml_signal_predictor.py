#!/usr/bin/env python3
"""
Machine Learning Signal Predictor
Advanced AI-powered signal generation with multiple model ensemble
"""

import numpy as np
import pandas as pd
import sqlite3
import os
import ccxt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLSignalPredictor:
    def __init__(self):
        self.exchange = self.connect_okx()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_weights = {
            'random_forest': 0.4,
            'gradient_boost': 0.35,
            'logistic_regression': 0.25
        }
        
    def connect_okx(self):
        """Connect to OKX exchange"""
        try:
            return ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'rateLimit': 2000,
                'enableRateLimit': True,
            })
        except Exception as e:
            print(f"ML predictor OKX connection error: {e}")
            return None
    
    def fetch_training_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch market data for model training"""
        if not self.exchange:
            return pd.DataFrame()
        
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=days * 24)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            print(f"Data fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty:
            return df
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=10).std()
        
        return df
    
    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 4) -> pd.DataFrame:
        """Create trading labels based on future price movements"""
        if df.empty:
            return df
        
        # Calculate future returns
        df['future_return'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
        
        # Create labels: 1 for buy signal, 0 for hold/sell
        buy_threshold = 0.02  # 2% gain threshold
        df['label'] = (df['future_return'] > buy_threshold).astype(int)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix and labels for training"""
        if df.empty:
            return np.array([]), np.array([])
        
        # Select feature columns (exclude non-feature columns)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + ['label']].dropna()
        
        if df_clean.empty:
            return np.array([]), np.array([])
        
        X = df_clean[feature_cols].values
        y = df_clean['label'].values
        
        self.feature_columns = feature_cols
        
        return X, y
    
    def train_models(self, symbol: str, retrain: bool = False):
        """Train ensemble of ML models"""
        model_path = f'models_{symbol.replace("/", "_")}.joblib'
        
        if os.path.exists(model_path) and not retrain:
            print(f"Loading existing models for {symbol}")
            self.models[symbol] = joblib.load(model_path)
            return
        
        print(f"Training ML models for {symbol}")
        
        # Fetch and prepare data
        df = self.fetch_training_data(symbol, days=60)
        if df.empty:
            print(f"No data available for {symbol}")
            return
        
        # Calculate features and labels
        df = self.calculate_technical_features(df)
        df = self.create_labels(df)
        
        X, y = self.prepare_features(df)
        
        if len(X) == 0 or len(y) == 0:
            print(f"Insufficient data for training {symbol}")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {}
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        models['random_forest'] = {'model': rf_model, 'accuracy': rf_accuracy}
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        models['gradient_boost'] = {'model': gb_model, 'accuracy': gb_accuracy}
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        models['logistic_regression'] = {'model': lr_model, 'accuracy': lr_accuracy}
        
        # Store models and scaler
        self.models[symbol] = models
        self.scalers[symbol] = scaler
        
        # Save models
        joblib.dump({'models': models, 'scaler': scaler, 'feature_columns': self.feature_columns}, model_path)
        
        print(f"Model training completed for {symbol}")
        print(f"Random Forest accuracy: {rf_accuracy:.3f}")
        print(f"Gradient Boost accuracy: {gb_accuracy:.3f}")
        print(f"Logistic Regression accuracy: {lr_accuracy:.3f}")
    
    def generate_ml_signal(self, symbol: str) -> dict:
        """Generate trading signal using ensemble of ML models"""
        if symbol not in self.models:
            self.train_models(symbol)
        
        if symbol not in self.models:
            return {'signal': 'HOLD', 'confidence': 0, 'error': 'Model training failed'}
        
        # Fetch recent data
        df = self.fetch_training_data(symbol, days=5)
        if df.empty:
            return {'signal': 'HOLD', 'confidence': 0, 'error': 'No recent data'}
        
        # Calculate features for latest data point
        df = self.calculate_technical_features(df)
        
        # Get latest features
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols and col in self.feature_columns]
        
        if not feature_cols:
            return {'signal': 'HOLD', 'confidence': 0, 'error': 'No features available'}
        
        latest_features = df[feature_cols].iloc[-1:].values
        
        if np.isnan(latest_features).any():
            return {'signal': 'HOLD', 'confidence': 0, 'error': 'NaN in features'}
        
        # Scale features
        scaler = self.scalers.get(symbol)
        if scaler is None:
            return {'signal': 'HOLD', 'confidence': 0, 'error': 'No scaler available'}
        
        latest_features_scaled = scaler.transform(latest_features)
        
        # Get predictions from all models
        models = self.models[symbol]
        predictions = {}
        probabilities = {}
        
        for model_name, model_data in models.items():
            model = model_data['model']
            pred = model.predict(latest_features_scaled)[0]
            pred_proba = model.predict_proba(latest_features_scaled)[0]
            
            predictions[model_name] = pred
            probabilities[model_name] = pred_proba[1] if len(pred_proba) > 1 else 0.5
        
        # Calculate ensemble prediction
        weighted_prob = sum(
            probabilities[model_name] * self.model_weights[model_name]
            for model_name in predictions.keys()
        )
        
        # Generate signal
        confidence_threshold = 0.6
        
        if weighted_prob > confidence_threshold:
            signal = 'BUY'
            confidence = int(weighted_prob * 100)
        else:
            signal = 'HOLD'
            confidence = int((1 - weighted_prob) * 100)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'ml_probability': weighted_prob,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_generate_signals(self, symbols: list) -> dict:
        """Generate ML signals for multiple symbols"""
        signals = {}
        
        for symbol in symbols:
            try:
                signal_data = self.generate_ml_signal(symbol)
                signals[symbol] = signal_data
                
                # Save to database
                self.save_ml_signal_to_db(symbol, signal_data)
                
            except Exception as e:
                signals[symbol] = {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'error': str(e)
                }
        
        return signals
    
    def save_ml_signal_to_db(self, symbol: str, signal_data: dict):
        """Save ML signal to database"""
        try:
            conn = sqlite3.connect('trading_platform.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    signal TEXT,
                    confidence INTEGER,
                    ml_probability REAL,
                    model_details TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO ml_signals 
                (timestamp, symbol, signal, confidence, ml_probability, model_details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                signal_data['signal'],
                signal_data['confidence'],
                signal_data.get('ml_probability', 0),
                str(signal_data.get('individual_predictions', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"ML signal database save error: {e}")

def main():
    """Test ML signal predictor"""
    predictor = MLSignalPredictor()
    
    print("MACHINE LEARNING SIGNAL PREDICTOR")
    print("=" * 50)
    
    # Test symbols
    symbols = ['BTC/USDT', 'ETH/USDT']
    
    # Generate signals
    signals = predictor.batch_generate_signals(symbols)
    
    for symbol, signal_data in signals.items():
        print(f"\n{symbol}:")
        print(f"  Signal: {signal_data['signal']}")
        print(f"  Confidence: {signal_data['confidence']}%")
        if 'ml_probability' in signal_data:
            print(f"  ML Probability: {signal_data['ml_probability']:.3f}")
    
    print("\n" + "=" * 50)

if __name__ == '__main__':
    main()