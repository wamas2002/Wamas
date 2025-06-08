"""
Comprehensive ML Pipeline for Trading Bot Integration
Works with existing OKX data feeds and trading engine
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb

# Handle optional dependencies
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    lgb = None
    
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except (ImportError, OSError):
    cb = None
import joblib
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

class TradingMLPipeline:
    """Enhanced ML pipeline for integration with existing trading bot"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._setup_database()
    
    def _setup_database(self):
        """Setup ML pipeline database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ML predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    confidence REAL,
                    features_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    f1_score REAL,
                    precision_score REAL,
                    recall_score REAL,
                    training_samples INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database setup error: {e}")
    
    def generate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive features optimized for live trading"""
        
        data = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                self.logger.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        try:
            # Technical indicators
            data['rsi_14'] = ta.rsi(data['close'], length=14)
            data['rsi_7'] = ta.rsi(data['close'], length=7)
            data['rsi_21'] = ta.rsi(data['close'], length=21)
            
            # MACD
            macd = ta.macd(data['close'])
            if macd is not None and not macd.empty:
                data['macd'] = macd.iloc[:, 0]
                data['macd_signal'] = macd.iloc[:, 2]
                data['macd_histogram'] = macd.iloc[:, 1]
            
            # Bollinger Bands
            bb = ta.bbands(data['close'], length=20)
            if bb is not None and not bb.empty:
                data['bb_upper'] = bb.iloc[:, 0]
                data['bb_middle'] = bb.iloc[:, 1]
                data['bb_lower'] = bb.iloc[:, 2]
                data['bb_width'] = bb.iloc[:, 3]
                data['bb_percent'] = bb.iloc[:, 4]
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                data[f'sma_{period}'] = ta.sma(data['close'], length=period)
                data[f'ema_{period}'] = ta.ema(data['close'], length=period)
            
            # Volume indicators
            data['obv'] = ta.obv(data['close'], data['volume'])
            data['vwap'] = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
            
            # ATR
            data['atr_14'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            
            # Stochastic
            stoch = ta.stoch(data['high'], data['low'], data['close'])
            if stoch is not None and not stoch.empty:
                data['stoch_k'] = stoch.iloc[:, 1]
                data['stoch_d'] = stoch.iloc[:, 0]
            
            # Price action features
            data['body_size'] = abs(data['close'] - data['open'])
            data['upper_wick'] = data['high'] - np.maximum(data['close'], data['open'])
            data['lower_wick'] = np.minimum(data['close'], data['open']) - data['low']
            data['total_range'] = data['high'] - data['low']
            
            # Normalized features
            data['body_ratio'] = data['body_size'] / (data['total_range'] + 1e-8)
            data['upper_wick_ratio'] = data['upper_wick'] / (data['total_range'] + 1e-8)
            data['lower_wick_ratio'] = data['lower_wick'] / (data['total_range'] + 1e-8)
            
            # Price changes
            for period in [1, 2, 3, 5, 10]:
                data[f'price_change_{period}'] = data['close'].pct_change(period)
                data[f'volume_change_{period}'] = data['volume'].pct_change(period)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                data[f'price_std_{window}'] = data['close'].rolling(window).std()
                data[f'volume_ma_{window}'] = data['volume'].rolling(window).mean()
                
                # Price position within rolling range
                data[f'price_min_{window}'] = data['close'].rolling(window).min()
                data[f'price_max_{window}'] = data['close'].rolling(window).max()
                data[f'price_position_{window}'] = (data['close'] - data[f'price_min_{window}']) / (
                    data[f'price_max_{window}'] - data[f'price_min_{window}'] + 1e-8
                )
            
            # Time-based features
            if 'timestamp' in data.columns:
                dt_series = pd.to_datetime(data['timestamp'], unit='ms')
                data['hour'] = dt_series.dt.hour
                data['day_of_week'] = dt_series.dt.dayofweek
                data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
                data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
                data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
                data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                data[f'close_lag_{lag}'] = data['close'].shift(lag)
                data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
            
            # Clean data
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Get feature columns (exclude OHLCV and metadata)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime', 'symbol']
            feature_cols = [col for col in data.columns if col not in exclude_cols and not col.endswith('_min_') and not col.endswith('_max_')]
            
            # Fill NaN values
            data[feature_cols] = data[feature_cols].fillna(method='ffill').fillna(0)
            
            self.logger.info(f"Generated {len(feature_cols)} features")
            return data
            
        except Exception as e:
            self.logger.error(f"Feature generation error: {e}")
            return data
    
    def prepare_training_data(self, symbol: str, lookback_days: int = 30) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare training data from database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent OHLCV data
            cutoff_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=[symbol, cutoff_time])
            conn.close()
            
            if df.empty or len(df) < 100:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} records")
                return pd.DataFrame(), []
            
            # Generate features
            df = self.generate_enhanced_features(df)
            
            # Create labels (predict if price will be higher in 5 minutes)
            df['future_price'] = df['close'].shift(-5)
            df['target'] = (df['future_price'] > df['close'] * 1.002).astype(int)  # 0.2% threshold
            
            # Remove rows with NaN targets
            df = df.dropna(subset=['target'])
            
            if len(df) < 50:
                self.logger.warning(f"Insufficient valid samples for {symbol}: {len(df)}")
                return pd.DataFrame(), []
            
            # Get feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime', 
                           'symbol', 'target', 'future_price']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            self.logger.info(f"Prepared training data for {symbol}: {len(df)} samples, {len(feature_cols)} features")
            return df, feature_cols
            
        except Exception as e:
            self.logger.error(f"Error preparing training data for {symbol}: {e}")
            return pd.DataFrame(), []
    
    def train_models(self, symbol: str) -> Dict[str, Any]:
        """Train multiple ML models for a symbol"""
        
        # Prepare data
        df, feature_cols = self.prepare_training_data(symbol)
        
        if df.empty:
            return {}
        
        # Prepare features and targets
        X = df[feature_cols].values
        y = df['target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {
            'xgboost': xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostClassifier(iterations=100, depth=5, random_state=42, verbose=False)
        
        results = {}
        best_f1 = 0
        best_model_name = None
        
        for model_name, model in models.items():
            try:
                self.logger.info(f"Training {model_name} for {symbol}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                }
                
                # Track best model
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = model_name
                
                self.logger.info(f"{model_name} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
                # Store metrics in database
                self._store_model_metrics(symbol, model_name, f1, precision, recall, len(X_train))
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                continue
        
        if best_model_name:
            # Store best model and scaler
            self.models[symbol] = results[best_model_name]['model']
            self.scalers[symbol] = scaler
            self.feature_columns[symbol] = feature_cols
            
            self.logger.info(f"Best model for {symbol}: {best_model_name} (F1: {best_f1:.4f})")
        
        return results
    
    def _store_model_metrics(self, symbol: str, model_name: str, f1: float, 
                           precision: float, recall: float, samples: int):
        """Store model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_metrics 
                (symbol, model_name, f1_score, precision_score, recall_score, training_samples)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, model_name, f1, precision, recall, samples))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")
    
    def predict(self, symbol: str, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ML prediction for current market data"""
        
        if symbol not in self.models:
            return {'prediction': 0, 'confidence': 0.0, 'status': 'no_model'}
        
        try:
            # Generate features
            featured_data = self.generate_enhanced_features(current_data)
            
            if featured_data.empty:
                return {'prediction': 0, 'confidence': 0.0, 'status': 'feature_error'}
            
            # Get latest row
            latest_row = featured_data.iloc[-1]
            
            # Extract features
            feature_cols = self.feature_columns[symbol]
            features = []
            
            for col in feature_cols:
                if col in latest_row:
                    value = latest_row[col]
                    features.append(0.0 if pd.isna(value) else float(value))
                else:
                    features.append(0.0)
            
            # Scale features
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scalers[symbol].transform(X)
            
            # Make prediction
            model = self.models[symbol]
            prediction = model.predict(X_scaled)[0]
            
            # Get confidence if available
            confidence = 0.5
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                confidence = max(proba)
            
            # Store prediction
            timestamp = int(datetime.now().timestamp() * 1000)
            self._store_prediction(symbol, timestamp, prediction, confidence, features)
            
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'status': 'success',
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error for {symbol}: {e}")
            return {'prediction': 0, 'confidence': 0.0, 'status': 'error'}
    
    def _store_prediction(self, symbol: str, timestamp: int, prediction: int, 
                         confidence: float, features: List[float]):
        """Store prediction in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            datetime_str = datetime.fromtimestamp(timestamp / 1000).isoformat()
            features_json = str(features)  # Simple string representation
            
            cursor.execute('''
                INSERT INTO ml_predictions 
                (symbol, timestamp, datetime, model_name, prediction, confidence, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timestamp, datetime_str, 'best_model', prediction, confidence, features_json))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing prediction: {e}")
    
    def get_recent_predictions(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get recent predictions for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            
            query = """
                SELECT timestamp, datetime, prediction, confidence
                FROM ml_predictions 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[symbol, cutoff_time])
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting predictions: {e}")
            return pd.DataFrame()
    
    def retrain_if_needed(self, symbol: str, hours_threshold: int = 24) -> bool:
        """Check if model needs retraining and retrain if necessary"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check last training time
            cursor.execute('''
                SELECT MAX(created_at) FROM model_metrics 
                WHERE symbol = ?
            ''', (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result[0]:
                # No previous training
                self.logger.info(f"No previous training found for {symbol}, training now")
                self.train_models(symbol)
                return True
            
            last_training = datetime.fromisoformat(result[0])
            hours_since = (datetime.now() - last_training).total_seconds() / 3600
            
            if hours_since >= hours_threshold:
                self.logger.info(f"Retraining {symbol} (last trained {hours_since:.1f} hours ago)")
                self.train_models(symbol)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking retrain status: {e}")
            return False

# Integration function for existing trading bot
def get_ml_signal(symbol: str, current_data: pd.DataFrame, ml_pipeline: TradingMLPipeline = None) -> Dict[str, Any]:
    """
    Get ML trading signal for integration with existing trading bot
    
    Returns:
        Dict with 'signal', 'confidence', 'status'
    """
    
    if ml_pipeline is None:
        ml_pipeline = TradingMLPipeline()
    
    # Ensure model is trained and up to date
    ml_pipeline.retrain_if_needed(symbol, hours_threshold=24)
    
    # Get prediction
    result = ml_pipeline.predict(symbol, current_data)
    
    # Convert to trading signal format
    signal = 'BUY' if result['prediction'] == 1 else 'HOLD'
    
    return {
        'signal': signal,
        'confidence': result['confidence'],
        'status': result['status'],
        'ml_prediction': result['prediction']
    }

if __name__ == "__main__":
    # Test the pipeline
    pipeline = TradingMLPipeline()
    
    # Test symbols
    test_symbols = ['BTC/USDT', 'ETH/USDT']
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        
        # Train models
        results = pipeline.train_models(symbol)
        
        if results:
            print(f"Successfully trained {len(results)} models for {symbol}")
            
            # Show recent predictions
            predictions = pipeline.get_recent_predictions(symbol, hours=1)
            print(f"Recent predictions: {len(predictions)}")
        else:
            print(f"No models trained for {symbol} - insufficient data")