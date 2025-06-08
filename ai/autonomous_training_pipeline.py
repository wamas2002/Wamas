"""
Autonomous ML Training Pipeline
Comprehensive model training with auto-retraining every 24h
Includes GradientBoosting, XGBoost, CatBoost, LSTM+Attention, RandomForest, LightGBM, Prophet
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
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
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data.feature_engineer import AdvancedFeatureEngineer
from data.binance_data_collector import BinanceDataCollector
from data.news_sentiment_collector import NewsSentimentCollector

class AutonomousTrainingPipeline:
    """Comprehensive autonomous training pipeline for crypto trading"""
    
    def __init__(self, db_path: str = "data/trading_data.db", models_path: str = "models/"):
        self.db_path = db_path
        self.models_path = models_path
        self.feature_engineer = AdvancedFeatureEngineer(db_path)
        
        # Ensure models directory exists
        import os
        os.makedirs(models_path, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'catboost': {
                'model': cb.CatBoostClassifier,
                'params': {
                    'iterations': [100, 200],
                    'depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            }
        }
        
        # Best models storage
        self.best_models = {}
        self.model_performance = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def collect_fresh_data(self, symbols: List[str], days_back: int = 30):
        """Collect fresh data from multiple sources"""
        
        self.logger.info("Collecting fresh market data...")
        
        # Initialize collectors
        binance_collector = BinanceDataCollector(self.db_path)
        news_collector = NewsSentimentCollector(self.db_path)
        
        # Collect OHLCV data
        binance_collector.collect_multiple_symbols(symbols, '1m', days_back)
        
        # Collect news sentiment
        self.logger.info("Collecting news sentiment data...")
        news_items = news_collector.collect_all_news()
        if news_items:
            news_collector.save_news_to_database(news_items)
            
            # Aggregate sentiment for each symbol
            for symbol in symbols:
                symbol_clean = symbol.replace('/', '').replace('USDT', '').replace('USD', '')
                news_collector.aggregate_sentiment_by_minute(symbol_clean)
    
    def prepare_dataset(self, symbol: str, min_samples: int = 1000) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare comprehensive dataset with all features"""
        
        self.logger.info(f"Preparing dataset for {symbol}...")
        
        try:
            # Get feature-engineered data
            df, feature_cols = self.feature_engineer.prepare_training_data(
                symbol, '1m', 'binary', prediction_horizon=5  # Predict 5 minutes ahead
            )
            
            if len(df) < min_samples:
                raise ValueError(f"Insufficient data: {len(df)} samples (minimum {min_samples})")
            
            # Remove rows with too many NaN values
            df = df.dropna(thresh=len(feature_cols) * 0.8)  # Keep rows with at least 80% features
            
            # Fill remaining NaN values
            df[feature_cols] = df[feature_cols].fillna(method='forward').fillna(0)
            
            self.logger.info(f"Dataset prepared: {len(df)} samples, {len(feature_cols)} features")
            return df, feature_cols
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset for {symbol}: {e}")
            raise
    
    def train_model(self, model_name: str, X_train: np.ndarray, X_val: np.ndarray, 
                   y_train: np.ndarray, y_val: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train individual model with hyperparameter optimization"""
        
        self.logger.info(f"Training {model_name}...")
        
        try:
            config = self.model_configs[model_name]
            model_class = config['model']
            param_grid = config['params']
            
            # Handle CatBoost verbosity
            if model_name == 'catboost':
                base_params = {'verbose': False, 'random_state': 42}
            else:
                base_params = {'random_state': 42}
            
            # Grid search with cross-validation
            model = model_class(**base_params)
            
            # Simplified grid search for faster training
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=3, 
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate on validation set
            y_pred = best_model.predict(X_val)
            y_pred_proba = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(feature_names, best_model.feature_importances_))
            
            results = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'feature_importance': feature_importance,
                'training_time': datetime.now().isoformat()
            }
            
            self.logger.info(f"{model_name} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training {model_name}: {e}")
            return None
    
    def train_lstm_attention(self, X_train: np.ndarray, X_val: np.ndarray, 
                           y_train: np.ndarray, y_val: np.ndarray, 
                           sequence_length: int = 60) -> Dict[str, Any]:
        """Train LSTM model with attention mechanism"""
        
        try:
            # Import tensorflow here to avoid import issues
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
            
            self.logger.info("Training LSTM with attention...")
            
            # Reshape data for LSTM (samples, timesteps, features)
            def create_sequences(X, y, seq_length):
                X_seq, y_seq = [], []
                for i in range(seq_length, len(X)):
                    X_seq.append(X[i-seq_length:i])
                    y_seq.append(y[i])
                return np.array(X_seq), np.array(y_seq)
            
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
            X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
            
            if len(X_train_seq) < 100:  # Need minimum samples for LSTM
                self.logger.warning("Insufficient data for LSTM training")
                return None
            
            # Build LSTM model with attention
            input_layer = Input(shape=(sequence_length, X_train.shape[1]))
            
            # LSTM layers
            lstm1 = LSTM(50, return_sequences=True, dropout=0.2)(input_layer)
            lstm2 = LSTM(50, return_sequences=True, dropout=0.2)(lstm1)
            
            # Attention mechanism
            attention = Attention()([lstm2, lstm2])
            pooling = GlobalAveragePooling1D()(attention)
            
            # Dense layers
            dense1 = Dense(25, activation='relu')(pooling)
            dropout1 = Dropout(0.2)(dense1)
            output = Dense(1, activation='sigmoid')(dropout1)
            
            model = Model(inputs=input_layer, outputs=output)
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
            
            # Train model
            early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
            
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            y_pred_proba = model.predict(X_val_seq)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            f1 = f1_score(y_val_seq, y_pred)
            precision = precision_score(y_val_seq, y_pred)
            recall = recall_score(y_val_seq, y_pred)
            
            results = {
                'model': model,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba.flatten(),
                'training_history': history.history,
                'training_time': datetime.now().isoformat()
            }
            
            self.logger.info(f"LSTM+Attention - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            return results
            
        except ImportError:
            self.logger.warning("TensorFlow not available for LSTM training")
            return None
        except Exception as e:
            self.logger.error(f"Error training LSTM: {e}")
            return None
    
    def train_prophet_baseline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet model for baseline trend prediction"""
        
        try:
            from prophet import Prophet
            
            self.logger.info("Training Prophet baseline model...")
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df['close']
            })
            
            # Create and train model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_df)
            
            # Generate future predictions
            future = model.make_future_dataframe(periods=60, freq='min')  # 1 hour ahead
            forecast = model.predict(future)
            
            # Calculate trend-based signals
            recent_forecast = forecast.tail(100)
            trend_change = recent_forecast['trend'].pct_change().tail(20).mean()
            
            # Simple trend-based classification
            y_pred = (trend_change > 0.001).astype(int)  # Positive trend = buy signal
            
            results = {
                'model': model,
                'forecast': forecast,
                'trend_change': trend_change,
                'signal': y_pred,
                'training_time': datetime.now().isoformat()
            }
            
            self.logger.info(f"Prophet baseline completed - Trend change: {trend_change:.6f}")
            
            return results
            
        except ImportError:
            self.logger.warning("Prophet not available")
            return None
        except Exception as e:
            self.logger.error(f"Error training Prophet: {e}")
            return None
    
    def train_all_models(self, symbol: str) -> Dict[str, Any]:
        """Train all models and return results"""
        
        self.logger.info(f"Starting comprehensive training for {symbol}")
        
        # Prepare dataset
        df, feature_cols = self.prepare_dataset(symbol)
        
        # Prepare features and targets
        X = df[feature_cols].values
        y = df['target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # Train traditional ML models
        for model_name in self.model_configs.keys():
            try:
                model_results = self.train_model(
                    model_name, X_train, X_val, y_train, y_val, feature_cols
                )
                if model_results:
                    results[model_name] = model_results
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
        
        # Train LSTM with attention
        lstm_results = self.train_lstm_attention(X_train, X_val, y_train, y_val)
        if lstm_results:
            results['lstm_attention'] = lstm_results
        
        # Train Prophet baseline
        prophet_results = self.train_prophet_baseline(df)
        if prophet_results:
            results['prophet'] = prophet_results
        
        # Add metadata
        results['symbol'] = symbol
        results['training_date'] = datetime.now().isoformat()
        results['feature_scaler'] = scaler
        results['feature_columns'] = feature_cols
        results['dataset_size'] = len(df)
        
        return results
    
    def select_best_model(self, results: Dict[str, Any]) -> Tuple[str, Any]:
        """Select best model based on F1 score"""
        
        best_model_name = None
        best_f1 = 0
        best_model = None
        
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'f1_score' in model_results:
                f1 = model_results['f1_score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = model_name
                    best_model = model_results
        
        self.logger.info(f"Best model: {best_model_name} (F1: {best_f1:.4f})")
        
        return best_model_name, best_model
    
    def save_models(self, symbol: str, results: Dict[str, Any]):
        """Save trained models to disk"""
        
        try:
            symbol_clean = symbol.replace('/', '_')
            
            for model_name, model_results in results.items():
                if isinstance(model_results, dict) and 'model' in model_results:
                    model_path = f"{self.models_path}{symbol_clean}_{model_name}.joblib"
                    
                    # Save different model types appropriately
                    if model_name == 'lstm_attention':
                        # Save TensorFlow model
                        tf_model_path = f"{self.models_path}{symbol_clean}_{model_name}.h5"
                        model_results['model'].save(tf_model_path)
                        self.logger.info(f"Saved LSTM model to {tf_model_path}")
                    else:
                        # Save sklearn/other models
                        joblib.dump(model_results, model_path)
                        self.logger.info(f"Saved {model_name} to {model_path}")
            
            # Save best model info
            best_model_name, best_model = self.select_best_model(results)
            if best_model:
                best_model_path = f"{self.models_path}{symbol_clean}_best_model.joblib"
                joblib.dump({
                    'model_name': best_model_name,
                    'model_results': best_model,
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }, best_model_path)
                
                self.logger.info(f"Saved best model ({best_model_name}) to {best_model_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def store_performance_metrics(self, symbol: str, results: Dict[str, Any]):
        """Store model performance metrics in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    f1_score REAL,
                    precision_score REAL,
                    recall_score REAL,
                    training_date TEXT NOT NULL,
                    dataset_size INTEGER,
                    is_best_model INTEGER DEFAULT 0
                )
            ''')
            
            best_model_name, _ = self.select_best_model(results)
            
            for model_name, model_results in results.items():
                if isinstance(model_results, dict) and 'f1_score' in model_results:
                    cursor.execute('''
                        INSERT INTO model_performance 
                        (symbol, model_name, f1_score, precision_score, recall_score, 
                         training_date, dataset_size, is_best_model)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        model_name,
                        model_results['f1_score'],
                        model_results['precision'],
                        model_results['recall'],
                        model_results['training_time'],
                        results.get('dataset_size', 0),
                        1 if model_name == best_model_name else 0
                    ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored performance metrics for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error storing performance metrics: {e}")
    
    def run_full_training_cycle(self, symbols: List[str]):
        """Run complete training cycle for all symbols"""
        
        self.logger.info("Starting full training cycle...")
        
        # Collect fresh data
        self.collect_fresh_data(symbols, days_back=30)
        
        # Train models for each symbol
        for symbol in symbols:
            try:
                self.logger.info(f"Training models for {symbol}")
                
                # Train all models
                results = self.train_all_models(symbol)
                
                if results:
                    # Save models
                    self.save_models(symbol, results)
                    
                    # Store performance metrics
                    self.store_performance_metrics(symbol, results)
                    
                    # Store in memory for easy access
                    self.best_models[symbol] = results
                
            except Exception as e:
                self.logger.error(f"Failed to train models for {symbol}: {e}")
                continue
        
        self.logger.info("Full training cycle completed")
    
    def should_retrain(self, symbol: str, hours_threshold: int = 24) -> bool:
        """Check if model should be retrained based on age"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT training_date FROM model_performance 
                WHERE symbol = ? AND is_best_model = 1
                ORDER BY training_date DESC LIMIT 1
            ''', (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return True  # No previous training found
            
            last_training = datetime.fromisoformat(result[0])
            hours_since = (datetime.now() - last_training).total_seconds() / 3600
            
            return hours_since >= hours_threshold
            
        except Exception as e:
            self.logger.error(f"Error checking retrain status: {e}")
            return True  # Default to retrain on error
    
    def autonomous_training_loop(self, symbols: List[str], check_interval_hours: int = 1):
        """Run autonomous training loop with periodic retraining"""
        
        self.logger.info("Starting autonomous training loop...")
        
        while True:
            try:
                # Check which symbols need retraining
                symbols_to_retrain = [s for s in symbols if self.should_retrain(s)]
                
                if symbols_to_retrain:
                    self.logger.info(f"Retraining models for: {symbols_to_retrain}")
                    self.run_full_training_cycle(symbols_to_retrain)
                else:
                    self.logger.info("All models are up to date")
                
                # Wait before next check
                import time
                time.sleep(check_interval_hours * 3600)
                
            except KeyboardInterrupt:
                self.logger.info("Training loop stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                import time
                time.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    # Initialize training pipeline
    pipeline = AutonomousTrainingPipeline()
    
    # Major trading pairs to train
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
    
    # Run single training cycle for testing
    pipeline.run_full_training_cycle(symbols)
    
    print("Training completed! Check models/ directory for saved models.")