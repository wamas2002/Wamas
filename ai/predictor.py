import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class AIPredictor:
    """AI predictor combining multiple models for price prediction"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.models = {}
        self.is_trained = False
        self.feature_columns = []
        self.prediction_horizon = 24  # hours
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        try:
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Need: {required_cols}")
            
            features_df = df.copy()
            
            # Price-based features
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['high_low_ratio'] = features_df['high'] / features_df['low']
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                features_df[f'sma_{window}'] = features_df['close'].rolling(window).mean()
                features_df[f'price_to_sma_{window}'] = features_df['close'] / features_df[f'sma_{window}']
            
            # Volatility features
            features_df['volatility_5'] = features_df['price_change'].rolling(5).std()
            features_df['volatility_20'] = features_df['price_change'].rolling(20).std()
            
            # Momentum features
            features_df['momentum_5'] = features_df['close'] / features_df['close'].shift(5) - 1
            features_df['momentum_10'] = features_df['close'] / features_df['close'].shift(10) - 1
            
            # Volume features
            features_df['volume_sma_10'] = features_df['volume'].rolling(10).mean()
            features_df['volume_std_10'] = features_df['volume'].rolling(10).std()
            
            # Target variable (future return)
            features_df['target'] = features_df['close'].shift(-1) / features_df['close'] - 1
            
            # Drop rows with NaN values
            features_df = features_df.dropna()
            
            # Store feature columns
            self.feature_columns = [col for col in features_df.columns 
                                  if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
            
            return features_df
            
        except Exception as e:
            print(f"Error in prepare_features: {e}")
            return pd.DataFrame()
    
    def train_lstm_model(self, data: np.ndarray, lookback: int = 60) -> dict:
        """Train a simple LSTM-like model using sklearn"""
        try:
            # Create sequences for time series prediction
            X, y = [], []
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i])
                y.append(data[i])
            
            X, y = np.array(X), np.array(y)
            
            # Reshape for sklearn (flatten sequences)
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
            
            # Train Random Forest as LSTM substitute
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            # Final training on all data
            model.fit(X, y)
            
            return {
                'model': model,
                'lookback': lookback,
                'validation_scores': scores,
                'mean_score': np.mean(scores)
            }
            
        except Exception as e:
            print(f"Error in train_lstm_model: {e}")
            return {}
    
    def train_prophet_model(self, df: pd.DataFrame) -> dict:
        """Train a Prophet-like model using trend decomposition"""
        try:
            # Simple trend and seasonality extraction
            close_prices = df['close'].values
            
            # Trend (using moving average)
            trend = pd.Series(close_prices).rolling(20, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            
            # Seasonality (using FFT to find dominant frequencies)
            detrended = close_prices - trend
            fft = np.fft.fft(detrended)
            frequencies = np.fft.fftfreq(len(detrended))
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            dominant_freq = frequencies[dominant_freq_idx]
            
            # Create seasonal component
            time_index = np.arange(len(close_prices))
            seasonal = np.sin(2 * np.pi * dominant_freq * time_index)
            
            return {
                'trend': trend,
                'seasonal_freq': dominant_freq,
                'seasonal_amplitude': np.std(detrended),
                'residual_std': np.std(detrended - seasonal * np.std(detrended))
            }
            
        except Exception as e:
            print(f"Error in train_prophet_model: {e}")
            return {}
    
    def train_transformer_model(self, features_df: pd.DataFrame) -> dict:
        """Train a Transformer-like model using attention mechanism simulation"""
        try:
            if features_df.empty or len(self.feature_columns) == 0:
                return {}
            
            X = features_df[self.feature_columns].values
            y = features_df['target'].values
            
            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]
            
            if len(X) == 0:
                return {}
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Simulate attention mechanism using feature importance
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            feature_importance = np.zeros(len(self.feature_columns))
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
                feature_importance += model.feature_importances_
            
            # Final training
            model.fit(X_scaled, y)
            feature_importance /= len(scores)
            
            return {
                'model': model,
                'scaler': self.scaler,
                'feature_importance': feature_importance,
                'feature_names': self.feature_columns,
                'validation_scores': scores,
                'mean_score': np.mean(scores)
            }
            
        except Exception as e:
            print(f"Error in train_transformer_model: {e}")
            return {}
    
    def train_models(self, df: pd.DataFrame) -> dict:
        """Train all AI models"""
        try:
            if df.empty:
                return {"error": "Empty dataframe provided"}
            
            print("Preparing features...")
            features_df = self.prepare_features(df)
            
            if features_df.empty:
                return {"error": "Failed to prepare features"}
            
            results = {}
            
            # Train LSTM model
            print("Training LSTM model...")
            lstm_data = features_df[['close']].values
            lstm_scaled = self.scaler.fit_transform(lstm_data)
            results['lstm'] = self.train_lstm_model(lstm_scaled)
            
            # Train Prophet model
            print("Training Prophet model...")
            results['prophet'] = self.train_prophet_model(features_df)
            
            # Train Transformer model
            print("Training Transformer model...")
            results['transformer'] = self.train_transformer_model(features_df)
            
            self.models = results
            self.is_trained = True
            
            return results
            
        except Exception as e:
            print(f"Error in train_models: {e}")
            return {"error": str(e)}
    
    def predict(self, df: pd.DataFrame) -> dict:
        """Generate predictions from all models"""
        try:
            if not self.is_trained or not self.models:
                return {"error": "Models not trained"}
            
            features_df = self.prepare_features(df)
            if features_df.empty:
                return {"error": "Failed to prepare features"}
            
            predictions = {}
            
            # LSTM prediction
            if 'lstm' in self.models and self.models['lstm']:
                try:
                    lstm_model = self.models['lstm']['model']
                    lookback = self.models['lstm']['lookback']
                    
                    if len(features_df) >= lookback:
                        close_data = features_df[['close']].values
                        close_scaled = self.scaler.transform(close_data)
                        
                        # Get last sequence
                        last_sequence = close_scaled[-lookback:].reshape(1, -1)
                        lstm_pred = lstm_model.predict(last_sequence)[0]
                        
                        predictions['lstm'] = lstm_pred
                    else:
                        predictions['lstm'] = 0.0
                        
                except Exception as e:
                    print(f"LSTM prediction error: {e}")
                    predictions['lstm'] = 0.0
            
            # Prophet prediction
            if 'prophet' in self.models and self.models['prophet']:
                try:
                    prophet_model = self.models['prophet']
                    current_price = features_df['close'].iloc[-1]
                    
                    # Simple trend continuation
                    trend_pred = prophet_model['trend'].iloc[-1] if len(prophet_model['trend']) > 0 else current_price
                    seasonal_pred = prophet_model['seasonal_amplitude'] * np.sin(2 * np.pi * prophet_model['seasonal_freq'])
                    
                    prophet_pred = (trend_pred + seasonal_pred) / current_price - 1
                    predictions['prophet'] = prophet_pred
                    
                except Exception as e:
                    print(f"Prophet prediction error: {e}")
                    predictions['prophet'] = 0.0
            
            # Transformer prediction
            if 'transformer' in self.models and self.models['transformer']:
                try:
                    transformer_model = self.models['transformer']['model']
                    scaler = self.models['transformer']['scaler']
                    
                    if len(features_df) > 0 and len(self.feature_columns) > 0:
                        latest_features = features_df[self.feature_columns].iloc[-1:].values
                        
                        if not np.isnan(latest_features).any():
                            features_scaled = scaler.transform(latest_features)
                            transformer_pred = transformer_model.predict(features_scaled)[0]
                            predictions['transformer'] = transformer_pred
                        else:
                            predictions['transformer'] = 0.0
                    else:
                        predictions['transformer'] = 0.0
                        
                except Exception as e:
                    print(f"Transformer prediction error: {e}")
                    predictions['transformer'] = 0.0
            
            # Ensemble prediction with confidence
            if predictions:
                weights = {'lstm': 0.4, 'prophet': 0.3, 'transformer': 0.3}
                ensemble_pred = sum(predictions.get(model, 0) * weight 
                                 for model, weight in weights.items())
                
                # Calculate confidence based on agreement between models
                pred_values = list(predictions.values())
                confidence = 1.0 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8))
                confidence = max(0.0, min(1.0, confidence))
                
                predictions['ensemble'] = ensemble_pred
                predictions['confidence'] = confidence
            
            return predictions
            
        except Exception as e:
            print(f"Error in predict: {e}")
            return {"error": str(e)}
    
    def get_model_performance(self) -> dict:
        """Get performance metrics for all models"""
        if not self.is_trained:
            return {}
        
        performance = {}
        
        for model_name, model_data in self.models.items():
            if isinstance(model_data, dict) and 'validation_scores' in model_data:
                scores = model_data['validation_scores']
                performance[model_name] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores)
                }
        
        return performance
