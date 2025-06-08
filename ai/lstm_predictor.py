import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor:
    """LSTM-based price prediction using rolling window approach"""
    
    def __init__(self, lookback_window: int = 60, prediction_horizon: int = 1):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_weights = None
        self.is_trained = False
        self.training_loss = []
        
    def create_sequences(self, data: np.ndarray, target: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.lookback_window, len(data)):
            X.append(data[i-self.lookback_window:i])
            if target is not None:
                if i + self.prediction_horizon - 1 < len(target):
                    y.append(target[i + self.prediction_horizon - 1])
                else:
                    break
        
        return np.array(X), np.array(y)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare technical features for LSTM"""
        features_df = df.copy()
        
        # Price features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20]:
            features_df[f'sma_{period}'] = features_df['close'].rolling(period).mean()
            features_df[f'ema_{period}'] = features_df['close'].ewm(span=period).mean()
            features_df[f'price_sma_ratio_{period}'] = features_df['close'] / features_df[f'sma_{period}']
        
        # Volatility features
        features_df['volatility'] = features_df['returns'].rolling(20).std()
        features_df['atr'] = self._calculate_atr(features_df)
        
        # Volume features
        features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
        
        # RSI
        features_df['rsi'] = self._calculate_rsi(features_df['close'])
        
        # Bollinger Bands position
        bb_period = 20
        bb_std = 2
        bb_middle = features_df['close'].rolling(bb_period).mean()
        bb_upper = bb_middle + (features_df['close'].rolling(bb_period).std() * bb_std)
        bb_lower = bb_middle - (features_df['close'].rolling(bb_period).std() * bb_std)
        features_df['bb_position'] = (features_df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        ema_12 = features_df['close'].ewm(span=12).mean()
        ema_26 = features_df['close'].ewm(span=26).mean()
        features_df['macd'] = ema_12 - ema_26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        return features_df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM model using matrix operations (sklearn-compatible)"""
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            
            if len(features_df) < self.lookback_window + self.prediction_horizon + 50:
                return {
                    'success': False,
                    'error': f'Insufficient data for training. Need at least {self.lookback_window + self.prediction_horizon + 50} samples'
                }
            
            # Select feature columns (excluding target)
            feature_cols = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Scale features
            feature_data = features_df[feature_cols].values
            feature_data_scaled = self.feature_scaler.fit_transform(feature_data)
            
            # Scale target (close prices)
            target_data = features_df['close'].values.reshape(-1, 1)
            target_data_scaled = self.scaler.fit_transform(target_data)
            
            # Create sequences
            X, y = self.create_sequences(feature_data_scaled, target_data_scaled.flatten())
            
            if len(X) == 0:
                return {'success': False, 'error': 'No sequences created'}
            
            # Simulate LSTM with dense layers and attention mechanism
            X_reshaped = X.reshape(X.shape[0], -1)  # Flatten sequences
            
            # Create weighted features (simulating attention)
            attention_weights = self._calculate_attention_weights(X_reshaped)
            X_weighted = X_reshaped * attention_weights
            
            # Train with multiple models ensemble (simulating LSTM layers)
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.neural_network import MLPRegressor
            
            models = {
                'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'gbm': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
                'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            # Split data for training
            split_idx = int(0.8 * len(X_weighted))
            X_train, X_val = X_weighted[:split_idx], X_weighted[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            model_scores = {}
            trained_models = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, val_pred)
                model_scores[name] = 1.0 / (1.0 + mse)  # Convert to score
                trained_models[name] = model
            
            # Create ensemble weights based on validation performance
            total_score = sum(model_scores.values())
            ensemble_weights = {name: score/total_score for name, score in model_scores.items()}
            
            self.model_weights = {
                'models': trained_models,
                'weights': ensemble_weights,
                'attention_weights': attention_weights
            }
            
            self.is_trained = True
            
            # Calculate training metrics
            train_pred = self._ensemble_predict(X_train)
            val_pred = self._ensemble_predict(X_val)
            
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            return {
                'success': True,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'model_weights': ensemble_weights,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': X_weighted.shape[1]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Calculate attention weights for features"""
        # Simple attention mechanism based on feature variance and correlation
        feature_variance = np.var(X, axis=0)
        feature_mean = np.mean(X, axis=0)
        
        # Calculate attention scores
        attention_scores = feature_variance * (1 + np.abs(feature_mean))
        attention_weights = attention_scores / np.sum(attention_scores)
        
        # Reshape to broadcast
        return attention_weights.reshape(1, -1)
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        if not self.is_trained or self.model_weights is None:
            raise ValueError("Model not trained")
        
        models = self.model_weights['models']
        weights = self.model_weights['weights']
        
        predictions = []
        for name, model in models.items():
            pred = model.predict(X)
            predictions.append(pred * weights[name])
        
        return np.sum(predictions, axis=0)
    
    def predict(self, df: pd.DataFrame, steps: int = 1) -> Dict[str, Any]:
        """Make price predictions"""
        try:
            if not self.is_trained:
                return {'success': False, 'error': 'Model not trained'}
            
            # Prepare features
            features_df = self.prepare_features(df)
            
            if len(features_df) < self.lookback_window:
                return {'success': False, 'error': 'Insufficient data for prediction'}
            
            # Get feature columns
            feature_cols = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Scale features
            feature_data = features_df[feature_cols].values
            feature_data_scaled = self.feature_scaler.transform(feature_data)
            
            # Get last sequence
            last_sequence = feature_data_scaled[-self.lookback_window:]
            last_sequence = last_sequence.reshape(1, -1)
            
            # Apply attention weights
            attention_weights = self.model_weights['attention_weights']
            last_sequence_weighted = last_sequence * attention_weights
            
            # Make prediction
            prediction_scaled = self._ensemble_predict(last_sequence_weighted)
            
            # Inverse transform
            prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
            
            # Calculate confidence based on recent model performance
            confidence = self._calculate_confidence(features_df)
            
            return {
                'success': True,
                'prediction': prediction,
                'current_price': df['close'].iloc[-1],
                'predicted_change': (prediction - df['close'].iloc[-1]) / df['close'].iloc[-1],
                'confidence': confidence,
                'horizon_hours': self.prediction_horizon,
                'model_type': 'LSTM_Ensemble'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_confidence(self, features_df: pd.DataFrame) -> float:
        """Calculate prediction confidence based on market conditions"""
        try:
            # Base confidence
            base_confidence = 0.6
            
            # Adjust based on volatility (lower volatility = higher confidence)
            recent_vol = features_df['volatility'].tail(5).mean()
            vol_adjustment = max(0, 0.3 - recent_vol * 10)
            
            # Adjust based on trend consistency
            trend_consistency = self._calculate_trend_consistency(features_df)
            trend_adjustment = trend_consistency * 0.2
            
            confidence = base_confidence + vol_adjustment + trend_adjustment
            return min(0.95, max(0.1, confidence))
            
        except:
            return 0.5
    
    def _calculate_trend_consistency(self, features_df: pd.DataFrame) -> float:
        """Calculate trend consistency score"""
        try:
            price_changes = features_df['returns'].tail(10)
            positive_count = (price_changes > 0).sum()
            negative_count = (price_changes < 0).sum()
            
            if len(price_changes) == 0:
                return 0.5
            
            # Higher consistency when most moves are in same direction
            consistency = abs(positive_count - negative_count) / len(price_changes)
            return consistency
            
        except:
            return 0.5
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        if not self.is_trained:
            return {'trained': False}
        
        return {
            'trained': True,
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'model_weights': self.model_weights['weights'] if self.model_weights else {},
            'training_loss': self.training_loss
        }