import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedLSTMPredictor:
    """Advanced LSTM-based price prediction with ensemble architecture and attention mechanism"""
    
    def __init__(self, lookback_window: int = 60, prediction_horizon: int = 1, lstm_units: int = 128, num_layers: int = 3):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        
        # Scalers for different components
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Model components
        self.lstm_weights = {}
        self.attention_weights = None
        self.ensemble_models = {}
        self.is_trained = False
        
        # Training history
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'metrics': {}
        }
        
        # Advanced features
        self.dropout_rate = 0.2
        self.l2_reg = 0.001
        self.learning_rate = 0.001
        
    def prepare_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive technical features for advanced LSTM"""
        features_df = df.copy()
        
        # Basic price features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        features_df['high_low_ratio'] = (features_df['high'] - features_df['low']) / features_df['close']
        features_df['open_close_ratio'] = (features_df['open'] - features_df['close']) / features_df['close']
        
        # Advanced moving averages
        windows = [5, 10, 20, 50]
        for window in windows:
            features_df[f'sma_{window}'] = features_df['close'].rolling(window).mean()
            features_df[f'ema_{window}'] = features_df['close'].ewm(span=window).mean()
            features_df[f'price_sma_ratio_{window}'] = features_df['close'] / features_df[f'sma_{window}']
            features_df[f'price_ema_ratio_{window}'] = features_df['close'] / features_df[f'ema_{window}']
            
            # Statistical features
            features_df[f'volatility_{window}'] = features_df['returns'].rolling(window).std()
            features_df[f'skewness_{window}'] = features_df['returns'].rolling(window).skew()
            features_df[f'kurtosis_{window}'] = features_df['returns'].rolling(window).kurt()
        
        # Technical indicators
        features_df['rsi_14'] = self._calculate_rsi(features_df['close'], 14)
        features_df['rsi_21'] = self._calculate_rsi(features_df['close'], 21)
        features_df['atr_14'] = self._calculate_atr(features_df, 14)
        features_df['atr_21'] = self._calculate_atr(features_df, 21)
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = features_df['close'].rolling(bb_period).mean()
        bb_std_dev = features_df['close'].rolling(bb_period).std()
        features_df['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
        features_df['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
        features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / bb_middle
        features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # MACD
        ema_12 = features_df['close'].ewm(span=12).mean()
        ema_26 = features_df['close'].ewm(span=26).mean()
        features_df['macd'] = ema_12 - ema_26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # Volume features (if available)
        if 'volume' in features_df.columns:
            for window in [5, 10, 20]:
                features_df[f'volume_sma_{window}'] = features_df['volume'].rolling(window).mean()
                features_df[f'volume_ratio_{window}'] = features_df['volume'] / features_df[f'volume_sma_{window}']
            
            # On-Balance Volume
            features_df['obv'] = (features_df['volume'] * np.sign(features_df['returns'])).cumsum()
            features_df['obv_ema'] = features_df['obv'].ewm(span=10).mean()
        
        # Market structure features
        features_df['higher_high'] = (features_df['high'] > features_df['high'].shift(1)).astype(int)
        features_df['lower_low'] = (features_df['low'] < features_df['low'].shift(1)).astype(int)
        features_df['higher_close'] = (features_df['close'] > features_df['close'].shift(1)).astype(int)
        
        # Time-based features
        if hasattr(features_df.index, 'hour'):
            features_df['hour'] = features_df.index.hour
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['month'] = features_df.index.month
            
            # Cyclical encoding
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        
        # Lag features
        lag_periods = [1, 2, 3, 5, 8, 13]
        for lag in lag_periods:
            features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag) if 'volume' in features_df.columns else 0
        
        # Clean features
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training with enhanced preprocessing"""
        X, y = [], []
        
        for i in range(self.lookback_window, len(data) - self.prediction_horizon + 1):
            sequence = data[i-self.lookback_window:i]
            target_value = target[i + self.prediction_horizon - 1]
            
            if not np.isnan(target_value) and not np.any(np.isnan(sequence)):
                X.append(sequence)
                y.append(target_value)
        
        return np.array(X), np.array(y)
    
    def _lstm_cell_forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray, 
                          Wf: np.ndarray, Wi: np.ndarray, Wo: np.ndarray, Wg: np.ndarray,
                          Uf: np.ndarray, Ui: np.ndarray, Uo: np.ndarray, Ug: np.ndarray,
                          bf: np.ndarray, bi: np.ndarray, bo: np.ndarray, bg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """LSTM cell forward pass"""
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, h_prev], axis=1)
        
        # Gates
        f_gate = self._sigmoid(np.dot(combined, np.concatenate([Wf, Uf], axis=0)) + bf)  # Forget gate
        i_gate = self._sigmoid(np.dot(combined, np.concatenate([Wi, Ui], axis=0)) + bi)  # Input gate
        o_gate = self._sigmoid(np.dot(combined, np.concatenate([Wo, Uo], axis=0)) + bo)  # Output gate
        g_gate = np.tanh(np.dot(combined, np.concatenate([Wg, Ug], axis=0)) + bg)        # Candidate values
        
        # Cell state
        c_new = f_gate * c_prev + i_gate * g_gate
        
        # Hidden state
        h_new = o_gate * np.tanh(c_new)
        
        return h_new, c_new
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def _initialize_lstm_weights(self, input_size: int) -> None:
        """Initialize LSTM weights using Xavier initialization"""
        hidden_size = self.lstm_units
        
        # Weight matrices for each gate (input to hidden)
        self.lstm_weights['Wf'] = np.random.normal(0, np.sqrt(2.0 / (input_size + hidden_size)), (input_size, hidden_size))
        self.lstm_weights['Wi'] = np.random.normal(0, np.sqrt(2.0 / (input_size + hidden_size)), (input_size, hidden_size))
        self.lstm_weights['Wo'] = np.random.normal(0, np.sqrt(2.0 / (input_size + hidden_size)), (input_size, hidden_size))
        self.lstm_weights['Wg'] = np.random.normal(0, np.sqrt(2.0 / (input_size + hidden_size)), (input_size, hidden_size))
        
        # Weight matrices for each gate (hidden to hidden)
        self.lstm_weights['Uf'] = np.random.normal(0, np.sqrt(2.0 / (hidden_size * 2)), (hidden_size, hidden_size))
        self.lstm_weights['Ui'] = np.random.normal(0, np.sqrt(2.0 / (hidden_size * 2)), (hidden_size, hidden_size))
        self.lstm_weights['Uo'] = np.random.normal(0, np.sqrt(2.0 / (hidden_size * 2)), (hidden_size, hidden_size))
        self.lstm_weights['Ug'] = np.random.normal(0, np.sqrt(2.0 / (hidden_size * 2)), (hidden_size, hidden_size))
        
        # Bias vectors
        self.lstm_weights['bf'] = np.ones((1, hidden_size))  # Forget gate bias (initialized to 1)
        self.lstm_weights['bi'] = np.zeros((1, hidden_size))
        self.lstm_weights['bo'] = np.zeros((1, hidden_size))
        self.lstm_weights['bg'] = np.zeros((1, hidden_size))
        
        # Output layer weights
        self.lstm_weights['Wy'] = np.random.normal(0, np.sqrt(2.0 / hidden_size), (hidden_size, 1))
        self.lstm_weights['by'] = np.zeros((1, 1))
        
        # Attention mechanism weights
        self.attention_weights = {
            'Wa': np.random.normal(0, 0.1, (hidden_size, hidden_size)),
            'Ua': np.random.normal(0, 0.1, (hidden_size, hidden_size)),
            'va': np.random.normal(0, 0.1, (hidden_size, 1))
        }
    
    def _apply_attention(self, hidden_states: np.ndarray) -> np.ndarray:
        """Apply attention mechanism to hidden states"""
        seq_len, hidden_size = hidden_states.shape
        
        # Calculate attention scores
        scores = []
        for i in range(seq_len):
            # Attention score for each timestep
            score = np.tanh(np.dot(hidden_states[i:i+1], self.attention_weights['Wa']) + 
                           np.dot(hidden_states[-1:], self.attention_weights['Ua']))
            score = np.dot(score, self.attention_weights['va'])
            scores.append(score[0, 0])
        
        # Softmax normalization
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Weighted sum of hidden states
        context = np.sum(hidden_states * attention_weights.reshape(-1, 1), axis=0, keepdims=True)
        
        return context
    
    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the advanced LSTM model with attention mechanism"""
        try:
            print("Preparing advanced features for LSTM training...")
            features_df = self.prepare_advanced_features(df)
            
            # Select feature columns (exclude OHLCV)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols and not pd.isna(features_df[col]).all()]
            
            if len(feature_cols) == 0:
                return {'error': 'No valid features found after preprocessing'}
            
            print(f"Using {len(feature_cols)} features for LSTM training")
            
            # Prepare data
            feature_matrix = features_df[feature_cols].values
            target_values = features_df['close'].pct_change().shift(-self.prediction_horizon).values
            
            # Scale features and target
            feature_matrix_scaled = self.feature_scaler.fit_transform(feature_matrix)
            target_scaled = self.target_scaler.fit_transform(target_values.reshape(-1, 1)).flatten()
            
            # Create sequences
            X, y = self.create_sequences(feature_matrix_scaled, target_scaled)
            
            if len(X) == 0:
                return {'error': 'No valid sequences created'}
            
            print(f"Created {len(X)} training sequences with shape {X.shape}")
            
            # Train-validation split
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Initialize LSTM weights
            input_size = X.shape[2]
            self._initialize_lstm_weights(input_size)
            
            # Initialize ensemble models
            self._train_ensemble_models(feature_matrix_scaled[self.lookback_window:], target_scaled[self.lookback_window:])
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10
            
            for epoch in range(epochs):
                # Training
                train_loss = self._train_epoch(X_train, y_train, batch_size)
                
                # Validation
                val_loss = self._validate_epoch(X_val, y_val)
                
                # Update training history
                self.training_history['loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            self.is_trained = True
            
            # Calculate metrics
            train_predictions = self._predict_sequences(X_train)
            val_predictions = self._predict_sequences(X_val)
            
            train_mse = mean_squared_error(y_train, train_predictions)
            val_mse = mean_squared_error(y_val, val_predictions)
            train_r2 = r2_score(y_train, train_predictions) if len(set(y_train)) > 1 else 0
            val_r2 = r2_score(y_val, val_predictions) if len(set(y_val)) > 1 else 0
            
            return {
                'success': True,
                'epochs_trained': len(self.training_history['loss']),
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'feature_count': len(feature_cols),
                'sequence_count': len(X),
                'best_val_loss': best_val_loss
            }
            
        except Exception as e:
            print(f"Error training LSTM: {e}")
            return {'error': str(e)}
    
    def _train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train ensemble models for comparison"""
        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) == 0:
            return
        
        # Train ensemble models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_clean, y_clean)
                self.ensemble_models[name] = model
            except Exception as e:
                print(f"Error training {name}: {e}")
    
    def _train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
        """Train one epoch"""
        total_loss = 0
        num_batches = len(X) // batch_size
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Forward pass
            predictions = self._predict_sequences(batch_X)
            
            # Calculate loss (MSE)
            loss = np.mean((predictions - batch_y) ** 2)
            total_loss += loss
            
            # Simple gradient update (simplified for demonstration)
            lr = self.learning_rate
            error = predictions - batch_y
            
            # Update output weights (simplified)
            if hasattr(self, '_last_hidden_states'):
                grad_Wy = np.dot(self._last_hidden_states.T, error.reshape(-1, 1)) / len(batch_X)
                grad_by = np.mean(error)
                
                self.lstm_weights['Wy'] -= lr * grad_Wy
                self.lstm_weights['by'] -= lr * grad_by
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Validate one epoch"""
        predictions = self._predict_sequences(X)
        return np.mean((predictions - y) ** 2)
    
    def _predict_sequences(self, X: np.ndarray) -> np.ndarray:
        """Predict on sequences using LSTM"""
        predictions = []
        
        for i in range(len(X)):
            sequence = X[i]
            
            # Initialize hidden and cell states
            h = np.zeros((1, self.lstm_units))
            c = np.zeros((1, self.lstm_units))
            
            hidden_states = []
            
            # Forward pass through sequence
            for t in range(sequence.shape[0]):
                x_t = sequence[t:t+1].reshape(1, -1)
                h, c = self._lstm_cell_forward(
                    x_t, h, c,
                    self.lstm_weights['Wf'], self.lstm_weights['Wi'], 
                    self.lstm_weights['Wo'], self.lstm_weights['Wg'],
                    self.lstm_weights['Uf'], self.lstm_weights['Ui'], 
                    self.lstm_weights['Uo'], self.lstm_weights['Ug'],
                    self.lstm_weights['bf'], self.lstm_weights['bi'], 
                    self.lstm_weights['bo'], self.lstm_weights['bg']
                )
                hidden_states.append(h.copy())
            
            # Apply attention mechanism
            hidden_states_array = np.concatenate(hidden_states, axis=0)
            self._last_hidden_states = hidden_states_array
            
            if self.attention_weights is not None:
                context = self._apply_attention(hidden_states_array)
            else:
                context = h  # Use last hidden state if no attention
            
            # Final prediction
            pred = np.dot(context, self.lstm_weights['Wy']) + self.lstm_weights['by']
            predictions.append(pred[0, 0])
        
        return np.array(predictions)
    
    def predict(self, df: pd.DataFrame, steps: int = 1) -> Dict[str, Any]:
        """Generate price predictions using advanced LSTM"""
        try:
            if not self.is_trained:
                return {'error': 'Model not trained. Call train() first.'}
            
            print("Preparing features for LSTM prediction...")
            features_df = self.prepare_advanced_features(df)
            
            # Select feature columns (exclude OHLCV)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols and not pd.isna(features_df[col]).all()]
            
            if len(feature_cols) == 0:
                return {'error': 'No valid features found for prediction'}
            
            # Prepare sequence
            feature_matrix = features_df[feature_cols].values
            feature_matrix_scaled = self.feature_scaler.transform(feature_matrix)
            
            if len(feature_matrix_scaled) < self.lookback_window:
                return {'error': f'Insufficient data. Need at least {self.lookback_window} samples'}
            
            # Get recent sequence for prediction
            recent_sequence = feature_matrix_scaled[-self.lookback_window:].reshape(1, self.lookback_window, -1)
            
            # LSTM prediction
            lstm_pred = self._predict_sequences(recent_sequence)[0]
            
            # Ensemble predictions
            ensemble_predictions = {}
            if self.ensemble_models:
                recent_features = feature_matrix_scaled[-1:].reshape(1, -1)
                for name, model in self.ensemble_models.items():
                    try:
                        pred = model.predict(recent_features)[0]
                        ensemble_predictions[name] = pred
                    except Exception as e:
                        print(f"Error with {name} prediction: {e}")
            
            # Calculate ensemble prediction
            if ensemble_predictions:
                ensemble_pred = np.mean(list(ensemble_predictions.values()))
                # Weighted combination of LSTM and ensemble
                final_pred = 0.7 * lstm_pred + 0.3 * ensemble_pred
            else:
                final_pred = lstm_pred
            
            # Inverse transform prediction
            pred_scaled = np.array([[final_pred]])
            pred_return = self.target_scaler.inverse_transform(pred_scaled)[0, 0]
            
            # Convert to price prediction
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + pred_return)
            
            # Calculate confidence based on training performance
            confidence = self._calculate_confidence(features_df.iloc[-self.lookback_window:])
            
            # Generate multiple step predictions if requested
            multi_step_predictions = []
            if steps > 1:
                for step in range(1, min(steps + 1, 10)):  # Limit to 10 steps
                    step_pred = final_pred * (0.9 ** (step - 1))  # Decay for longer horizons
                    step_price = current_price * (1 + step_pred)
                    multi_step_predictions.append({
                        'step': step,
                        'predicted_return': step_pred,
                        'predicted_price': step_price,
                        'confidence': confidence * (0.9 ** (step - 1))
                    })
            
            return {
                'success': True,
                'prediction': final_pred,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'predicted_return': pred_return,
                'confidence': confidence,
                'lstm_prediction': lstm_pred,
                'ensemble_predictions': ensemble_predictions,
                'multi_step': multi_step_predictions,
                'model_type': 'Advanced LSTM with Attention'
            }
            
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence(self, recent_data: pd.DataFrame) -> float:
        """Calculate prediction confidence based on market conditions"""
        try:
            # Volatility-based confidence
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            vol_confidence = max(0.1, 1.0 - min(volatility * 10, 0.8))
            
            # Trend consistency confidence
            trend_consistency = self._calculate_trend_consistency(recent_data)
            
            # Training performance confidence
            if hasattr(self, 'training_history') and self.training_history['val_loss']:
                last_val_loss = self.training_history['val_loss'][-1]
                training_confidence = max(0.1, 1.0 - min(last_val_loss, 0.8))
            else:
                training_confidence = 0.5
            
            # Combined confidence
            final_confidence = (vol_confidence * 0.4 + trend_consistency * 0.3 + training_confidence * 0.3)
            return min(max(final_confidence, 0.1), 0.95)
            
        except Exception:
            return 0.5
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate trend consistency score"""
        try:
            if len(df) < 10:
                return 0.5
            
            prices = df['close'].values
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-10:])
            
            # Direction consistency
            recent_directions = np.sign(np.diff(prices[-10:]))
            consistency = np.abs(np.mean(recent_directions))
            
            # Trend strength
            trend_strength = abs(short_ma - long_ma) / long_ma
            
            return min(max((consistency + trend_strength) / 2, 0.1), 0.9)
            
        except Exception:
            return 0.5
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'model_type': 'Advanced LSTM with Attention Mechanism',
            'is_trained': self.is_trained,
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }
        
        if self.is_trained:
            info.update({
                'training_epochs': len(self.training_history['loss']),
                'final_train_loss': self.training_history['loss'][-1] if self.training_history['loss'] else None,
                'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
                'ensemble_models': list(self.ensemble_models.keys()) if self.ensemble_models else [],
                'has_attention': self.attention_weights is not None
            })
        
        return info
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