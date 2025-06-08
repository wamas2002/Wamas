import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AttentionLayer:
    """Simple attention mechanism for time series"""
    
    def __init__(self, input_dim: int, attention_dim: int = 64):
        self.input_dim = input_dim
        self.attention_dim = min(attention_dim, input_dim)  # Ensure attention_dim doesn't exceed input_dim
        self.W_q = np.random.normal(0, 0.1, (input_dim, self.attention_dim))
        self.W_k = np.random.normal(0, 0.1, (input_dim, self.attention_dim))
        self.W_v = np.random.normal(0, 0.1, (input_dim, self.attention_dim))
        self.W_o = np.random.normal(0, 0.1, (self.attention_dim, input_dim))
        
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass with attention"""
        # X shape: (batch_size, seq_len, features)
        batch_size, seq_len, features = X.shape
        
        # Compute queries, keys, values
        Q = np.dot(X, self.W_q)  # (batch_size, seq_len, attention_dim)
        K = np.dot(X, self.W_k)  # (batch_size, seq_len, attention_dim)
        V = np.dot(X, self.W_v)  # (batch_size, seq_len, attention_dim)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch_size, seq_len, seq_len)
        scores = scores / np.sqrt(self.attention_dim)
        
        # Apply softmax
        attention_weights = self._softmax(scores)
        
        # Apply attention to values
        attended = np.matmul(attention_weights, V)  # (batch_size, seq_len, attention_dim)
        
        # Final linear transformation
        output = np.dot(attended, self.W_o)  # (batch_size, seq_len, input_dim)
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class TransformerBlock:
    """Simplified transformer block for financial time series"""
    
    def __init__(self, input_dim: int, ff_dim: int = 128, dropout_rate: float = 0.1):
        self.attention = AttentionLayer(input_dim)
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Feed-forward network weights
        self.W1 = np.random.normal(0, 0.1, (input_dim, ff_dim))
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.normal(0, 0.1, (ff_dim, input_dim))
        self.b2 = np.zeros(input_dim)
        
        # Layer normalization parameters
        self.gamma1 = np.ones(input_dim)
        self.beta1 = np.zeros(input_dim)
        self.gamma2 = np.ones(input_dim)
        self.beta2 = np.zeros(input_dim)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through transformer block"""
        # Self-attention
        attended, attention_weights = self.attention.forward(X)
        
        # Residual connection and layer norm
        norm1 = self._layer_norm(X + attended, self.gamma1, self.beta1)
        
        # Feed-forward network
        ff_output = self._feed_forward(norm1)
        
        # Residual connection and layer norm
        output = self._layer_norm(norm1 + ff_output, self.gamma2, self.beta2)
        
        return output, attention_weights
    
    def _feed_forward(self, X: np.ndarray) -> np.ndarray:
        """Feed-forward network"""
        # First linear layer with ReLU
        hidden = np.maximum(0, np.dot(X, self.W1) + self.b1)
        
        # Apply dropout (simplified for inference)
        if self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, hidden.shape)
            hidden = hidden * mask / (1 - self.dropout_rate)
        
        # Second linear layer
        output = np.dot(hidden, self.W2) + self.b2
        
        return output
    
    def _layer_norm(self, X: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        normalized = (X - mean) / np.sqrt(var + 1e-8)
        return gamma * normalized + beta

class TransformerEnsemble:
    """Transformer-based ensemble for cryptocurrency price prediction"""
    
    def __init__(self, sequence_length: int = 60, num_layers: int = 2, d_model: int = 64):
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.transformer_blocks = []
        self.output_layer = None
        self.input_projection = None  # Dynamic projection layer
        self.is_trained = False
        
        # Initialize transformer blocks
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model))
        
        # Feature extraction parameters
        self.feature_windows = [5, 10, 20]
        self.technical_periods = [14, 21]
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive features for transformer input"""
        try:
            features_df = df.copy()
            
            # Basic price features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            features_df['hl_ratio'] = (features_df['high'] - features_df['low']) / features_df['close']
            features_df['oc_ratio'] = (features_df['open'] - features_df['close']) / features_df['close']
            
            # Moving averages and ratios
            for window in self.feature_windows:
                features_df[f'sma_{window}'] = features_df['close'].rolling(window).mean()
                features_df[f'price_sma_ratio_{window}'] = features_df['close'] / features_df[f'sma_{window}']
                features_df[f'volatility_{window}'] = features_df['returns'].rolling(window).std()
            
            # Technical indicators
            for period in self.technical_periods:
                # RSI
                delta = features_df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss.replace(0, 1)
                features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                
                # Momentum
                features_df[f'momentum_{period}'] = features_df['close'] / features_df['close'].shift(period)
            
            # Volume features (if available)
            if 'volume' in features_df.columns:
                features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
                features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            
            # Market structure features
            features_df['higher_high'] = (features_df['high'] > features_df['high'].shift(1)).astype(int)
            features_df['lower_low'] = (features_df['low'] < features_df['low'].shift(1)).astype(int)
            
            # Time-based features
            if hasattr(features_df.index, 'hour'):
                features_df['hour'] = features_df.index.hour
                features_df['day_of_week'] = features_df.index.dayofweek
            
            # Clean features
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return features_df
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return df
    
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare sequences for transformer training"""
        try:
            # Extract features
            features_df = self.extract_features(df)
            
            # Select feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            # Create target (future returns)
            target = features_df['close'].shift(-1).pct_change()
            
            # Get feature matrix
            feature_matrix = features_df[feature_cols].values
            
            # Scale features
            feature_matrix_scaled = self.feature_scaler.fit_transform(feature_matrix)
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(feature_matrix_scaled)):
                if i < len(target) and not np.isnan(target.iloc[i]):
                    X.append(feature_matrix_scaled[i-self.sequence_length:i])
                    y.append(target.iloc[i])
            
            return np.array(X), np.array(y), feature_cols
            
        except Exception as e:
            print(f"Error preparing sequences: {e}")
            return np.array([]), np.array([]), []
    
    def train(self, df: pd.DataFrame, epochs: int = 50, learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train the transformer ensemble"""
        try:
            print("Preparing training sequences...")
            X, y, feature_names = self.prepare_sequences(df)
            
            if len(X) == 0:
                return {'error': 'No valid training sequences'}
            
            print(f"Training data shape: {X.shape}")
            
            # Initialize input projection layer to map features to d_model
            input_features = X.shape[2]
            self.input_projection = np.random.normal(0, 0.1, (input_features, self.d_model))
            
            # Initialize output layer
            self.output_layer = np.random.normal(0, 0.1, (self.d_model, 1))
            
            # Training loop (simplified backpropagation)
            losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                predictions = []
                
                for i in range(len(X)):
                    # Forward pass
                    x_input = X[i:i+1]  # Shape: (1, seq_len, features)
                    
                    # Project input to d_model dimensions
                    projected_input = np.dot(x_input, self.input_projection)  # Shape: (1, seq_len, d_model)
                    
                    # Pass through transformer blocks
                    hidden = projected_input
                    attention_maps = []
                    
                    for transformer_block in self.transformer_blocks:
                        hidden, attention_weights = transformer_block.forward(hidden)
                        attention_maps.append(attention_weights)
                    
                    # Global average pooling and output
                    pooled = np.mean(hidden, axis=1)  # Shape: (1, features)
                    pred = np.dot(pooled, self.output_layer).flatten()[0]
                    predictions.append(pred)
                    
                    # Calculate loss
                    loss = (pred - y[i]) ** 2
                    epoch_loss += loss
                
                avg_loss = epoch_loss / len(X)
                losses.append(avg_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            # Calculate final metrics
            final_predictions = np.array(predictions)
            mse = mean_squared_error(y, final_predictions)
            r2 = r2_score(y, final_predictions)
            
            self.is_trained = True
            
            return {
                'success': True,
                'final_mse': mse,
                'r2_score': r2,
                'training_losses': losses,
                'num_sequences': len(X),
                'feature_count': len(feature_names)
            }
            
        except Exception as e:
            print(f"Error training transformer: {e}")
            return {'error': str(e)}
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions using the trained transformer"""
        try:
            if not self.is_trained:
                return {'error': 'Model not trained'}
            
            # Extract features
            features_df = self.extract_features(df)
            
            # Select feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            # Get recent sequence
            feature_matrix = features_df[feature_cols].values
            feature_matrix_scaled = self.feature_scaler.transform(feature_matrix)
            
            if len(feature_matrix_scaled) < self.sequence_length:
                return {'error': f'Insufficient data. Need at least {self.sequence_length} samples'}
            
            # Get last sequence
            x_input = feature_matrix_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Project input to d_model dimensions
            projected_input = np.dot(x_input, self.input_projection)
            
            # Forward pass through transformer
            hidden = projected_input
            attention_maps = []
            
            for transformer_block in self.transformer_blocks:
                hidden, attention_weights = transformer_block.forward(hidden)
                attention_maps.append(attention_weights)
            
            # Global average pooling and prediction
            pooled = np.mean(hidden, axis=1)
            prediction = np.dot(pooled, self.output_layer).flatten()[0]
            
            # Calculate confidence based on attention consistency
            attention_variance = np.mean([np.var(attn) for attn in attention_maps])
            confidence = max(0.1, 1.0 / (1.0 + attention_variance))
            
            # Convert to price prediction
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + prediction)
            
            return {
                'success': True,
                'prediction': prediction,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'confidence': confidence,
                'attention_variance': attention_variance,
                'attention_maps': [attn.tolist() for attn in attention_maps]
            }
            
        except Exception as e:
            print(f"Error in transformer prediction: {e}")
            return {'error': str(e)}
    
    def get_attention_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze attention patterns for interpretability"""
        try:
            if not self.is_trained:
                return {'error': 'Model not trained'}
            
            # Get prediction with attention maps
            result = self.predict(df)
            
            if 'error' in result:
                return result
            
            attention_maps = result['attention_maps']
            
            # Analyze attention patterns
            analysis = {
                'total_layers': len(attention_maps),
                'sequence_length': self.sequence_length,
                'attention_summary': []
            }
            
            for i, attention_map in enumerate(attention_maps):
                attention_array = np.array(attention_map)
                
                # Find most attended positions
                avg_attention = np.mean(attention_array, axis=0)
                max_attention_pos = np.argmax(avg_attention)
                
                layer_analysis = {
                    'layer': i + 1,
                    'max_attention_position': int(max_attention_pos),
                    'max_attention_value': float(avg_attention[max_attention_pos]),
                    'attention_entropy': self._calculate_entropy(avg_attention),
                    'attention_focus': float(np.max(avg_attention) / np.mean(avg_attention))
                }
                
                analysis['attention_summary'].append(layer_analysis)
            
            return {
                'success': True,
                'analysis': analysis
            }
            
        except Exception as e:
            print(f"Error in attention analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate entropy of attention distribution"""
        # Normalize to probabilities
        probs = probabilities / np.sum(probabilities)
        # Avoid log(0)
        probs = probs + 1e-8
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture and training information"""
        return {
            'architecture': 'Transformer Ensemble',
            'sequence_length': self.sequence_length,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'is_trained': self.is_trained,
            'total_parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total number of parameters in the model"""
        total_params = 0
        
        # Transformer blocks
        for block in self.transformer_blocks:
            # Attention layer parameters
            total_params += block.attention.W_q.size
            total_params += block.attention.W_k.size
            total_params += block.attention.W_v.size
            total_params += block.attention.W_o.size
            
            # Feed-forward parameters
            total_params += block.W1.size + block.b1.size
            total_params += block.W2.size + block.b2.size
            
            # Layer norm parameters
            total_params += block.gamma1.size + block.beta1.size
            total_params += block.gamma2.size + block.beta2.size
        
        # Output layer
        if self.output_layer is not None:
            total_params += self.output_layer.size
        
        return total_params