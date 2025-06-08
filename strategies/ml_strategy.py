import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from .base_strategy import TechnicalStrategy
import warnings
warnings.filterwarnings('ignore')

class MLStrategy(TechnicalStrategy):
    """Machine Learning strategy using various ML algorithms"""
    
    def __init__(self):
        super().__init__("ML Strategy")
        
        # ML Models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = {}
        self.model_scores = {}
        
        # Feature engineering parameters
        self.lookback_periods = [5, 10, 20, 50]
        self.target_horizon = 5  # Predict 5 periods ahead
        self.min_samples = 200   # Minimum samples for training
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models"""
        try:
            if len(data) < max(self.lookback_periods) + self.target_horizon:
                return pd.DataFrame()
            
            features_df = data.copy()
            
            # Price-based features
            features_df['returns'] = data['close'].pct_change()
            features_df['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            # Volatility features
            for period in [5, 10, 20]:
                features_df[f'volatility_{period}'] = features_df['returns'].rolling(period).std()
                features_df[f'realized_vol_{period}'] = np.sqrt(
                    (features_df['log_returns'] ** 2).rolling(period).sum()
                )
            
            # Price momentum features
            for period in self.lookback_periods:
                features_df[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
                features_df[f'roc_{period}'] = data['close'].pct_change(period)
            
            # Moving average features
            for period in [10, 20, 50]:
                ma = data['close'].rolling(period).mean()
                features_df[f'ma_{period}'] = ma
                features_df[f'price_to_ma_{period}'] = data['close'] / ma - 1
                features_df[f'ma_slope_{period}'] = ma.diff(5) / ma.shift(5)
            
            # Volume features
            features_df['volume_ma_20'] = data['volume'].rolling(20).mean()
            features_df['volume_ratio'] = data['volume'] / features_df['volume_ma_20']
            features_df['volume_roc'] = data['volume'].pct_change(10)
            
            # Price channels and bands
            for period in [10, 20]:
                high_channel = data['high'].rolling(period).max()
                low_channel = data['low'].rolling(period).min()
                features_df[f'channel_position_{period}'] = (
                    (data['close'] - low_channel) / (high_channel - low_channel)
                )
                features_df[f'channel_width_{period}'] = (
                    (high_channel - low_channel) / data['close']
                )
            
            # Technical indicators
            features_df = self._add_technical_features(features_df, data)
            
            # Lagged features
            features_df = self._add_lagged_features(features_df)
            
            # Statistical features
            features_df = self._add_statistical_features(features_df, data)
            
            # Create target variable
            future_returns = data['close'].shift(-self.target_horizon) / data['close'] - 1
            
            # Classify returns into bins
            features_df['target_return'] = future_returns
            features_df['target_class'] = self._classify_returns(future_returns)
            
            # Remove rows with NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            print(f"Error engineering features: {e}")
            return pd.DataFrame()
    
    def _add_technical_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            features_df['macd'] = exp1 - exp2
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_ma = data['close'].rolling(bb_period).mean()
            bb_std_dev = data['close'].rolling(bb_period).std()
            features_df['bb_upper'] = bb_ma + (bb_std_dev * bb_std)
            features_df['bb_lower'] = bb_ma - (bb_std_dev * bb_std)
            features_df['bb_position'] = (data['close'] - features_df['bb_lower']) / (
                features_df['bb_upper'] - features_df['bb_lower']
            )
            features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / bb_ma
            
            # Stochastic Oscillator
            high_14 = data['high'].rolling(14).max()
            low_14 = data['low'].rolling(14).min()
            features_df['stoch_k'] = 100 * ((data['close'] - low_14) / (high_14 - low_14))
            features_df['stoch_d'] = features_df['stoch_k'].rolling(3).mean()
            
            return features_df
            
        except Exception as e:
            print(f"Error adding technical features: {e}")
            return features_df
    
    def _add_lagged_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of key features"""
        try:
            key_features = ['returns', 'volatility_5', 'rsi', 'macd']
            lags = [1, 2, 3, 5]
            
            for feature in key_features:
                if feature in features_df.columns:
                    for lag in lags:
                        features_df[f'{feature}_lag_{lag}'] = features_df[feature].shift(lag)
            
            return features_df
            
        except Exception as e:
            print(f"Error adding lagged features: {e}")
            return features_df
    
    def _add_statistical_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        try:
            # Rolling statistics
            for period in [5, 10, 20]:
                features_df[f'skew_{period}'] = data['close'].pct_change().rolling(period).skew()
                features_df[f'kurt_{period}'] = data['close'].pct_change().rolling(period).kurt()
                
                # Price quantiles
                features_df[f'price_quantile_{period}'] = (
                    data['close'].rolling(period).rank() / period
                )
            
            # Autocorrelation
            returns = data['close'].pct_change()
            for lag in [1, 2, 5]:
                features_df[f'autocorr_lag_{lag}'] = (
                    returns.rolling(20).apply(lambda x: x.autocorr(lag=lag))
                )
            
            return features_df
            
        except Exception as e:
            print(f"Error adding statistical features: {e}")
            return features_df
    
    def _classify_returns(self, returns: pd.Series) -> pd.Series:
        """Classify returns into discrete classes"""
        try:
            # Define thresholds for classification
            buy_threshold = 0.01    # 1% positive return
            sell_threshold = -0.01  # 1% negative return
            
            conditions = [
                returns > buy_threshold,
                returns < sell_threshold
            ]
            choices = ['BUY', 'SELL']
            
            classes = pd.Series(np.select(conditions, choices, default='HOLD'), index=returns.index)
            return classes
            
        except Exception as e:
            print(f"Error classifying returns: {e}")
            return pd.Series(['HOLD'] * len(returns), index=returns.index)
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        try:
            if features_df.empty or len(features_df) < self.min_samples:
                return None, None, None
            
            # Select feature columns (exclude target and OHLCV columns)
            exclude_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'target_return', 'target_class'
            ]
            
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            X = features_df[feature_cols].values
            y = features_df['target_class'].values
            timestamps = features_df.index
            
            # Remove samples with NaN values
            mask = ~(np.isnan(X).any(axis=1) | pd.isna(y))
            X = X[mask]
            y = y[mask]
            timestamps = timestamps[mask]
            
            if len(X) < self.min_samples:
                return None, None, None
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            return X, y_encoded, feature_cols
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None, None
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ML models"""
        try:
            print("Engineering features...")
            features_df = self.engineer_features(data)
            
            if features_df.empty:
                return {'error': 'Failed to engineer features'}
            
            print("Preparing training data...")
            X, y, feature_cols = self.prepare_training_data(features_df)
            
            if X is None:
                return {'error': 'Insufficient training data'}
            
            print(f"Training with {len(X)} samples and {len(feature_cols)} features...")
            
            results = {}
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            for model_name, model in self.models.items():
                print(f"Training {model_name}...")
                
                try:
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[model_name] = scaler
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
                    
                    # Train final model
                    model.fit(X_scaled, y)
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = dict(zip(feature_cols, model.feature_importances_))
                        self.feature_importance[model_name] = importance
                    
                    results[model_name] = {
                        'cv_scores': cv_scores.tolist(),
                        'mean_cv_score': cv_scores.mean(),
                        'std_cv_score': cv_scores.std(),
                        'feature_count': len(feature_cols),
                        'training_samples': len(X)
                    }
                    
                    self.model_scores[model_name] = cv_scores.mean()
                    
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
            
            self.is_trained = True
            return results
            
        except Exception as e:
            print(f"Error in train_models: {e}")
            return {'error': str(e)}
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ML predictions"""
        try:
            if not self.is_trained:
                return {'error': 'Models not trained'}
            
            features_df = self.engineer_features(data)
            if features_df.empty or len(features_df) == 0:
                return {'error': 'Failed to engineer features for prediction'}
            
            # Get latest features
            exclude_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'target_return', 'target_class'
            ]
            
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            latest_features = features_df[feature_cols].iloc[-1:].values
            
            if np.isnan(latest_features).any():
                return {'error': 'NaN values in features'}
            
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name in self.scalers:
                        scaler = self.scalers[model_name]
                        features_scaled = scaler.transform(latest_features)
                        
                        # Get prediction and probability
                        pred_class = model.predict(features_scaled)[0]
                        pred_proba = model.predict_proba(features_scaled)[0]
                        
                        # Convert back to original labels
                        pred_label = self.label_encoder.inverse_transform([pred_class])[0]
                        
                        predictions[model_name] = pred_label
                        confidences[model_name] = max(pred_proba)
                        
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
                    predictions[model_name] = 'HOLD'
                    confidences[model_name] = 0.0
            
            # Ensemble prediction
            if predictions:
                # Weight predictions by model performance
                weighted_votes = {}
                for model_name, prediction in predictions.items():
                    weight = self.model_scores.get(model_name, 0.5)
                    if prediction not in weighted_votes:
                        weighted_votes[prediction] = 0
                    weighted_votes[prediction] += weight * confidences.get(model_name, 0.5)
                
                # Get prediction with highest weighted vote
                ensemble_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
                ensemble_confidence = max(weighted_votes.values()) / sum(weighted_votes.values())
                
                return {
                    'ensemble_prediction': ensemble_prediction,
                    'ensemble_confidence': ensemble_confidence,
                    'individual_predictions': predictions,
                    'individual_confidences': confidences,
                    'feature_count': len(feature_cols)
                }
            
            return {'error': 'No valid predictions generated'}
            
        except Exception as e:
            print(f"Error in predict: {e}")
            return {'error': str(e)}
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ML-based trading signal"""
        try:
            prediction_result = self.predict(data)
            
            if 'error' in prediction_result:
                return {
                    'signal': 'HOLD',
                    'strength': 0.0,
                    'confidence': 0.0,
                    'error': prediction_result['error']
                }
            
            ensemble_pred = prediction_result.get('ensemble_prediction', 'HOLD')
            ensemble_conf = prediction_result.get('ensemble_confidence', 0.0)
            
            # Convert confidence to strength
            strength = ensemble_conf
            
            signal_result = {
                'signal': ensemble_pred,
                'strength': strength,
                'confidence': ensemble_conf,
                'ml_predictions': prediction_result,
                'model_performance': self.model_scores
            }
            
            self.add_signal_to_history(signal_result)
            return signal_result
            
        except Exception as e:
            print(f"Error generating ML signal: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators (required by base class)"""
        return self.engineer_features(data)
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """Get feature importance for models"""
        if model_name and model_name in self.feature_importance:
            return self.feature_importance[model_name]
        return self.feature_importance
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return {
            'model_scores': self.model_scores,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance
        }
