import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimpleMLPipeline:
    """Simplified ML pipeline using sklearn models only"""
    
    def __init__(self, prediction_horizon: int = 1):
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.feature_importance = {}
        self.model_scores = {}
        self.is_trained = False
        self.scaler = StandardScaler()
        
        # Available model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05,
                'random_state': 42
            }
        }
        
        # Feature generation parameters
        self.feature_windows = [5, 10, 20, 50]
        self.technical_periods = [14, 21, 50]
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        try:
            features_df = df.copy()
            
            # Price-based features
            for window in self.feature_windows:
                if len(df) > window:
                    features_df[f'sma_{window}'] = df['close'].rolling(window).mean()
                    features_df[f'std_{window}'] = df['close'].rolling(window).std()
                    features_df[f'min_{window}'] = df['close'].rolling(window).min()
                    features_df[f'max_{window}'] = df['close'].rolling(window).max()
                    features_df[f'range_{window}'] = features_df[f'max_{window}'] - features_df[f'min_{window}']
            
            # Returns and momentum
            features_df['returns'] = df['close'].pct_change()
            features_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            for window in [5, 10, 20]:
                if len(df) > window:
                    features_df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
                    features_df[f'roc_{window}'] = df['close'].pct_change(window)
            
            # Technical indicators
            if len(df) > 14:
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volume features if available
            if 'volume' in df.columns:
                for window in [5, 10, 20]:
                    if len(df) > window:
                        features_df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                        features_df[f'volume_ratio_{window}'] = df['volume'] / features_df[f'volume_sma_{window}']
            
            # Time-based features
            if 'timestamp' in features_df.columns:
                features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
                features_df['day_of_week'] = pd.to_datetime(features_df['timestamp']).dt.dayofweek
                features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
                features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
                features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
            
            return features_df
            
        except Exception as e:
            print(f"Error generating features: {e}")
            return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data with target variable"""
        try:
            # Generate features
            features_df = self.generate_features(df)
            
            # Create target variable (future returns)
            target = features_df['close'].shift(-self.prediction_horizon).pct_change()
            
            # Select feature columns (exclude price columns and target)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            # Prepare arrays
            X = features_df[feature_cols].fillna(0).values
            y = target.fillna(0).values
            
            # Remove rows with NaN target
            valid_mask = ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Scale features
            if len(X) > 0:
                X = self.scaler.fit_transform(X)
            
            return X, y, feature_cols
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return np.array([]), np.array([]), []
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train individual model"""
        try:
            if model_name == 'random_forest':
                model = RandomForestRegressor(**self.model_configs['random_forest'])
            elif model_name == 'gradient_boosting':
                model = GradientBoostingRegressor(**self.model_configs['gradient_boosting'])
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Train model
            model.fit(X, y)
            
            # Feature importance
            importance = dict(zip(feature_names, model.feature_importances_))
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=3), 
                                      scoring='neg_mean_squared_error')
            cv_score = -cv_scores.mean()
            
            # Store model and results
            self.models[model_name] = model
            self.feature_importance[model_name] = importance
            self.model_scores[model_name] = {
                'cv_score': cv_score,
                'cv_std': cv_scores.std(),
                'train_score': mean_squared_error(y, model.predict(X))
            }
            
            return {
                'success': True,
                'cv_score': cv_score,
                'feature_importance': importance
            }
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all available models"""
        try:
            print("Preparing training data...")
            X, y, feature_names = self.prepare_training_data(df)
            
            if len(X) == 0:
                return {'error': 'No valid training data'}
            
            print(f"Training data shape: {X.shape}")
            print(f"Number of features: {len(feature_names)}")
            
            results = {}
            available_models = list(self.model_configs.keys())
            
            for model_name in available_models:
                print(f"Training {model_name}...")
                result = self.train_model(model_name, X, y, feature_names)
                results[model_name] = result
            
            self.is_trained = True
            
            return {
                'success': True,
                'results': results,
                'feature_count': len(feature_names),
                'sample_count': len(X),
                'available_models': available_models
            }
            
        except Exception as e:
            print(f"Error in training pipeline: {e}")
            return {'error': str(e)}
    
    def predict_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate ensemble predictions"""
        try:
            if not self.is_trained:
                return {'error': 'Models not trained'}
            
            # Generate features for prediction
            features_df = self.generate_features(df)
            
            # Select same features used in training
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            # Get last row for prediction
            X_pred = features_df[feature_cols].fillna(0).iloc[-1:].values
            X_pred = self.scaler.transform(X_pred)
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                pred = model.predict(X_pred)[0]
                predictions[model_name] = pred
            
            # Ensemble prediction (average)
            ensemble_pred = np.mean(list(predictions.values()))
            
            # Calculate confidence based on agreement
            pred_values = list(predictions.values())
            confidence = 1.0 - (np.std(pred_values) / (np.abs(np.mean(pred_values)) + 1e-8))
            confidence = max(0.0, min(1.0, confidence))
            
            # Convert to price prediction
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + ensemble_pred)
            
            return {
                'success': True,
                'ensemble_prediction': ensemble_pred,
                'predicted_price': predicted_price,
                'confidence': confidence,
                'individual_predictions': predictions
            }
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return {'error': str(e)}
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get feature importance summary across all models"""
        try:
            if not self.feature_importance:
                return {'error': 'No trained models'}
            
            # Aggregate importance across models
            all_features = set()
            for model_importance in self.feature_importance.values():
                all_features.update(model_importance.keys())
            
            aggregated_importance = {}
            for feature in all_features:
                importances = []
                for model_importance in self.feature_importance.values():
                    if feature in model_importance:
                        importances.append(model_importance[feature])
                
                if importances:
                    aggregated_importance[feature] = {
                        'mean': np.mean(importances),
                        'std': np.std(importances),
                        'count': len(importances)
                    }
            
            # Sort by mean importance
            sorted_features = sorted(aggregated_importance.items(), 
                                   key=lambda x: x[1]['mean'], reverse=True)
            
            return {
                'success': True,
                'top_features': sorted_features[:20],
                'total_features': len(all_features)
            }
            
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {'error': str(e)}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            if not self.model_scores:
                return {'error': 'No trained models'}
            
            performance = {}
            for model_name, scores in self.model_scores.items():
                performance[model_name] = {
                    'cross_validation_score': scores['cv_score'],
                    'cv_std': scores['cv_std'],
                    'training_score': scores['train_score'],
                    'stability': 1.0 / (1.0 + scores['cv_std'])  # Higher is more stable
                }
            
            return {
                'success': True,
                'model_performance': performance
            }
            
        except Exception as e:
            print(f"Error getting model performance: {e}")
            return {'error': str(e)}