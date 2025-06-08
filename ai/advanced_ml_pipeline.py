import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLPipeline:
    """Comprehensive ML pipeline with gradient boosting models and auto feature generation"""
    
    def __init__(self, prediction_horizon: int = 1):
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_scores = {}
        self.is_trained = False
        
        # Initialize models with optimized parameters
        self.model_configs = {
            'lightgbm': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            },
            'xgboost': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'random_state': 42,
                'verbosity': 0
            },
            'catboost': {
                'iterations': 100,
                'learning_rate': 0.05,
                'depth': 6,
                'loss_function': 'RMSE',
                'verbose': False,
                'random_seed': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        
        # Feature generation parameters
        self.feature_windows = [5, 10, 20, 50]
        self.technical_periods = [14, 21, 50]
        
    def generate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set with FreqAI-level sophistication"""
        try:
            features_df = df.copy()
            
            # Basic price features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            features_df['price_change'] = features_df['close'].diff()
            
            # Rolling statistics for multiple windows
            for window in self.feature_windows:
                # Price statistics
                features_df[f'close_mean_{window}'] = features_df['close'].rolling(window).mean()
                features_df[f'close_std_{window}'] = features_df['close'].rolling(window).std()
                features_df[f'close_min_{window}'] = features_df['close'].rolling(window).min()
                features_df[f'close_max_{window}'] = features_df['close'].rolling(window).max()
                features_df[f'close_median_{window}'] = features_df['close'].rolling(window).median()
                
                # Price ratios
                features_df[f'price_to_mean_{window}'] = features_df['close'] / features_df[f'close_mean_{window}']
                features_df[f'price_to_max_{window}'] = features_df['close'] / features_df[f'close_max_{window}']
                features_df[f'price_to_min_{window}'] = features_df['close'] / features_df[f'close_min_{window}']
                
                # Volatility features
                features_df[f'volatility_{window}'] = features_df['returns'].rolling(window).std()
                features_df[f'volatility_ratio_{window}'] = (
                    features_df[f'volatility_{window}'] / features_df[f'volatility_{window}'].rolling(window*2).mean()
                )
                
                # Volume features if available
                if 'volume' in features_df.columns:
                    features_df[f'volume_mean_{window}'] = features_df['volume'].rolling(window).mean()
                    features_df[f'volume_std_{window}'] = features_df['volume'].rolling(window).std()
                    features_df[f'volume_ratio_{window}'] = features_df['volume'] / features_df[f'volume_mean_{window}']
                
                # Momentum features
                features_df[f'momentum_{window}'] = features_df['close'] / features_df['close'].shift(window)
                features_df[f'roc_{window}'] = ((features_df['close'] - features_df['close'].shift(window)) / 
                                              features_df['close'].shift(window)) * 100
            
            # Technical indicators
            for period in self.technical_periods:
                # RSI
                delta = features_df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss.replace(0, 1)
                features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                sma = features_df['close'].rolling(period).mean()
                std = features_df['close'].rolling(period).std()
                features_df[f'bb_upper_{period}'] = sma + (std * 2)
                features_df[f'bb_lower_{period}'] = sma - (std * 2)
                features_df[f'bb_position_{period}'] = (features_df['close'] - features_df[f'bb_lower_{period}']) / (
                    features_df[f'bb_upper_{period}'] - features_df[f'bb_lower_{period}']
                )
                
                # MACD components
                if period == 14:  # Use default MACD parameters
                    ema_12 = features_df['close'].ewm(span=12).mean()
                    ema_26 = features_df['close'].ewm(span=26).mean()
                    features_df['macd'] = ema_12 - ema_26
                    features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
                    features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Advanced statistical features
            # Skewness and Kurtosis
            for window in [20, 50]:
                features_df[f'skewness_{window}'] = features_df['returns'].rolling(window).skew()
                features_df[f'kurtosis_{window}'] = features_df['returns'].rolling(window).kurt()
            
            # Autocorrelation features
            for lag in [1, 2, 5, 10]:
                features_df[f'autocorr_lag_{lag}'] = features_df['returns'].rolling(50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
                )
            
            # Trend strength indicators
            for window in [10, 20, 50]:
                # Linear trend slope
                def calculate_slope(series):
                    if len(series) < 2:
                        return np.nan
                    x = np.arange(len(series))
                    return np.polyfit(x, series, 1)[0]
                
                features_df[f'trend_slope_{window}'] = features_df['close'].rolling(window).apply(calculate_slope)
                
                # R-squared of linear trend
                def calculate_r2(series):
                    if len(series) < 2:
                        return np.nan
                    x = np.arange(len(series))
                    try:
                        slope, intercept = np.polyfit(x, series, 1)
                        y_pred = slope * x + intercept
                        ss_res = np.sum((series - y_pred) ** 2)
                        ss_tot = np.sum((series - np.mean(series)) ** 2)
                        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    except:
                        return np.nan
                
                features_df[f'trend_strength_{window}'] = features_df['close'].rolling(window).apply(calculate_r2)
            
            # Cross-asset features (if multiple timeframes available)
            # Price acceleration
            features_df['price_acceleration'] = features_df['returns'].diff()
            
            # Regime indicators
            features_df['high_volatility_regime'] = (features_df['volatility_20'] > 
                                                   features_df['volatility_20'].rolling(100).quantile(0.8)).astype(int)
            
            # Remove infinite and NaN values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return features_df
            
        except Exception as e:
            print(f"Error in feature generation: {e}")
            return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data with target variable"""
        try:
            # Generate features
            features_df = self.generate_comprehensive_features(df)
            
            # Create target variable (future returns)
            target = features_df['close'].shift(-self.prediction_horizon).pct_change()
            
            # Select feature columns (exclude OHLCV and target)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            # Remove rows with NaN target
            valid_indices = ~target.isna()
            X = features_df[feature_cols].loc[valid_indices].values
            y = target.loc[valid_indices].values
            
            return X, y, feature_cols
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return np.array([]), np.array([]), []
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train individual model"""
        try:
            if model_name == 'lightgbm':
                # LightGBM training
                train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
                model = lgb.train(
                    self.model_configs['lightgbm'],
                    train_data,
                    num_boost_round=100,
                    valid_sets=[train_data],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
                
                # Feature importance
                importance = dict(zip(feature_names, model.feature_importance()))
                
            elif model_name == 'xgboost':
                # XGBoost training
                model = xgb.XGBRegressor(**self.model_configs['xgboost'])
                model.fit(X, y)
                
                # Feature importance
                importance = dict(zip(feature_names, model.feature_importances_))
                
            elif model_name == 'catboost':
                # CatBoost training
                model = cb.CatBoostRegressor(**self.model_configs['catboost'])
                model.fit(X, y)
                
                # Feature importance
                importance = dict(zip(feature_names, model.feature_importances_))
                
            elif model_name == 'random_forest':
                # Random Forest training
                model = RandomForestRegressor(**self.model_configs['random_forest'])
                model.fit(X, y)
                
                # Feature importance
                importance = dict(zip(feature_names, model.feature_importances_))
            
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
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
                'train_score': mean_squared_error(y, model.predict(X) if hasattr(model, 'predict') else model.predict(X))
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
        """Train all models in the pipeline"""
        try:
            print("Preparing training data...")
            X, y, feature_names = self.prepare_training_data(df)
            
            if len(X) == 0:
                return {'error': 'No valid training data'}
            
            print(f"Training data shape: {X.shape}")
            print(f"Number of features: {len(feature_names)}")
            
            results = {}
            
            # Train each model
            for model_name in ['lightgbm', 'xgboost', 'catboost', 'random_forest']:
                print(f"Training {model_name}...")
                result = self.train_model(model_name, X, y, feature_names)
                results[model_name] = result
            
            self.is_trained = True
            
            return {
                'success': True,
                'results': results,
                'feature_count': len(feature_names),
                'sample_count': len(X)
            }
            
        except Exception as e:
            print(f"Error in training pipeline: {e}")
            return {'error': str(e)}
    
    def predict_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate ensemble predictions from all models"""
        try:
            if not self.is_trained:
                return {'error': 'Models not trained'}
            
            # Generate features
            features_df = self.generate_comprehensive_features(df)
            
            # Get feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            # Prepare input data
            X = features_df[feature_cols].iloc[-1:].values
            
            predictions = {}
            confidences = {}
            
            # Generate predictions from each model
            for model_name, model in self.models.items():
                try:
                    if model_name == 'lightgbm':
                        pred = model.predict(X)[0]
                    else:
                        pred = model.predict(X)[0]
                    
                    predictions[model_name] = pred
                    
                    # Calculate confidence based on cross-validation score
                    cv_score = self.model_scores[model_name]['cv_score']
                    confidence = max(0.1, 1.0 / (1.0 + cv_score))  # Convert error to confidence
                    confidences[model_name] = confidence
                    
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
                    predictions[model_name] = 0.0
                    confidences[model_name] = 0.1
            
            # Calculate weighted ensemble prediction
            total_weight = sum(confidences.values())
            if total_weight > 0:
                ensemble_pred = sum(pred * confidences[model] for model, pred in predictions.items()) / total_weight
                ensemble_confidence = sum(confidences.values()) / len(confidences)
            else:
                ensemble_pred = 0.0
                ensemble_confidence = 0.1
            
            # Convert prediction to price change
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + ensemble_pred)
            
            return {
                'success': True,
                'ensemble_prediction': ensemble_pred,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'model_confidences': confidences,
                'prediction_horizon': self.prediction_horizon
            }
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return {'error': str(e)}
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get comprehensive feature importance analysis"""
        try:
            if not self.feature_importance:
                return {'error': 'No feature importance data available'}
            
            # Aggregate feature importance across models
            all_features = set()
            for model_importance in self.feature_importance.values():
                all_features.update(model_importance.keys())
            
            aggregated_importance = {}
            for feature in all_features:
                importances = []
                for model_name, model_importance in self.feature_importance.items():
                    if feature in model_importance:
                        importances.append(model_importance[feature])
                
                if importances:
                    aggregated_importance[feature] = {
                        'mean': np.mean(importances),
                        'std': np.std(importances),
                        'models_count': len(importances)
                    }
            
            # Sort by mean importance
            sorted_features = sorted(aggregated_importance.items(), 
                                   key=lambda x: x[1]['mean'], reverse=True)
            
            return {
                'success': True,
                'top_features': sorted_features[:20],  # Top 20 features
                'total_features': len(aggregated_importance),
                'model_specific': self.feature_importance
            }
            
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {'error': str(e)}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics"""
        try:
            if not self.model_scores:
                return {'error': 'No model performance data available'}
            
            performance_summary = {}
            
            for model_name, scores in self.model_scores.items():
                performance_summary[model_name] = {
                    'cross_validation_score': scores['cv_score'],
                    'cv_std': scores['cv_std'],
                    'training_score': scores['train_score'],
                    'stability': 1.0 / (1.0 + scores['cv_std'])  # Higher is more stable
                }
            
            # Rank models by performance
            ranked_models = sorted(performance_summary.items(), 
                                 key=lambda x: x[1]['cross_validation_score'])
            
            return {
                'success': True,
                'model_performance': performance_summary,
                'best_model': ranked_models[0][0] if ranked_models else None,
                'model_ranking': [model[0] for model in ranked_models]
            }
            
        except Exception as e:
            print(f"Error getting model performance: {e}")
            return {'error': str(e)}