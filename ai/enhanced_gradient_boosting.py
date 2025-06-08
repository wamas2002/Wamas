import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced gradient boosting libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, OSError) as e:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except (ImportError, OSError) as e:
    CATBOOST_AVAILABLE = False
    cb = None

class EnhancedGradientBoostingPipeline:
    """Enhanced gradient boosting pipeline with LightGBM, XGBoost, and CatBoost"""
    
    def __init__(self, prediction_horizon: int = 1):
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.feature_importance = {}
        self.model_scores = {}
        self.is_trained = False
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Model configurations
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
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_estimators': 100,
                'verbosity': 0
            },
            'catboost': {
                'objective': 'RMSE',
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': 42,
                'iterations': 100,
                'verbose': False
            },
            'sklearn_gbm': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'subsample': 0.8,
                'random_state': 42
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
        """Generate comprehensive feature set for gradient boosting models"""
        try:
            features_df = df.copy()
            
            # Basic price features
            for window in self.feature_windows:
                if len(features_df) >= window:
                    # Price-based features
                    features_df[f'sma_{window}'] = features_df['close'].rolling(window=window).mean()
                    features_df[f'ema_{window}'] = features_df['close'].ewm(span=window).mean()
                    features_df[f'price_std_{window}'] = features_df['close'].rolling(window=window).std()
                    features_df[f'price_change_{window}'] = features_df['close'].pct_change(window)
                    features_df[f'price_momentum_{window}'] = features_df['close'] / features_df['close'].shift(window) - 1
                    
                    # High-Low features
                    features_df[f'hl_ratio_{window}'] = features_df['high'] / features_df['low']
                    features_df[f'price_position_{window}'] = (features_df['close'] - features_df['low'].rolling(window).min()) / (features_df['high'].rolling(window).max() - features_df['low'].rolling(window).min())
                    
                    # Volume features (if available)
                    if 'volume' in features_df.columns:
                        features_df[f'volume_sma_{window}'] = features_df['volume'].rolling(window=window).mean()
                        features_df[f'volume_ratio_{window}'] = features_df['volume'] / features_df[f'volume_sma_{window}']
                        features_df[f'price_volume_trend_{window}'] = features_df[f'price_change_{window}'] * features_df[f'volume_ratio_{window}']
            
            # Technical indicators
            for period in self.technical_periods:
                if len(features_df) >= period:
                    # RSI
                    features_df[f'rsi_{period}'] = self._calculate_rsi(features_df['close'], period)
                    
                    # Bollinger Bands
                    bb_middle = features_df['close'].rolling(window=period).mean()
                    bb_std = features_df['close'].rolling(window=period).std()
                    features_df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
                    features_df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
                    features_df[f'bb_position_{period}'] = (features_df['close'] - features_df[f'bb_lower_{period}']) / (features_df[f'bb_upper_{period}'] - features_df[f'bb_lower_{period}'])
                    
                    # Stochastic
                    low_min = features_df['low'].rolling(window=period).min()
                    high_max = features_df['high'].rolling(window=period).max()
                    features_df[f'stoch_k_{period}'] = 100 * (features_df['close'] - low_min) / (high_max - low_min)
                    features_df[f'stoch_d_{period}'] = features_df[f'stoch_k_{period}'].rolling(window=3).mean()
            
            # MACD
            exp1 = features_df['close'].ewm(span=12).mean()
            exp2 = features_df['close'].ewm(span=26).mean()
            features_df['macd'] = exp1 - exp2
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # ATR (Average True Range)
            high_low = features_df['high'] - features_df['low']
            high_close = np.abs(features_df['high'] - features_df['close'].shift())
            low_close = np.abs(features_df['low'] - features_df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            features_df['atr'] = true_range.rolling(window=14).mean()
            
            # Williams %R
            features_df['williams_r'] = -100 * (features_df['high'].rolling(window=14).max() - features_df['close']) / (features_df['high'].rolling(window=14).max() - features_df['low'].rolling(window=14).min())
            
            # Commodity Channel Index (CCI)
            tp = (features_df['high'] + features_df['low'] + features_df['close']) / 3
            features_df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
            
            # Advanced features
            # Price acceleration
            features_df['price_acceleration'] = features_df['close'].diff().diff()
            
            # Volatility measures
            features_df['volatility_5'] = features_df['close'].rolling(window=5).std()
            features_df['volatility_20'] = features_df['close'].rolling(window=20).std()
            features_df['volatility_ratio'] = features_df['volatility_5'] / features_df['volatility_20']
            
            # Time-based features
            if hasattr(features_df.index, 'hour'):
                features_df['hour'] = features_df.index.hour
                features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
                features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            
            if hasattr(features_df.index, 'dayofweek'):
                features_df['dayofweek'] = features_df.index.dayofweek
                features_df['dayofweek_sin'] = np.sin(2 * np.pi * features_df['dayofweek'] / 7)
                features_df['dayofweek_cos'] = np.cos(2 * np.pi * features_df['dayofweek'] / 7)
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
                features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag) if 'volume' in features_df.columns else 0
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features_df[f'close_min_{window}'] = features_df['close'].rolling(window=window).min()
                features_df[f'close_max_{window}'] = features_df['close'].rolling(window=window).max()
                features_df[f'close_median_{window}'] = features_df['close'].rolling(window=window).median()
                features_df[f'close_skew_{window}'] = features_df['close'].rolling(window=window).skew()
                features_df[f'close_kurt_{window}'] = features_df['close'].rolling(window=window).kurt()
            
            # Drop rows with NaN values
            features_df = features_df.dropna()
            
            # Remove non-numeric columns except target
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            features_df = features_df[numeric_columns]
            
            return features_df
            
        except Exception as e:
            print(f"Error generating features: {e}")
            return df.copy()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data with target variable"""
        try:
            features_df = self.generate_comprehensive_features(df)
            
            # Create target variable (future price change)
            features_df['target'] = features_df['close'].shift(-self.prediction_horizon) / features_df['close'] - 1
            
            # Remove rows with NaN targets
            features_df = features_df.dropna()
            
            if len(features_df) == 0:
                raise ValueError("No valid data after feature generation")
            
            # Separate features and target
            target_col = 'target'
            feature_cols = [col for col in features_df.columns if col != target_col]
            
            X = features_df[feature_cols].values
            y = features_df[target_col].values
            
            self.feature_names = feature_cols
            
            return X, y, feature_cols
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            # Return minimal features if advanced feature generation fails
            X = df[['open', 'high', 'low', 'close']].values[:-self.prediction_horizon]
            y = (df['close'].shift(-self.prediction_horizon) / df['close'] - 1).dropna().values
            return X, y, ['open', 'high', 'low', 'close']
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all available gradient boosting models"""
        try:
            print("Preparing enhanced training data...")
            X, y, feature_names = self.prepare_training_data(df)
            
            print(f"Training data shape: {X.shape}")
            print(f"Number of features: {len(feature_names)}")
            
            if len(X) < 50:
                return {
                    'success': False,
                    'error': 'Insufficient training data',
                    'available_models': []
                }
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            results = {}
            available_models = []
            
            # Train sklearn models
            print("Training sklearn gradient boosting...")
            sklearn_result = self._train_sklearn_models(X_scaled, y, feature_names)
            if sklearn_result['success']:
                results.update(sklearn_result['models'])
                available_models.extend(sklearn_result['model_names'])
            
            # Train LightGBM
            if LIGHTGBM_AVAILABLE:
                print("Training LightGBM...")
                lgb_result = self._train_lightgbm(X_scaled, y, feature_names)
                if lgb_result['success']:
                    results['lightgbm'] = lgb_result
                    available_models.append('lightgbm')
            
            # Train XGBoost
            if XGBOOST_AVAILABLE:
                print("Training XGBoost...")
                xgb_result = self._train_xgboost(X_scaled, y, feature_names)
                if xgb_result['success']:
                    results['xgboost'] = xgb_result
                    available_models.append('xgboost')
            
            # Train CatBoost
            if CATBOOST_AVAILABLE:
                print("Training CatBoost...")
                cb_result = self._train_catboost(X_scaled, y, feature_names)
                if cb_result['success']:
                    results['catboost'] = cb_result
                    available_models.append('catboost')
            
            self.models = results
            self.is_trained = True
            
            # Calculate ensemble performance
            ensemble_score = self._calculate_ensemble_performance(X_scaled, y)
            
            return {
                'success': True,
                'models': results,
                'available_models': available_models,
                'feature_count': len(feature_names),
                'sample_count': len(X),
                'ensemble_score': ensemble_score,
                'feature_names': feature_names
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'available_models': []
            }
    
    def _train_sklearn_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train sklearn-based models"""
        try:
            models = {}
            model_names = []
            
            # Gradient Boosting
            gbm = GradientBoostingRegressor(**self.model_configs['sklearn_gbm'])
            gbm.fit(X, y)
            
            models['sklearn_gbm'] = {
                'model': gbm,
                'feature_importance': dict(zip(feature_names, gbm.feature_importances_)),
                'score': gbm.score(X, y),
                'oob_score': getattr(gbm, 'oob_improvement_', None)
            }
            model_names.append('sklearn_gbm')
            
            # Random Forest
            rf = RandomForestRegressor(**self.model_configs['random_forest'])
            rf.fit(X, y)
            
            models['random_forest'] = {
                'model': rf,
                'feature_importance': dict(zip(feature_names, rf.feature_importances_)),
                'score': rf.score(X, y),
                'oob_score': getattr(rf, 'oob_score_', None)
            }
            model_names.append('random_forest')
            
            return {
                'success': True,
                'models': models,
                'model_names': model_names
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'models': {},
                'model_names': []
            }
    
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train LightGBM model"""
        try:
            # Create LightGBM dataset
            train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
            
            # Train model
            model = lgb.train(
                self.model_configs['lightgbm'],
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Get feature importance
            feature_importance = dict(zip(feature_names, model.feature_importance()))
            
            # Calculate score
            y_pred = model.predict(X)
            score = r2_score(y, y_pred)
            
            return {
                'success': True,
                'model': model,
                'feature_importance': feature_importance,
                'score': score,
                'num_trees': model.num_trees()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train XGBoost model"""
        try:
            # Create XGBoost regressor
            model = xgb.XGBRegressor(**self.model_configs['xgboost'])
            
            # Train model
            model.fit(X, y, eval_set=[(X, y)], verbose=False)
            
            # Get feature importance
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Calculate score
            score = model.score(X, y)
            
            return {
                'success': True,
                'model': model,
                'feature_importance': feature_importance,
                'score': score,
                'best_iteration': getattr(model, 'best_iteration', None)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_catboost(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train CatBoost model"""
        try:
            # Create CatBoost regressor
            model = cb.CatBoostRegressor(**self.model_configs['catboost'])
            
            # Train model
            model.fit(X, y, verbose=False)
            
            # Get feature importance
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Calculate score
            score = model.score(X, y)
            
            return {
                'success': True,
                'model': model,
                'feature_importance': feature_importance,
                'score': score,
                'tree_count': model.tree_count_
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_ensemble_performance(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate ensemble performance using cross-validation"""
        try:
            if not self.models:
                return 0.0
            
            # Get predictions from all models
            predictions = []
            weights = []
            
            for model_name, model_data in self.models.items():
                if 'model' in model_data:
                    model = model_data['model']
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                        predictions.append(pred)
                        weights.append(model_data.get('score', 0.5))
            
            if not predictions:
                return 0.0
            
            # Calculate weighted ensemble prediction
            predictions = np.array(predictions)
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            ensemble_score = r2_score(y, ensemble_pred)
            
            return max(0.0, ensemble_score)
            
        except Exception as e:
            print(f"Error calculating ensemble performance: {e}")
            return 0.0
    
    def predict_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate ensemble predictions from all trained models"""
        try:
            if not self.is_trained or not self.models:
                return {
                    'success': False,
                    'error': 'Models not trained yet'
                }
            
            # Generate features
            features_df = self.generate_comprehensive_features(df)
            
            # Get feature matrix
            if self.feature_names:
                # Use only features that were used during training
                available_features = [f for f in self.feature_names if f in features_df.columns]
                if len(available_features) < len(self.feature_names) * 0.8:  # At least 80% of features
                    return {
                        'success': False,
                        'error': 'Insufficient features for prediction'
                    }
                X = features_df[available_features].iloc[-1:].values
            else:
                X = features_df[['open', 'high', 'low', 'close']].iloc[-1:].values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            predictions = {}
            weights = []
            pred_values = []
            
            for model_name, model_data in self.models.items():
                if 'model' in model_data:
                    try:
                        model = model_data['model']
                        pred = model.predict(X_scaled)[0]
                        predictions[model_name] = {
                            'prediction': pred,
                            'confidence': model_data.get('score', 0.5)
                        }
                        pred_values.append(pred)
                        weights.append(model_data.get('score', 0.5))
                    except Exception as e:
                        print(f"Error predicting with {model_name}: {e}")
            
            if not pred_values:
                return {
                    'success': False,
                    'error': 'No successful predictions'
                }
            
            # Calculate ensemble prediction
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            ensemble_prediction = np.average(pred_values, weights=weights)
            
            # Calculate prediction confidence
            pred_std = np.std(pred_values)
            avg_confidence = np.average(weights)
            overall_confidence = max(0.1, avg_confidence * (1.0 - min(1.0, pred_std * 10)))
            
            return {
                'success': True,
                'ensemble_prediction': ensemble_prediction,
                'individual_predictions': predictions,
                'confidence': overall_confidence,
                'prediction_std': pred_std,
                'num_models': len(predictions)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get comprehensive feature importance analysis"""
        try:
            if not self.models:
                return {'available': False}
            
            # Aggregate feature importance across models
            feature_importance_agg = {}
            
            for model_name, model_data in self.models.items():
                if 'feature_importance' in model_data:
                    for feature, importance in model_data['feature_importance'].items():
                        if feature not in feature_importance_agg:
                            feature_importance_agg[feature] = []
                        feature_importance_agg[feature].append(importance)
            
            # Calculate average importance
            avg_importance = {}
            for feature, importances in feature_importance_agg.items():
                avg_importance[feature] = np.mean(importances)
            
            # Sort by importance
            sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'available': True,
                'top_features': sorted_importance[:20],
                'all_features': sorted_importance,
                'num_features': len(sorted_importance)
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics"""
        try:
            if not self.models:
                return {'available': False}
            
            performance = {}
            
            for model_name, model_data in self.models.items():
                if 'score' in model_data:
                    performance[model_name] = {
                        'r2_score': model_data['score'],
                        'additional_metrics': {}
                    }
                    
                    # Add model-specific metrics
                    if 'oob_score' in model_data and model_data['oob_score'] is not None:
                        performance[model_name]['additional_metrics']['oob_score'] = model_data['oob_score']
                    
                    if 'num_trees' in model_data:
                        performance[model_name]['additional_metrics']['num_trees'] = model_data['num_trees']
                    
                    if 'best_iteration' in model_data and model_data['best_iteration'] is not None:
                        performance[model_name]['additional_metrics']['best_iteration'] = model_data['best_iteration']
                    
                    if 'tree_count' in model_data:
                        performance[model_name]['additional_metrics']['tree_count'] = model_data['tree_count']
            
            return {
                'available': True,
                'model_performance': performance,
                'best_model': max(performance.items(), key=lambda x: x[1]['r2_score'])[0] if performance else None
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def optimize_hyperparameters(self, df: pd.DataFrame, cv_folds: int = 3) -> Dict[str, Any]:
        """Optimize hyperparameters for all models"""
        try:
            X, y, feature_names = self.prepare_training_data(df)
            X_scaled = self.scaler.fit_transform(X)
            
            optimization_results = {}
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Optimize each available model
            if LIGHTGBM_AVAILABLE:
                lgb_params = {
                    'num_leaves': [20, 31, 50],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'feature_fraction': [0.8, 0.9, 1.0]
                }
                optimization_results['lightgbm'] = self._optimize_lightgbm(X_scaled, y, lgb_params, tscv)
            
            if XGBOOST_AVAILABLE:
                xgb_params = {
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0]
                }
                optimization_results['xgboost'] = self._optimize_xgboost(X_scaled, y, xgb_params, tscv)
            
            return {
                'success': True,
                'optimization_results': optimization_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _optimize_lightgbm(self, X: np.ndarray, y: np.ndarray, param_grid: Dict, cv) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""
        try:
            best_score = -np.inf
            best_params = None
            
            for num_leaves in param_grid['num_leaves']:
                for learning_rate in param_grid['learning_rate']:
                    for feature_fraction in param_grid['feature_fraction']:
                        params = self.model_configs['lightgbm'].copy()
                        params.update({
                            'num_leaves': num_leaves,
                            'learning_rate': learning_rate,
                            'feature_fraction': feature_fraction
                        })
                        
                        scores = []
                        for train_idx, val_idx in cv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            train_data = lgb.Dataset(X_train, label=y_train)
                            model = lgb.train(params, train_data, num_boost_round=50, verbose_eval=False)
                            
                            y_pred = model.predict(X_val)
                            score = r2_score(y_val, y_pred)
                            scores.append(score)
                        
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = params
            
            return {
                'best_params': best_params,
                'best_score': best_score
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _optimize_xgboost(self, X: np.ndarray, y: np.ndarray, param_grid: Dict, cv) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        try:
            best_score = -np.inf
            best_params = None
            
            for max_depth in param_grid['max_depth']:
                for learning_rate in param_grid['learning_rate']:
                    for subsample in param_grid['subsample']:
                        params = self.model_configs['xgboost'].copy()
                        params.update({
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'subsample': subsample
                        })
                        
                        scores = []
                        for train_idx, val_idx in cv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            model = xgb.XGBRegressor(**params)
                            model.fit(X_train, y_train, verbose=False)
                            
                            y_pred = model.predict(X_val)
                            score = r2_score(y_val, y_pred)
                            scores.append(score)
                        
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = params
            
            return {
                'best_params': best_params,
                'best_score': best_score
            }
            
        except Exception as e:
            return {'error': str(e)}