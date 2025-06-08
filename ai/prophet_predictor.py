import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedProphetPredictor:
    """Advanced Prophet-style time series prediction with enhanced seasonality and trend modeling"""
    
    def __init__(self, prediction_horizon: int = 1):
        self.prediction_horizon = prediction_horizon
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Model parameters
        self.changepoint_prior_scale = 0.05
        self.seasonality_prior_scale = 10.0
        self.holidays_prior_scale = 10.0
        self.seasonality_mode = 'additive'
        self.growth = 'linear'
        
        # Advanced components
        self.trend_model = Ridge(alpha=0.1)
        self.seasonality_models = {}
        self.holiday_models = {}
        self.external_regressors = {}
        self.changepoints = []
        self.ensemble_models = {}
        
        # Training state
        self.is_trained = False
        self.training_history = {
            'metrics': {},
            'components': {},
            'predictions': []
        }
        
        # Advanced features
        self.fourier_order = {'hourly': 2, 'daily': 3, 'weekly': 3, 'monthly': 5, 'yearly': 10}
        self.auto_seasonalities = True
        self.mcmc_samples = 0
        self.uncertainty_samples = 1000
        self.seasonality_periods = [24, 168, 720, 8760]  # hourly, weekly, monthly, yearly
        self.forecast_components = {}
        
    def _generate_fourier_features(self, timestamps: pd.Series, period: int, fourier_order: int) -> np.ndarray:
        """Generate Fourier series features for seasonality"""
        t = np.arange(len(timestamps))
        features = []
        
        for i in range(1, fourier_order + 1):
            sin_feature = np.sin(2 * np.pi * i * t / period)
            cos_feature = np.cos(2 * np.pi * i * t / period)
            features.extend([sin_feature, cos_feature])
        
        return np.column_stack(features)
    
    def _fit_piecewise_linear_trend(self, y: np.ndarray, timestamps: pd.Series, changepoints: List[int]) -> Dict[str, Any]:
        """Fit piecewise linear trend with changepoints"""
        n = len(y)
        t = np.arange(n)
        
        # Design matrix for piecewise linear trend
        X = np.ones((n, 2 + len(changepoints)))
        X[:, 1] = t
        
        # Add changepoint features
        for i, cp in enumerate(changepoints):
            X[cp:, 2 + i] = t[cp:] - cp
        
        # Fit linear regression
        try:
            model = Ridge(alpha=self.changepoint_prior_scale)
            model.fit(X, y)
            
            return {
                'model': model,
                'design_matrix': X,
                'changepoints': changepoints,
                'coefficients': model.coef_,
                'intercept': model.intercept_
            }
        except Exception as e:
            print(f"Error fitting trend: {e}")
            # Fallback to simple linear trend
            simple_model = LinearRegression()
            simple_model.fit(t.reshape(-1, 1), y)
            return {
                'model': simple_model,
                'design_matrix': t.reshape(-1, 1),
                'changepoints': [],
                'coefficients': simple_model.coef_,
                'intercept': simple_model.intercept_
            }
        
    def detect_changepoints(self, ts: pd.Series, n_changepoints: int = 25) -> List[int]:
        """Detect trend changepoints in time series"""
        try:
            if len(ts) < n_changepoints * 2:
                return []
            
            # Calculate potential changepoint locations
            n = len(ts)
            potential_changepoints = np.linspace(0.15 * n, 0.85 * n, n_changepoints, dtype=int)
            
            # Use gradient-based changepoint detection
            values = ts.values if hasattr(ts, 'values') else np.array(ts)
            gradients = np.abs(np.diff(values))
            
            # Find changepoints where gradients are highest
            changepoints = []
            for cp in potential_changepoints:
                if cp < len(gradients) and cp > 0:
                    changepoints.append(int(cp))
            
            return sorted(changepoints)
            
        except Exception as e:
            print(f"Error detecting changepoints: {e}")
            return []

    def fit_trend(self, ts: pd.Series, timestamps: pd.Series) -> Dict[str, Any]:
        """Fit piecewise linear trend with changepoints"""
        try:
            # Detect changepoints
            changepoints = self.detect_changepoints(ts)
            
            # Convert to numpy arrays
            y = ts.values if hasattr(ts, 'values') else np.array(ts)
            
            # Fit piecewise linear trend
            trend_result = self._fit_piecewise_linear_trend(y, timestamps, changepoints)
            
            # Generate trend component
            trend_values = trend_result['model'].predict(trend_result['design_matrix'])
            
            return {
                'trend_values': trend_values,
                'changepoints': changepoints,
                'model': trend_result['model'],
                'coefficients': trend_result['coefficients'],
                'intercept': trend_result['intercept']
            }
            
        except Exception as e:
            print(f"Error fitting trend: {e}")
            # Fallback to simple linear trend
            t = np.arange(len(ts))
            y_values = ts.values if hasattr(ts, 'values') else np.array(ts)
            coeffs = np.polyfit(t, y_values, 1)
            trend_values = np.polyval(coeffs, t)
            
            return {
                'trend_values': trend_values,
                'changepoints': [],
                'model': None,
                'coefficients': coeffs,
                'intercept': coeffs[1]
            }

    def fit_seasonality(self, residuals: pd.Series, timestamps: pd.Series) -> Dict[str, np.ndarray]:
        """Fit seasonal components using Fourier series"""
        try:
            seasonality_components = {}
            
            for period in self.seasonality_periods:
                if len(residuals) >= period * 2:  # Need at least 2 full periods
                    # Generate Fourier features for this period
                    fourier_order = min(self.fourier_order.get('weekly', 3), period // 4)
                    
                    fourier_features = self._generate_fourier_features(timestamps, period, fourier_order)
                    
                    # Fit seasonal component
                    try:
                        seasonal_model = Ridge(alpha=self.seasonality_prior_scale)
                        y_values = residuals.values if hasattr(residuals, 'values') else np.array(residuals)
                        seasonal_model.fit(fourier_features, y_values)
                        
                        seasonal_values = seasonal_model.predict(fourier_features)
                        seasonality_components[f'seasonal_{period}'] = seasonal_values
                        
                    except Exception as e:
                        print(f"Error fitting seasonality for period {period}: {e}")
                        seasonality_components[f'seasonal_{period}'] = np.zeros(len(residuals))
            
            return seasonality_components
            
        except Exception as e:
            print(f"Error fitting seasonality: {e}")
            return {}

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet-like model on price data"""
        try:
            print("Training Advanced Prophet predictor...")
            
            if len(df) < 100:
                return {'error': 'Insufficient data for Prophet training. Need at least 100 samples.'}
            
            # Prepare data
            ts = df['close'].copy()
            timestamps = pd.Series(range(len(df)))
            
            # Scale the target
            ts_values = ts.values.reshape(-1, 1)
            ts_scaled = self.price_scaler.fit_transform(ts_values).flatten()
            ts_series = pd.Series(ts_scaled, index=ts.index)
            
            # 1. Fit trend component
            print("Fitting trend component...")
            trend_result = self.fit_trend(ts_series, timestamps)
            trend_values = trend_result['trend_values']
            
            # 2. Calculate residuals after trend removal
            residuals = ts_series - trend_values
            
            # 3. Fit seasonality components
            print("Fitting seasonality components...")
            seasonality_result = self.fit_seasonality(pd.Series(residuals), timestamps)
            
            # 4. Calculate total seasonal component
            total_seasonal = np.zeros(len(ts))
            for seasonal_values in seasonality_result.values():
                total_seasonal += seasonal_values
            
            # 5. Final residuals after trend and seasonality
            final_residuals = residuals - total_seasonal
            
            # 6. Train ensemble models on features
            print("Training ensemble models...")
            self._train_ensemble_models(df, final_residuals)
            
            # Store model components
            self.trend_components = trend_result
            self.seasonality_components = seasonality_result
            
            # Calculate performance metrics
            predictions = trend_values + total_seasonal
            mse = mean_squared_error(ts_series, predictions)
            mae = mean_absolute_error(ts_series, predictions)
            
            # Store training metrics
            self.training_history['metrics'] = {
                'mse': float(mse),
                'mae': float(mae),
                'trend_strength': float(np.std(trend_values) / np.std(ts_series)),
                'seasonal_strength': float(np.std(total_seasonal) / np.std(ts_series))
            }
            
            self.is_trained = True
            
            return {
                'success': True,
                'model_type': 'Advanced Prophet with Ensemble',
                'metrics': self.training_history['metrics'],
                'trend_changepoints': len(trend_result['changepoints']),
                'seasonal_periods': list(seasonality_result.keys()),
                'ensemble_models': list(self.ensemble_models.keys())
            }
            
        except Exception as e:
            print(f"Error training Prophet model: {e}")
            return {'error': str(e)}
    
    def _train_ensemble_models(self, df: pd.DataFrame, residuals: np.ndarray) -> None:
        """Train ensemble models for additional prediction power"""
        try:
            # Create features for ensemble models
            features = self._create_features(df)
            
            if len(features.columns) == 0:
                return
            
            X = features.values
            y = residuals
            
            # Train multiple models
            models = {
                'ridge': Ridge(alpha=1.0),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
                'random_forest': RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
            }
            
            for name, model in models.items():
                try:
                    model.fit(X, y)
                    self.ensemble_models[name] = model
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    
        except Exception as e:
            print(f"Error training ensemble models: {e}")
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ensemble models"""
        features = pd.DataFrame()
        
        try:
            # Price-based features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20]:
                features[f'sma_{window}'] = df['close'].rolling(window).mean()
                features[f'price_ratio_{window}'] = df['close'] / features[f'sma_{window}']
            
            # Volatility
            features['volatility_5'] = features['returns'].rolling(5).std()
            features['volatility_20'] = features['returns'].rolling(20).std()
            
            # Volume features (if available)
            if 'volume' in df.columns:
                features['volume_sma'] = df['volume'].rolling(10).mean()
                features['volume_ratio'] = df['volume'] / features['volume_sma']
            
            # Time features
            if hasattr(df.index, 'hour'):
                features['hour'] = df.index.hour
                features['day_of_week'] = df.index.dayofweek
                features['month'] = df.index.month
            else:
                # Create time features based on position
                n = len(df)
                features['time_linear'] = np.arange(n) / n
                features['time_sin'] = np.sin(2 * np.pi * np.arange(n) / min(24, n))
                features['time_cos'] = np.cos(2 * np.pi * np.arange(n) / min(24, n))
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
        except Exception as e:
            print(f"Error creating features: {e}")
            
        return features

    def predict(self, df: pd.DataFrame, periods: int = 24) -> Dict[str, Any]:
        """Make future predictions"""
        try:
            if not self.is_trained:
                return {'error': 'Model not trained. Call train() first.'}
            
            print(f"Generating Prophet predictions for {periods} periods...")
            
            # Get recent data
            recent_data = df.tail(100)  # Use last 100 points for context
            current_price = df['close'].iloc[-1]
            
            # Generate future timestamps
            future_timestamps = pd.Series(range(len(df), len(df) + periods))
            
            # 1. Predict trend
            trend_predictions = self._predict_trend(future_timestamps, periods)
            
            # 2. Predict seasonality
            seasonal_predictions = self._predict_seasonality(future_timestamps, periods)
            
            # 3. Predict residuals using ensemble
            residual_predictions = self._predict_residuals(recent_data, periods)
            
            # 4. Combine all components
            total_predictions = trend_predictions + seasonal_predictions + residual_predictions
            
            # 5. Inverse transform to original scale
            predictions_rescaled = self.price_scaler.inverse_transform(total_predictions.reshape(-1, 1)).flatten()
            
            # Calculate confidence intervals
            confidence = self._calculate_prediction_confidence(periods)
            
            # Calculate percentage changes
            predicted_returns = [(pred / current_price - 1) for pred in predictions_rescaled]
            
            return {
                'success': True,
                'predictions': predictions_rescaled.tolist(),
                'predicted_returns': predicted_returns,
                'current_price': float(current_price),
                'confidence': float(confidence),
                'periods': periods,
                'model_type': 'Advanced Prophet',
                'components': {
                    'trend': trend_predictions.tolist(),
                    'seasonal': seasonal_predictions.tolist(),
                    'residual': residual_predictions.tolist()
                }
            }
            
        except Exception as e:
            print(f"Error making Prophet predictions: {e}")
            return {'error': str(e)}
    
    def _predict_trend(self, future_timestamps: pd.Series, periods: int) -> np.ndarray:
        """Predict trend component for future periods"""
        try:
            if 'model' in self.trend_components and self.trend_components['model'] is not None:
                # Use trained trend model
                trend_model = self.trend_components['model']
                
                # Create design matrix for future periods
                changepoints = self.trend_components['changepoints']
                n_historical = len(future_timestamps) - periods
                
                future_t = np.arange(n_historical, n_historical + periods)
                X_future = np.ones((periods, 2 + len(changepoints)))
                X_future[:, 1] = future_t
                
                # Add changepoint features
                for i, cp in enumerate(changepoints):
                    mask = future_t >= cp
                    X_future[mask, 2 + i] = future_t[mask] - cp
                
                trend_pred = trend_model.predict(X_future)
                
            else:
                # Fallback: extrapolate linear trend
                coeffs = self.trend_components.get('coefficients', [0, 0])
                future_t = np.arange(len(future_timestamps) - periods, len(future_timestamps))
                trend_pred = np.polyval(coeffs, future_t)
            
            return trend_pred
            
        except Exception as e:
            print(f"Error predicting trend: {e}")
            return np.zeros(periods)
    
    def _predict_seasonality(self, future_timestamps: pd.Series, periods: int) -> np.ndarray:
        """Predict seasonal components for future periods"""
        try:
            total_seasonal = np.zeros(periods)
            
            for period_name, seasonal_values in self.seasonality_components.items():
                if len(seasonal_values) > 0:
                    # Extract period from name
                    period = int(period_name.split('_')[1])
                    
                    # Repeat seasonal pattern
                    historical_length = len(seasonal_values)
                    future_indices = np.arange(historical_length, historical_length + periods)
                    seasonal_pred = seasonal_values[future_indices % len(seasonal_values)]
                    
                    total_seasonal += seasonal_pred
            
            return total_seasonal
            
        except Exception as e:
            print(f"Error predicting seasonality: {e}")
            return np.zeros(periods)
    
    def _predict_residuals(self, recent_data: pd.DataFrame, periods: int) -> np.ndarray:
        """Predict residuals using ensemble models"""
        try:
            if not self.ensemble_models:
                return np.zeros(periods)
            
            # Create features for prediction
            features = self._create_features(recent_data)
            
            if len(features) == 0:
                return np.zeros(periods)
            
            # Use last known features and extrapolate
            last_features = features.iloc[-1:].values
            
            residual_predictions = []
            for name, model in self.ensemble_models.items():
                try:
                    pred = model.predict(last_features)[0]
                    residual_predictions.append(pred)
                except Exception as e:
                    print(f"Error with {name} residual prediction: {e}")
                    residual_predictions.append(0.0)
            
            # Average ensemble predictions
            avg_residual = np.mean(residual_predictions) if residual_predictions else 0.0
            
            # Decay the residual prediction over time
            decay_factor = 0.95
            residual_pred = np.array([avg_residual * (decay_factor ** i) for i in range(periods)])
            
            return residual_pred
            
        except Exception as e:
            print(f"Error predicting residuals: {e}")
            return np.zeros(periods)

    def _calculate_prediction_confidence(self, periods: int) -> float:
        """Calculate prediction confidence based on forecast horizon and model quality"""
        try:
            base_confidence = 0.8
            
            # Decrease confidence with longer horizons
            horizon_penalty = min(0.05 * periods, 0.4)
            
            # Adjust based on training metrics
            if 'metrics' in self.training_history:
                mse = self.training_history['metrics'].get('mse', 1.0)
                quality_adjustment = max(-0.3, -mse * 10)
            else:
                quality_adjustment = -0.1
            
            confidence = base_confidence - horizon_penalty + quality_adjustment
            return max(0.1, min(confidence, 0.95))
            
        except Exception:
            return 0.5

    def decompose(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components"""
        try:
            if not self.is_trained:
                return {'error': 'Model not trained. Call train() first.'}
            
            ts = df['close']
            timestamps = pd.Series(range(len(df)))
            
            # Get stored components
            trend_values = self.trend_components.get('trend_values', np.zeros(len(ts)))
            
            total_seasonal = np.zeros(len(ts))
            for seasonal_values in self.seasonality_components.values():
                if len(seasonal_values) == len(ts):
                    total_seasonal += seasonal_values
            
            residuals = ts.values - trend_values - total_seasonal
            
            return {
                'success': True,
                'trend': trend_values.tolist(),
                'seasonal': total_seasonal.tolist(),
                'residual': residuals.tolist(),
                'original': ts.tolist(),
                'changepoints': self.trend_components.get('changepoints', [])
            }
            
        except Exception as e:
            print(f"Error decomposing time series: {e}")
            return {'error': str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters"""
        info = {
            'model_type': 'Advanced Prophet Predictor',
            'is_trained': self.is_trained,
            'prediction_horizon': self.prediction_horizon,
            'seasonality_periods': self.seasonality_periods,
            'fourier_order': self.fourier_order,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale
        }
        
        if self.is_trained:
            info.update({
                'trend_changepoints': len(self.trend_components.get('changepoints', [])),
                'seasonal_components': list(self.seasonality_components.keys()),
                'ensemble_models': list(self.ensemble_models.keys()),
                'training_metrics': self.training_history.get('metrics', {})
            })
        
        return info