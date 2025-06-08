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
        
    def detect_changepoints(self, ts: pd.Series, n_changepoints: int = 25) -> List[int]:
        """Detect trend changepoints in time series"""
        try:
            if len(ts) < n_changepoints * 2:
                return []
            
            # Calculate rolling statistics to detect changes
            window = max(5, len(ts) // n_changepoints)
            rolling_mean = ts.rolling(window=window, center=True).mean()
            rolling_std = ts.rolling(window=window, center=True).std()
            
            # Calculate rate of change
            rate_change = rolling_mean.diff().abs()
            volatility_change = rolling_std.diff().abs()
            
            # Combine signals
            changepoint_signal = rate_change + volatility_change
            changepoint_signal = changepoint_signal.fillna(0)
            
            # Find peaks in changepoint signal
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(changepoint_signal, height=np.percentile(changepoint_signal, 70))
            
            # Limit number of changepoints
            if len(peaks) > n_changepoints:
                peak_heights = changepoint_signal.iloc[peaks]
                top_peaks = peak_heights.nlargest(n_changepoints)
                peaks = top_peaks.index.tolist()
            
            return sorted(peaks)
            
        except Exception as e:
            print(f"Error detecting changepoints: {e}")
            return []
    
    def fit_trend(self, ts: pd.Series, timestamps: pd.Series) -> Dict[str, Any]:
        """Fit piecewise linear trend with changepoints"""
        try:
            # Convert timestamps to numeric
            time_numeric = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600  # hours
            
            # Detect changepoints
            changepoints = self.detect_changepoints(ts)
            self.trend_changepoints = [timestamps.iloc[cp] for cp in changepoints]
            
            # Create trend features
            X_trend = np.column_stack([time_numeric])
            
            # Add changepoint features
            for cp in changepoints:
                cp_feature = np.maximum(0, time_numeric - time_numeric.iloc[cp])
                X_trend = np.column_stack([X_trend, cp_feature])
            
            # Fit trend model
            self.trend_model.fit(X_trend, ts.values)
            trend_pred = self.trend_model.predict(X_trend)
            
            return {
                'trend_pred': trend_pred,
                'changepoints': len(changepoints),
                'trend_r2': self.trend_model.score(X_trend, ts.values)
            }
            
        except Exception as e:
            print(f"Error fitting trend: {e}")
            return {'trend_pred': ts.values, 'changepoints': 0, 'trend_r2': 0}
    
    def fit_seasonality(self, residuals: pd.Series, timestamps: pd.Series) -> Dict[str, np.ndarray]:
        """Fit seasonal components using Fourier series"""
        seasonality_components = {}
        
        try:
            time_numeric = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600
            
            for period in self.seasonality_periods:
                if len(residuals) < period * 2:
                    continue
                
                # Create Fourier features for this period
                n_fourier = min(10, period // 2)
                fourier_features = []
                
                for n in range(1, n_fourier + 1):
                    sin_feature = np.sin(2 * np.pi * n * time_numeric / period)
                    cos_feature = np.cos(2 * np.pi * n * time_numeric / period)
                    fourier_features.extend([sin_feature, cos_feature])
                
                if fourier_features:
                    X_seasonal = np.column_stack(fourier_features)
                    
                    # Fit linear regression for this seasonal component
                    seasonal_model = LinearRegression()
                    seasonal_model.fit(X_seasonal, residuals.values)
                    seasonal_pred = seasonal_model.predict(X_seasonal)
                    
                    # Store model and prediction
                    self.seasonality_models[period] = {
                        'model': seasonal_model,
                        'fourier_terms': n_fourier
                    }
                    seasonality_components[f'seasonal_{period}'] = seasonal_pred
                    
                    # Update residuals by removing this seasonal component
                    residuals = residuals - seasonal_pred
                    
        except Exception as e:
            print(f"Error fitting seasonality: {e}")
        
        return seasonality_components
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet-like model on price data"""
        try:
            # Prepare data
            if 'timestamp' not in df.columns:
                df = df.reset_index()
                if 'index' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['index'])
                else:
                    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
            
            ts = df['close'].copy()
            timestamps = pd.to_datetime(df['timestamp'])
            
            if len(ts) < 100:
                return {'success': False, 'error': 'Insufficient data for Prophet training'}
            
            # Remove any NaN values
            valid_mask = ~(ts.isna() | timestamps.isna())
            ts = ts[valid_mask]
            timestamps = timestamps[valid_mask]
            
            # Log transform to stabilize variance
            ts_log = np.log(ts + 1e-8)
            
            # 1. Fit trend component
            trend_result = self.fit_trend(ts_log, timestamps)
            trend_pred = trend_result['trend_pred']
            
            # 2. Calculate residuals after trend removal
            residuals = pd.Series(ts_log.values - trend_pred, index=ts.index)
            
            # 3. Fit seasonal components
            seasonality_components = self.fit_seasonality(residuals, timestamps)
            
            # 4. Calculate final residuals (noise)
            total_seasonal = np.zeros(len(residuals))
            for seasonal_pred in seasonality_components.values():
                total_seasonal += seasonal_pred
            
            final_residuals = residuals.values - total_seasonal
            noise_std = np.std(final_residuals)
            
            # Store components for forecasting
            self.forecast_components = {
                'trend': trend_pred,
                'seasonality': seasonality_components,
                'noise_std': noise_std,
                'timestamps': timestamps,
                'log_transform': True
            }
            
            self.is_trained = True
            
            # Calculate training metrics
            total_pred = trend_pred + total_seasonal
            total_pred_original = np.exp(total_pred) - 1e-8
            
            mse = mean_squared_error(ts.values, total_pred_original)
            mae = mean_absolute_error(ts.values, total_pred_original)
            
            return {
                'success': True,
                'mse': mse,
                'mae': mae,
                'trend_r2': trend_result['trend_r2'],
                'changepoints': trend_result['changepoints'],
                'seasonal_components': len(seasonality_components),
                'noise_std': noise_std,
                'training_samples': len(ts)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict(self, df: pd.DataFrame, periods: int = 24) -> Dict[str, Any]:
        """Make future predictions"""
        try:
            if not self.is_trained:
                return {'success': False, 'error': 'Model not trained'}
            
            # Get last timestamp and create future timestamps
            if 'timestamp' not in df.columns:
                df = df.reset_index()
                if 'index' in df.columns:
                    last_timestamp = pd.to_datetime(df['index'].iloc[-1])
                else:
                    last_timestamp = pd.Timestamp.now()
            else:
                last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
            
            # Create future timestamps
            future_timestamps = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=periods,
                freq='H'
            )
            
            # Combine historical and future timestamps
            all_timestamps = pd.concat([
                pd.to_datetime(self.forecast_components['timestamps']),
                future_timestamps
            ])
            
            # Convert to numeric time
            time_numeric = (all_timestamps - all_timestamps.iloc[0]).dt.total_seconds() / 3600
            future_time_numeric = time_numeric[-periods:]
            
            # 1. Predict trend for future periods
            X_trend = np.column_stack([future_time_numeric])
            
            # Add changepoint features
            historical_timestamps = self.forecast_components['timestamps']
            for cp_timestamp in self.trend_changepoints:
                cp_hours = (cp_timestamp - all_timestamps.iloc[0]).total_seconds() / 3600
                cp_feature = np.maximum(0, future_time_numeric - cp_hours)
                X_trend = np.column_stack([X_trend, cp_feature])
            
            trend_pred = self.trend_model.predict(X_trend)
            
            # 2. Predict seasonal components
            total_seasonal = np.zeros(periods)
            
            for period, model_info in self.seasonality_models.items():
                model = model_info['model']
                n_fourier = model_info['fourier_terms']
                
                # Create Fourier features for future periods
                fourier_features = []
                for n in range(1, n_fourier + 1):
                    sin_feature = np.sin(2 * np.pi * n * future_time_numeric / period)
                    cos_feature = np.cos(2 * np.pi * n * future_time_numeric / period)
                    fourier_features.extend([sin_feature, cos_feature])
                
                if fourier_features:
                    X_seasonal = np.column_stack(fourier_features)
                    seasonal_pred = model.predict(X_seasonal)
                    total_seasonal += seasonal_pred
            
            # 3. Combine components
            log_pred = trend_pred + total_seasonal
            
            # Add uncertainty
            noise_std = self.forecast_components['noise_std']
            
            # Transform back to original scale
            pred_mean = np.exp(log_pred) - 1e-8
            
            # Calculate confidence intervals (approximate)
            pred_std = pred_mean * noise_std  # Approximate std in original scale
            pred_lower = pred_mean - 1.96 * pred_std
            pred_upper = pred_mean + 1.96 * pred_std
            
            # Calculate trend direction and strength
            current_price = df['close'].iloc[-1]
            price_change = (pred_mean[0] - current_price) / current_price
            
            # Calculate confidence based on model fit quality
            confidence = self._calculate_prediction_confidence(periods)
            
            return {
                'success': True,
                'predictions': pred_mean.tolist(),
                'lower_bound': pred_lower.tolist(),
                'upper_bound': pred_upper.tolist(),
                'timestamps': future_timestamps.tolist(),
                'next_price': pred_mean[0],
                'price_change': price_change,
                'confidence': confidence,
                'trend_direction': 'up' if price_change > 0 else 'down',
                'model_type': 'Prophet'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_prediction_confidence(self, periods: int) -> float:
        """Calculate prediction confidence based on forecast horizon and model quality"""
        try:
            # Base confidence
            base_confidence = 0.7
            
            # Reduce confidence for longer forecasts
            horizon_penalty = min(0.3, periods * 0.01)
            
            # Adjust based on noise level
            noise_penalty = min(0.2, self.forecast_components['noise_std'])
            
            confidence = base_confidence - horizon_penalty - noise_penalty
            return max(0.1, min(0.95, confidence))
            
        except:
            return 0.5
    
    def decompose(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            components = self.forecast_components
            
            trend = components['trend']
            seasonality = components['seasonality']
            
            # Combine seasonal components
            total_seasonal = np.zeros(len(trend))
            for seasonal_pred in seasonality.values():
                total_seasonal += seasonal_pred
            
            # Calculate residuals
            ts_log = np.log(df['close'].values + 1e-8)
            residuals = ts_log - trend - total_seasonal
            
            # Transform back to original scale
            trend_original = np.exp(trend) - 1e-8
            seasonal_original = np.exp(total_seasonal) - 1
            
            return {
                'trend': trend_original.tolist(),
                'seasonal': seasonal_original.tolist(),
                'residual': residuals.tolist(),
                'timestamps': components['timestamps'].tolist(),
                'changepoints': [str(cp) for cp in self.trend_changepoints]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters"""
        if not self.is_trained:
            return {'trained': False}
        
        return {
            'trained': True,
            'seasonality_periods': self.seasonality_periods,
            'changepoints_count': len(self.trend_changepoints),
            'seasonal_components': len(self.seasonality_models),
            'noise_std': self.forecast_components.get('noise_std', 0)
        }