import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from .advanced_ml_pipeline import AdvancedMLPipeline
from .transformer_ensemble import TransformerEnsemble
from .lstm_predictor import AdvancedLSTMPredictor
import warnings
warnings.filterwarnings('ignore')

class FreqAILevelPipeline:
    """FreqAI-level ML pipeline combining multiple advanced models"""
    
    def __init__(self):
        self.ml_pipeline = AdvancedMLPipeline(prediction_horizon=1)
        self.transformer_ensemble = TransformerEnsemble(sequence_length=60)
        self.lstm_predictor = AdvancedLSTMPredictor(lookback_window=60)
        
        self.ensemble_weights = {
            'ml_pipeline': 0.4,
            'transformer': 0.35,
            'lstm': 0.25
        }
        
        self.is_trained = False
        self.training_history = {}
        self.performance_metrics = {}
        
    def auto_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive automatic feature engineering"""
        try:
            features_df = df.copy()
            
            # Price-based features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            
            # Multi-timeframe moving averages
            ma_periods = [5, 10, 20, 50, 100, 200]
            for period in ma_periods:
                if len(features_df) > period:
                    features_df[f'sma_{period}'] = features_df['close'].rolling(period).mean()
                    features_df[f'ema_{period}'] = features_df['close'].ewm(span=period).mean()
                    features_df[f'price_sma_ratio_{period}'] = features_df['close'] / features_df[f'sma_{period}']
                    features_df[f'sma_slope_{period}'] = features_df[f'sma_{period}'].diff()
            
            # Volatility features
            vol_periods = [10, 20, 50]
            for period in vol_periods:
                features_df[f'volatility_{period}'] = features_df['returns'].rolling(period).std()
                features_df[f'volatility_ratio_{period}'] = (
                    features_df[f'volatility_{period}'] / features_df[f'volatility_{period}'].rolling(period*2).mean()
                )
            
            # Technical indicators
            rsi_periods = [14, 21, 28]
            for period in rsi_periods:
                delta = features_df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss.replace(0, 1)
                features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_periods = [20, 50]
            for period in bb_periods:
                sma = features_df['close'].rolling(period).mean()
                std = features_df['close'].rolling(period).std()
                features_df[f'bb_upper_{period}'] = sma + (std * 2)
                features_df[f'bb_lower_{period}'] = sma - (std * 2)
                features_df[f'bb_position_{period}'] = (
                    (features_df['close'] - features_df[f'bb_lower_{period}']) /
                    (features_df[f'bb_upper_{period}'] - features_df[f'bb_lower_{period}'])
                )
                features_df[f'bb_width_{period}'] = (
                    (features_df[f'bb_upper_{period}'] - features_df[f'bb_lower_{period}']) / sma
                )
            
            # MACD variants
            macd_configs = [(12, 26, 9), (5, 35, 5), (19, 39, 9)]
            for fast, slow, signal in macd_configs:
                ema_fast = features_df['close'].ewm(span=fast).mean()
                ema_slow = features_df['close'].ewm(span=slow).mean()
                macd = ema_fast - ema_slow
                macd_signal = macd.ewm(span=signal).mean()
                features_df[f'macd_{fast}_{slow}'] = macd
                features_df[f'macd_signal_{fast}_{slow}'] = macd_signal
                features_df[f'macd_histogram_{fast}_{slow}'] = macd - macd_signal
            
            # Volume features (if available)
            if 'volume' in features_df.columns:
                vol_ma_periods = [10, 20, 50]
                for period in vol_ma_periods:
                    features_df[f'volume_sma_{period}'] = features_df['volume'].rolling(period).mean()
                    features_df[f'volume_ratio_{period}'] = features_df['volume'] / features_df[f'volume_sma_{period}']
                
                # Volume price trend
                features_df['vpt'] = (features_df['volume'] * features_df['returns']).cumsum()
                features_df['vpt_sma'] = features_df['vpt'].rolling(20).mean()
            
            # Statistical features
            stat_periods = [20, 50, 100]
            for period in stat_periods:
                if len(features_df) > period:
                    # Skewness and Kurtosis
                    features_df[f'skewness_{period}'] = features_df['returns'].rolling(period).skew()
                    features_df[f'kurtosis_{period}'] = features_df['returns'].rolling(period).kurt()
                    
                    # Quantiles
                    features_df[f'price_quantile_25_{period}'] = features_df['close'].rolling(period).quantile(0.25)
                    features_df[f'price_quantile_75_{period}'] = features_df['close'].rolling(period).quantile(0.75)
                    
                    # Price position within range
                    rolling_min = features_df['close'].rolling(period).min()
                    rolling_max = features_df['close'].rolling(period).max()
                    features_df[f'price_position_{period}'] = (
                        (features_df['close'] - rolling_min) / (rolling_max - rolling_min)
                    )
            
            # Momentum indicators
            momentum_periods = [5, 10, 20, 50]
            for period in momentum_periods:
                features_df[f'momentum_{period}'] = features_df['close'] / features_df['close'].shift(period)
                features_df[f'roc_{period}'] = (
                    (features_df['close'] - features_df['close'].shift(period)) /
                    features_df['close'].shift(period) * 100
                )
            
            # Trend strength indicators
            trend_periods = [10, 20, 50]
            for period in trend_periods:
                # Linear regression slope
                def calculate_slope(series):
                    if len(series) < 2 or series.isna().all():
                        return np.nan
                    x = np.arange(len(series))
                    valid_mask = ~series.isna()
                    if valid_mask.sum() < 2:
                        return np.nan
                    try:
                        return np.polyfit(x[valid_mask], series[valid_mask], 1)[0]
                    except:
                        return np.nan
                
                features_df[f'trend_slope_{period}'] = features_df['close'].rolling(period).apply(calculate_slope)
            
            # Cross-asset correlation features (autocorrelation)
            corr_lags = [1, 2, 5, 10]
            for lag in corr_lags:
                features_df[f'autocorr_lag_{lag}'] = features_df['returns'].rolling(50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag and not x.isna().all() else np.nan
                )
            
            # Market microstructure features
            features_df['hl_ratio'] = (features_df['high'] - features_df['low']) / features_df['close']
            features_df['oc_ratio'] = (features_df['open'] - features_df['close']) / features_df['close']
            features_df['price_efficiency'] = features_df['close'] / ((features_df['high'] + features_df['low']) / 2)
            
            # Time-based features
            if hasattr(features_df.index, 'hour'):
                features_df['hour_sin'] = np.sin(2 * np.pi * features_df.index.hour / 24)
                features_df['hour_cos'] = np.cos(2 * np.pi * features_df.index.hour / 24)
                features_df['day_sin'] = np.sin(2 * np.pi * features_df.index.dayofweek / 7)
                features_df['day_cos'] = np.cos(2 * np.pi * features_df.index.dayofweek / 7)
            
            # Regime indicators
            features_df['high_vol_regime'] = (
                features_df['volatility_20'] > features_df['volatility_20'].rolling(100).quantile(0.8)
            ).astype(int)
            
            features_df['trending_regime'] = (
                abs(features_df['trend_slope_20']) > features_df['trend_slope_20'].rolling(100).std()
            ).astype(int)
            
            # Clean features
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return features_df
            
        except Exception as e:
            print(f"Error in auto feature engineering: {e}")
            return df
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models in the FreqAI pipeline"""
        try:
            print("Starting FreqAI-level training...")
            
            # Auto feature engineering
            engineered_df = self.auto_feature_engineering(df)
            print(f"Generated {len(engineered_df.columns)} features")
            
            results = {}
            
            # Train ML pipeline (LightGBM, XGBoost, CatBoost, RandomForest)
            print("Training gradient boosting models...")
            ml_result = self.ml_pipeline.train_all_models(engineered_df)
            results['ml_pipeline'] = ml_result
            
            # Train Transformer ensemble
            print("Training Transformer ensemble...")
            transformer_result = self.transformer_ensemble.train(engineered_df, epochs=30)
            results['transformer'] = transformer_result
            
            # Train LSTM predictor
            print("Training LSTM predictor...")
            lstm_result = self.lstm_predictor.train(engineered_df)
            results['lstm'] = lstm_result
            
            # Store training history
            self.training_history = results
            
            # Calculate ensemble performance
            ensemble_performance = self._evaluate_ensemble_performance(engineered_df)
            results['ensemble_performance'] = ensemble_performance
            
            self.is_trained = True
            
            return {
                'success': True,
                'results': results,
                'feature_count': len(engineered_df.columns),
                'sample_count': len(engineered_df)
            }
            
        except Exception as e:
            print(f"Error in FreqAI training: {e}")
            return {'error': str(e)}
    
    def predict_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate ensemble predictions from all models"""
        try:
            if not self.is_trained:
                return {'error': 'Models not trained'}
            
            # Auto feature engineering
            engineered_df = self.auto_feature_engineering(df)
            
            predictions = {}
            confidences = {}
            
            # Get ML pipeline prediction
            try:
                ml_pred = self.ml_pipeline.predict_ensemble(engineered_df)
                if ml_pred.get('success'):
                    predictions['ml_pipeline'] = ml_pred['ensemble_prediction']
                    confidences['ml_pipeline'] = ml_pred['confidence']
                else:
                    predictions['ml_pipeline'] = 0.0
                    confidences['ml_pipeline'] = 0.1
            except Exception as e:
                print(f"ML pipeline prediction error: {e}")
                predictions['ml_pipeline'] = 0.0
                confidences['ml_pipeline'] = 0.1
            
            # Get Transformer prediction
            try:
                transformer_pred = self.transformer_ensemble.predict(engineered_df)
                if transformer_pred.get('success'):
                    predictions['transformer'] = transformer_pred['prediction']
                    confidences['transformer'] = transformer_pred['confidence']
                else:
                    predictions['transformer'] = 0.0
                    confidences['transformer'] = 0.1
            except Exception as e:
                print(f"Transformer prediction error: {e}")
                predictions['transformer'] = 0.0
                confidences['transformer'] = 0.1
            
            # Get LSTM prediction
            try:
                lstm_pred = self.lstm_predictor.predict(engineered_df)
                if lstm_pred.get('success'):
                    predictions['lstm'] = lstm_pred['prediction']
                    confidences['lstm'] = lstm_pred['confidence']
                else:
                    predictions['lstm'] = 0.0
                    confidences['lstm'] = 0.1
            except Exception as e:
                print(f"LSTM prediction error: {e}")
                predictions['lstm'] = 0.0
                confidences['lstm'] = 0.1
            
            # Calculate weighted ensemble prediction
            total_weight = 0
            weighted_prediction = 0
            
            for model_name, pred in predictions.items():
                weight = self.ensemble_weights.get(model_name, 0) * confidences.get(model_name, 0.1)
                weighted_prediction += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                final_prediction = weighted_prediction / total_weight
            else:
                final_prediction = 0.0
            
            # Calculate ensemble confidence
            avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.1
            
            # Convert to price prediction
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + final_prediction)
            
            return {
                'success': True,
                'ensemble_prediction': final_prediction,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'ensemble_confidence': avg_confidence,
                'individual_predictions': predictions,
                'model_confidences': confidences,
                'model_weights': self.ensemble_weights
            }
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return {'error': str(e)}
    
    def _evaluate_ensemble_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate ensemble performance using walk-forward analysis"""
        try:
            if len(df) < 200:
                return {'error': 'Insufficient data for evaluation'}
            
            # Use last 100 samples for evaluation
            eval_df = df.tail(100)
            
            predictions = []
            actuals = []
            
            # Simple walk-forward evaluation
            for i in range(50, len(eval_df) - 1):
                train_data = eval_df.iloc[:i]
                
                # Get prediction for next period
                try:
                    pred_result = self.predict_ensemble(train_data)
                    if pred_result.get('success'):
                        predictions.append(pred_result['ensemble_prediction'])
                        
                        # Get actual return
                        current_price = eval_df['close'].iloc[i]
                        next_price = eval_df['close'].iloc[i + 1]
                        actual_return = (next_price - current_price) / current_price
                        actuals.append(actual_return)
                except:
                    continue
            
            if len(predictions) == 0:
                return {'error': 'No valid predictions for evaluation'}
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate metrics
            mse = np.mean((predictions - actuals) ** 2)
            mae = np.mean(np.abs(predictions - actuals))
            correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
            
            # Direction accuracy
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            direction_accuracy = np.mean(pred_direction == actual_direction)
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'correlation': float(correlation),
                'direction_accuracy': float(direction_accuracy),
                'num_predictions': len(predictions)
            }
            
        except Exception as e:
            print(f"Error evaluating ensemble: {e}")
            return {'error': str(e)}
    
    def get_model_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive model insights and interpretability"""
        try:
            insights = {}
            
            # ML pipeline feature importance
            if self.ml_pipeline.is_trained:
                feature_importance = self.ml_pipeline.get_feature_importance_summary()
                insights['feature_importance'] = feature_importance
            
            # Transformer attention analysis
            if self.transformer_ensemble.is_trained:
                attention_analysis = self.transformer_ensemble.get_attention_analysis(df)
                insights['attention_analysis'] = attention_analysis
            
            # Model performance comparison
            if self.training_history:
                performance_comparison = {}
                for model_name, result in self.training_history.items():
                    if isinstance(result, dict) and 'success' in result:
                        if model_name == 'ml_pipeline' and result.get('success'):
                            perf = self.ml_pipeline.get_model_performance()
                            performance_comparison[model_name] = perf
                        elif model_name == 'transformer' and result.get('success'):
                            performance_comparison[model_name] = {
                                'r2_score': result.get('r2_score', 0),
                                'final_mse': result.get('final_mse', float('inf'))
                            }
                        elif model_name == 'lstm' and result.get('success'):
                            performance_comparison[model_name] = {
                                'validation_score': result.get('validation_score', 0),
                                'training_loss': result.get('training_loss', float('inf'))
                            }
                
                insights['performance_comparison'] = performance_comparison
            
            # Ensemble weights and confidence
            insights['ensemble_configuration'] = {
                'weights': self.ensemble_weights,
                'is_trained': self.is_trained,
                'training_history_available': bool(self.training_history)
            }
            
            return {
                'success': True,
                'insights': insights
            }
            
        except Exception as e:
            print(f"Error getting model insights: {e}")
            return {'error': str(e)}
    
    def optimize_ensemble_weights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize ensemble weights based on recent performance"""
        try:
            if not self.is_trained:
                return {'error': 'Models not trained'}
            
            # Evaluate individual model performance
            performance_scores = {}
            
            # Simple performance evaluation on recent data
            recent_df = df.tail(100) if len(df) > 100 else df
            
            for model_name in ['ml_pipeline', 'transformer', 'lstm']:
                try:
                    if model_name == 'ml_pipeline':
                        pred = self.ml_pipeline.predict_ensemble(recent_df)
                        score = pred.get('confidence', 0.1) if pred.get('success') else 0.1
                    elif model_name == 'transformer':
                        pred = self.transformer_ensemble.predict(recent_df)
                        score = pred.get('confidence', 0.1) if pred.get('success') else 0.1
                    elif model_name == 'lstm':
                        pred = self.lstm_predictor.predict(recent_df)
                        score = pred.get('confidence', 0.1) if pred.get('success') else 0.1
                    
                    performance_scores[model_name] = score
                except:
                    performance_scores[model_name] = 0.1
            
            # Normalize weights based on performance
            total_score = sum(performance_scores.values())
            if total_score > 0:
                optimized_weights = {
                    model: score / total_score 
                    for model, score in performance_scores.items()
                }
                
                # Apply smoothing to prevent extreme weight changes
                alpha = 0.3  # Learning rate for weight updates
                for model in self.ensemble_weights:
                    if model in optimized_weights:
                        self.ensemble_weights[model] = (
                            (1 - alpha) * self.ensemble_weights[model] +
                            alpha * optimized_weights[model]
                        )
                
                return {
                    'success': True,
                    'old_weights': {k: v for k, v in self.ensemble_weights.items()},
                    'new_weights': optimized_weights,
                    'performance_scores': performance_scores
                }
            else:
                return {'error': 'Unable to calculate performance scores'}
            
        except Exception as e:
            print(f"Error optimizing ensemble weights: {e}")
            return {'error': str(e)}