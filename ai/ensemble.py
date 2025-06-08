import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

class EnsembleStrategy:
    """Ensemble strategy combining multiple AI models with dynamic weighting"""
    
    def __init__(self):
        self.model_weights = {
            'lstm': 0.4,
            'prophet': 0.3,
            'transformer': 0.3
        }
        self.performance_history = {}
        self.adaptation_period = 100  # Number of predictions to track
        self.min_confidence = 0.6
        self.weight_adaptation_rate = 0.1
        
    def update_weights(self, predictions: Dict[str, float], actual_return: float):
        """Update model weights based on recent performance"""
        try:
            current_time = datetime.now()
            
            # Initialize performance tracking for each model
            for model_name in self.model_weights.keys():
                if model_name not in self.performance_history:
                    self.performance_history[model_name] = []
            
            # Calculate prediction errors for each model
            for model_name, prediction in predictions.items():
                if model_name in self.model_weights:
                    error = abs(prediction - actual_return)
                    
                    # Store performance data
                    self.performance_history[model_name].append({
                        'timestamp': current_time,
                        'error': error,
                        'prediction': prediction,
                        'actual': actual_return
                    })
                    
                    # Keep only recent history
                    if len(self.performance_history[model_name]) > self.adaptation_period:
                        self.performance_history[model_name] = \
                            self.performance_history[model_name][-self.adaptation_period:]
            
            # Update weights based on recent performance
            self._adapt_weights()
            
        except Exception as e:
            print(f"Error updating weights: {e}")
    
    def _adapt_weights(self):
        """Adapt weights based on recent model performance"""
        try:
            if not self.performance_history:
                return
            
            # Calculate recent performance metrics
            model_scores = {}
            for model_name, history in self.performance_history.items():
                if len(history) >= 10:  # Minimum history required
                    recent_errors = [h['error'] for h in history[-20:]]  # Last 20 predictions
                    avg_error = np.mean(recent_errors)
                    
                    # Convert error to performance score (lower error = higher score)
                    model_scores[model_name] = 1.0 / (1.0 + avg_error)
            
            if len(model_scores) >= 2:
                # Normalize scores to create new weights
                total_score = sum(model_scores.values())
                new_weights = {}
                
                for model_name, score in model_scores.items():
                    new_weight = score / total_score
                    
                    # Apply adaptation rate to smooth weight changes
                    current_weight = self.model_weights.get(model_name, 0.33)
                    adapted_weight = (current_weight * (1 - self.weight_adaptation_rate) + 
                                    new_weight * self.weight_adaptation_rate)
                    
                    new_weights[model_name] = adapted_weight
                
                # Ensure weights sum to 1
                weight_sum = sum(new_weights.values())
                if weight_sum > 0:
                    self.model_weights.update({k: v/weight_sum for k, v in new_weights.items()})
                
        except Exception as e:
            print(f"Error adapting weights: {e}")
    
    def calculate_ensemble_prediction(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate ensemble prediction with confidence scoring"""
        try:
            if not predictions:
                return {'prediction': 0.0, 'confidence': 0.0}
            
            # Filter out non-model predictions
            model_predictions = {k: v for k, v in predictions.items() 
                               if k in self.model_weights and not np.isnan(v)}
            
            if not model_predictions:
                return {'prediction': 0.0, 'confidence': 0.0}
            
            # Calculate weighted ensemble prediction
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for model_name, prediction in model_predictions.items():
                weight = self.model_weights.get(model_name, 0.0)
                weighted_sum += prediction * weight
                weight_sum += weight
            
            if weight_sum == 0:
                return {'prediction': 0.0, 'confidence': 0.0}
            
            ensemble_prediction = weighted_sum / weight_sum
            
            # Calculate confidence based on model agreement
            confidence = self._calculate_confidence(model_predictions, ensemble_prediction)
            
            return {
                'prediction': ensemble_prediction,
                'confidence': confidence,
                'model_weights': self.model_weights.copy(),
                'individual_predictions': model_predictions
            }
            
        except Exception as e:
            print(f"Error calculating ensemble prediction: {e}")
            return {'prediction': 0.0, 'confidence': 0.0}
    
    def _calculate_confidence(self, predictions: Dict[str, float], ensemble_pred: float) -> float:
        """Calculate confidence based on model agreement"""
        try:
            if len(predictions) < 2:
                return 0.5
            
            # Calculate variance of predictions
            pred_values = list(predictions.values())
            pred_variance = np.var(pred_values)
            
            # Calculate agreement (inverse of variance)
            # Higher agreement = lower variance = higher confidence
            max_variance = 0.01  # Maximum expected variance for full confidence
            confidence = max(0.0, 1.0 - (pred_variance / max_variance))
            
            # Additional confidence boost if models agree on direction
            pred_signs = [1 if x > 0 else -1 if x < 0 else 0 for x in pred_values]
            sign_agreement = len(set(pred_signs)) == 1 and pred_signs[0] != 0
            
            if sign_agreement:
                confidence *= 1.2  # Boost confidence if all models agree on direction
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5
    
    def generate_signal(self, ensemble_result: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading signal based on ensemble prediction"""
        try:
            prediction = ensemble_result.get('prediction', 0.0)
            confidence = ensemble_result.get('confidence', 0.0)
            
            # Determine signal strength
            signal_strength = abs(prediction) * confidence
            
            # Generate signal
            if confidence < self.min_confidence:
                signal = 'HOLD'
                strength = 0.0
            elif prediction > 0.002 and signal_strength > 0.001:  # 0.2% threshold
                signal = 'BUY'
                strength = min(1.0, signal_strength * 10)  # Scale strength
            elif prediction < -0.002 and signal_strength > 0.001:
                signal = 'SELL'
                strength = min(1.0, signal_strength * 10)
            else:
                signal = 'HOLD'
                strength = 0.0
            
            return {
                'signal': signal,
                'strength': strength,
                'confidence': confidence,
                'prediction': prediction,
                'timestamp': datetime.now(),
                'model_weights': self.model_weights.copy()
            }
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'prediction': 0.0,
                'timestamp': datetime.now()
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        try:
            summary = {}
            
            for model_name, history in self.performance_history.items():
                if len(history) > 0:
                    recent_history = history[-50:]  # Last 50 predictions
                    errors = [h['error'] for h in recent_history]
                    
                    summary[model_name] = {
                        'avg_error': np.mean(errors),
                        'error_std': np.std(errors),
                        'min_error': np.min(errors),
                        'max_error': np.max(errors),
                        'current_weight': self.model_weights.get(model_name, 0.0),
                        'prediction_count': len(history)
                    }
            
            return summary
            
        except Exception as e:
            print(f"Error getting performance summary: {e}")
            return {}
    
    def reset_adaptation(self):
        """Reset adaptation mechanism"""
        self.performance_history = {}
        self.model_weights = {
            'lstm': 0.4,
            'prophet': 0.3,
            'transformer': 0.3
        }
