import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas_ta as ta
from enum import Enum

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    STABLE = "stable"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"

class MarketRegimeDetector:
    """Advanced market regime detection using multiple indicators"""
    
    def __init__(self):
        self.regime_history = []
        self.regime_confidence = {}
        
        # Thresholds for regime classification
        self.thresholds = {
            'volatility': {
                'low': 0.01,    # 1% daily volatility
                'high': 0.04    # 4% daily volatility
            },
            'trend_strength': {
                'weak': 0.02,   # 2% threshold
                'strong': 0.05  # 5% threshold
            },
            'momentum': {
                'bearish': -0.03,  # -3% threshold
                'bullish': 0.03    # 3% threshold
            },
            'mean_reversion': {
                'low': 0.5,     # Low mean reversion
                'high': 0.8     # High mean reversion
            }
        }
        
        # Lookback periods for different analyses
        self.lookback_periods = {
            'short': 10,
            'medium': 20,
            'long': 50
        }
        
    def detect_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime based on multiple factors"""
        try:
            if len(data) < self.lookback_periods['long']:
                return MarketRegime.STABLE.value
            
            # Calculate regime indicators
            regime_indicators = self._calculate_regime_indicators(data)
            
            # Determine regime based on indicators
            regime = self._classify_regime(regime_indicators)
            
            # Update regime history
            self._update_regime_history(regime, regime_indicators)
            
            return regime
            
        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return MarketRegime.STABLE.value
    
    def _calculate_regime_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various indicators for regime detection"""
        try:
            indicators = {}
            
            # 1. Volatility Analysis
            returns = data['close'].pct_change().dropna()
            short_vol = returns.tail(self.lookback_periods['short']).std()
            medium_vol = returns.tail(self.lookback_periods['medium']).std()
            long_vol = returns.tail(self.lookback_periods['long']).std()
            
            indicators['volatility_short'] = short_vol
            indicators['volatility_medium'] = medium_vol
            indicators['volatility_long'] = long_vol
            indicators['volatility_trend'] = short_vol - long_vol
            
            # 2. Trend Strength Analysis
            sma_short = data['close'].rolling(self.lookback_periods['short']).mean()
            sma_medium = data['close'].rolling(self.lookback_periods['medium']).mean()
            sma_long = data['close'].rolling(self.lookback_periods['long']).mean()
            
            current_price = data['close'].iloc[-1]
            
            # Trend strength based on price position relative to moving averages
            trend_strength = abs((current_price - sma_long.iloc[-1]) / sma_long.iloc[-1])
            indicators['trend_strength'] = trend_strength
            
            # Trend direction
            if current_price > sma_short.iloc[-1] > sma_medium.iloc[-1] > sma_long.iloc[-1]:
                indicators['trend_direction'] = 1  # Strong uptrend
            elif current_price < sma_short.iloc[-1] < sma_medium.iloc[-1] < sma_long.iloc[-1]:
                indicators['trend_direction'] = -1  # Strong downtrend
            else:
                indicators['trend_direction'] = 0  # Mixed/sideways
            
            # 3. Momentum Analysis
            momentum_short = (current_price / data['close'].iloc[-self.lookback_periods['short']]) - 1
            momentum_medium = (current_price / data['close'].iloc[-self.lookback_periods['medium']]) - 1
            momentum_long = (current_price / data['close'].iloc[-self.lookback_periods['long']]) - 1
            
            indicators['momentum_short'] = momentum_short
            indicators['momentum_medium'] = momentum_medium
            indicators['momentum_long'] = momentum_long
            
            # 4. Mean Reversion Tendency
            # Calculate using price distance from moving average
            price_deviation = abs(current_price - sma_medium.iloc[-1]) / sma_medium.iloc[-1]
            bb_data = ta.bbands(data['close'], length=20, std=2)
            
            if bb_data is not None and not bb_data.empty:
                bb_position = self._calculate_bb_position(data['close'].iloc[-1], bb_data.iloc[-1])
                indicators['bb_position'] = bb_position
            else:
                indicators['bb_position'] = 0.5
            
            indicators['mean_reversion_tendency'] = price_deviation
            
            # 5. Volume Analysis
            volume_sma = data['volume'].rolling(self.lookback_periods['medium']).mean()
            volume_ratio = data['volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
            indicators['volume_ratio'] = volume_ratio
            
            # 6. Range Analysis (for range-bound markets)
            high_20 = data['high'].rolling(self.lookback_periods['medium']).max()
            low_20 = data['low'].rolling(self.lookback_periods['medium']).min()
            range_size = (high_20.iloc[-1] - low_20.iloc[-1]) / data['close'].iloc[-1]
            indicators['range_size'] = range_size
            
            # Price position within range
            range_position = (current_price - low_20.iloc[-1]) / (high_20.iloc[-1] - low_20.iloc[-1])
            indicators['range_position'] = range_position
            
            # 7. RSI for overbought/oversold conditions
            try:
                rsi = ta.rsi(data['close'], length=14)
                indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
            except:
                indicators['rsi'] = 50
            
            # 8. MACD for momentum confirmation
            try:
                macd_data = ta.macd(data['close'], fast=12, slow=26, signal=9)
                if macd_data is not None and not macd_data.empty:
                    macd_line = macd_data.iloc[-1, 0]
                    signal_line = macd_data.iloc[-1, 2]
                    indicators['macd_signal'] = 1 if macd_line > signal_line else -1
                else:
                    indicators['macd_signal'] = 0
            except:
                indicators['macd_signal'] = 0
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating regime indicators: {e}")
            return {}
    
    def _calculate_bb_position(self, price: float, bb_row: pd.Series) -> float:
        """Calculate position within Bollinger Bands"""
        try:
            bb_columns = bb_row.index.tolist()
            upper_col = [col for col in bb_columns if 'BBU' in col][0]
            lower_col = [col for col in bb_columns if 'BBL' in col][0]
            
            bb_upper = bb_row[upper_col]
            bb_lower = bb_row[lower_col]
            
            if bb_upper > bb_lower:
                return (price - bb_lower) / (bb_upper - bb_lower)
            else:
                return 0.5
                
        except:
            return 0.5
    
    def _classify_regime(self, indicators: Dict[str, float]) -> str:
        """Classify market regime based on calculated indicators"""
        try:
            if not indicators:
                return MarketRegime.STABLE.value
            
            vol_short = indicators.get('volatility_short', 0)
            vol_medium = indicators.get('volatility_medium', 0)
            trend_strength = indicators.get('trend_strength', 0)
            trend_direction = indicators.get('trend_direction', 0)
            momentum_medium = indicators.get('momentum_medium', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            range_size = indicators.get('range_size', 0)
            bb_position = indicators.get('bb_position', 0.5)
            
            # Calculate regime scores
            regime_scores = {
                MarketRegime.VOLATILE.value: 0,
                MarketRegime.TRENDING.value: 0,
                MarketRegime.RANGING.value: 0,
                MarketRegime.STABLE.value: 0
            }
            
            # Volatility-based scoring
            if vol_short > self.thresholds['volatility']['high']:
                regime_scores[MarketRegime.VOLATILE.value] += 3
            elif vol_short < self.thresholds['volatility']['low']:
                regime_scores[MarketRegime.STABLE.value] += 2
            
            # Trend-based scoring
            if trend_strength > self.thresholds['trend_strength']['strong']:
                regime_scores[MarketRegime.TRENDING.value] += 3
                
                # Further classify trending markets
                if momentum_medium > self.thresholds['momentum']['bullish']:
                    regime_scores['bull_market'] = regime_scores[MarketRegime.TRENDING.value] + 1
                elif momentum_medium < self.thresholds['momentum']['bearish']:
                    regime_scores['bear_market'] = regime_scores[MarketRegime.TRENDING.value] + 1
                    
            elif trend_strength < self.thresholds['trend_strength']['weak']:
                regime_scores[MarketRegime.RANGING.value] += 2
                regime_scores[MarketRegime.STABLE.value] += 1
            
            # Range-bound market detection
            if (range_size < 0.1 and  # Small range
                abs(bb_position - 0.5) < 0.3 and  # Near center of BB
                abs(momentum_medium) < 0.02):  # Low momentum
                regime_scores[MarketRegime.RANGING.value] += 2
            
            # Volume confirmation
            if volume_ratio > 1.5:  # High volume
                if regime_scores[MarketRegime.TRENDING.value] > 0:
                    regime_scores[MarketRegime.TRENDING.value] += 1
                if regime_scores[MarketRegime.VOLATILE.value] > 0:
                    regime_scores[MarketRegime.VOLATILE.value] += 1
            
            # Volatility trend consideration
            vol_trend = indicators.get('volatility_trend', 0)
            if vol_trend > 0.005:  # Increasing volatility
                regime_scores[MarketRegime.VOLATILE.value] += 1
            
            # Find regime with highest score
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            
            # Store confidence
            total_score = sum(regime_scores.values())
            confidence = best_regime[1] / total_score if total_score > 0 else 0.5
            self.regime_confidence[best_regime[0]] = confidence
            
            return best_regime[0]
            
        except Exception as e:
            print(f"Error classifying regime: {e}")
            return MarketRegime.STABLE.value
    
    def _update_regime_history(self, regime: str, indicators: Dict[str, float]):
        """Update regime history with current detection"""
        try:
            regime_record = {
                'timestamp': datetime.now(),
                'regime': regime,
                'confidence': self.regime_confidence.get(regime, 0.5),
                'indicators': indicators.copy()
            }
            
            self.regime_history.append(regime_record)
            
            # Keep only last 1000 records
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
                
        except Exception as e:
            print(f"Error updating regime history: {e}")
    
    def get_regime_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive regime analysis"""
        try:
            current_regime = self.detect_regime(data)
            indicators = self._calculate_regime_indicators(data)
            
            # Calculate regime stability (how consistent regime has been)
            recent_regimes = [r['regime'] for r in self.regime_history[-10:]]
            regime_stability = recent_regimes.count(current_regime) / len(recent_regimes) if recent_regimes else 0
            
            # Calculate regime duration
            regime_duration = self._calculate_regime_duration(current_regime)
            
            # Get regime characteristics
            regime_characteristics = self._get_regime_characteristics(current_regime, indicators)
            
            return {
                'current_regime': current_regime,
                'confidence': self.regime_confidence.get(current_regime, 0.5),
                'stability': regime_stability,
                'duration_periods': regime_duration,
                'characteristics': regime_characteristics,
                'indicators': indicators,
                'regime_history': self.regime_history[-20:],  # Last 20 regime detections
                'recommendations': self._get_regime_recommendations(current_regime)
            }
            
        except Exception as e:
            print(f"Error getting regime analysis: {e}")
            return {'current_regime': MarketRegime.STABLE.value, 'error': str(e)}
    
    def _calculate_regime_duration(self, current_regime: str) -> int:
        """Calculate how long current regime has been in effect"""
        try:
            if not self.regime_history:
                return 0
            
            duration = 0
            for record in reversed(self.regime_history):
                if record['regime'] == current_regime:
                    duration += 1
                else:
                    break
                    
            return duration
            
        except Exception as e:
            print(f"Error calculating regime duration: {e}")
            return 0
    
    def _get_regime_characteristics(self, regime: str, indicators: Dict[str, float]) -> Dict[str, str]:
        """Get characteristics description for current regime"""
        try:
            characteristics = {}
            
            if regime == MarketRegime.TRENDING.value:
                trend_direction = indicators.get('trend_direction', 0)
                if trend_direction > 0:
                    characteristics['direction'] = "Uptrend"
                    characteristics['description'] = "Strong upward price movement with momentum"
                elif trend_direction < 0:
                    characteristics['direction'] = "Downtrend"
                    characteristics['description'] = "Strong downward price movement with momentum"
                else:
                    characteristics['direction'] = "Mixed trend"
                    characteristics['description'] = "Trending but direction is unclear"
                    
                characteristics['trading_style'] = "Trend following strategies recommended"
                characteristics['risk_level'] = "Medium"
                
            elif regime == MarketRegime.RANGING.value:
                characteristics['direction'] = "Sideways"
                characteristics['description'] = "Price moving within defined range"
                characteristics['trading_style'] = "Mean reversion strategies recommended"
                characteristics['risk_level'] = "Low to Medium"
                
            elif regime == MarketRegime.VOLATILE.value:
                characteristics['direction'] = "Unpredictable"
                characteristics['description'] = "High volatility with rapid price changes"
                characteristics['trading_style'] = "Reduced position sizes, breakout strategies"
                characteristics['risk_level'] = "High"
                
            elif regime == MarketRegime.STABLE.value:
                characteristics['direction'] = "Stable"
                characteristics['description'] = "Low volatility with predictable movements"
                characteristics['trading_style'] = "Position sizing can be increased"
                characteristics['risk_level'] = "Low"
                
            else:
                characteristics['direction'] = "Unknown"
                characteristics['description'] = "Market regime unclear"
                characteristics['trading_style'] = "Conservative approach recommended"
                characteristics['risk_level'] = "Medium"
            
            return characteristics
            
        except Exception as e:
            print(f"Error getting regime characteristics: {e}")
            return {}
    
    def _get_regime_recommendations(self, regime: str) -> List[str]:
        """Get trading recommendations based on regime"""
        try:
            recommendations = []
            
            if regime == MarketRegime.TRENDING.value:
                recommendations.extend([
                    "Use trend-following strategies",
                    "Increase position sizes gradually",
                    "Set trailing stops to capture trends",
                    "Look for momentum confirmations",
                    "Avoid counter-trend positions"
                ])
                
            elif regime == MarketRegime.RANGING.value:
                recommendations.extend([
                    "Use mean reversion strategies",
                    "Buy at support, sell at resistance",
                    "Use oscillators for entry signals",
                    "Take profits at range boundaries",
                    "Avoid breakout strategies"
                ])
                
            elif regime == MarketRegime.VOLATILE.value:
                recommendations.extend([
                    "Reduce position sizes",
                    "Use wider stop losses",
                    "Increase monitoring frequency",
                    "Consider volatility-based strategies",
                    "Be prepared for rapid changes"
                ])
                
            elif regime == MarketRegime.STABLE.value:
                recommendations.extend([
                    "Moderate position sizes acceptable",
                    "Use standard risk management",
                    "Consider multiple strategies",
                    "Good environment for backtesting",
                    "Monitor for regime changes"
                ])
            
            # General recommendations
            recommendations.extend([
                "Monitor regime changes closely",
                "Adjust strategy parameters accordingly",
                "Maintain proper risk management"
            ])
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting regime recommendations: {e}")
            return ["Use conservative trading approach"]
    
    def get_regime_transition_probability(self) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probabilities based on history"""
        try:
            if len(self.regime_history) < 20:
                return {}
            
            # Get regime sequence
            regimes = [r['regime'] for r in self.regime_history]
            
            # Count transitions
            transitions = {}
            
            for i in range(len(regimes) - 1):
                current = regimes[i]
                next_regime = regimes[i + 1]
                
                if current not in transitions:
                    transitions[current] = {}
                
                if next_regime not in transitions[current]:
                    transitions[current][next_regime] = 0
                
                transitions[current][next_regime] += 1
            
            # Calculate probabilities
            transition_probs = {}
            
            for current, next_regimes in transitions.items():
                total_transitions = sum(next_regimes.values())
                transition_probs[current] = {}
                
                for next_regime, count in next_regimes.items():
                    transition_probs[current][next_regime] = count / total_transitions
            
            return transition_probs
            
        except Exception as e:
            print(f"Error calculating transition probabilities: {e}")
            return {}
    
    def predict_regime_change(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Predict potential regime changes"""
        try:
            current_regime = self.detect_regime(data)
            indicators = self._calculate_regime_indicators(data)
            
            # Calculate regime change signals
            change_signals = []
            
            # Volatility change signal
            vol_trend = indicators.get('volatility_trend', 0)
            if abs(vol_trend) > 0.01:  # Significant volatility change
                if vol_trend > 0:
                    change_signals.append("Volatility increasing - potential regime change")
                else:
                    change_signals.append("Volatility decreasing - potential regime change")
            
            # Trend strength changes
            trend_strength = indicators.get('trend_strength', 0)
            if (current_regime == MarketRegime.RANGING.value and 
                trend_strength > self.thresholds['trend_strength']['strong']):
                change_signals.append("Breaking out of range - potential trending regime")
            elif (current_regime == MarketRegime.TRENDING.value and 
                  trend_strength < self.thresholds['trend_strength']['weak']):
                change_signals.append("Trend weakening - potential ranging regime")
            
            # Momentum divergence
            momentum_short = indicators.get('momentum_short', 0)
            momentum_medium = indicators.get('momentum_medium', 0)
            
            if (momentum_short > 0 and momentum_medium < 0) or (momentum_short < 0 and momentum_medium > 0):
                change_signals.append("Momentum divergence detected - potential regime change")
            
            # Calculate change probability
            change_probability = len(change_signals) / 5  # Max 5 signals
            
            return {
                'current_regime': current_regime,
                'change_probability': min(change_probability, 1.0),
                'change_signals': change_signals,
                'monitoring_indicators': {
                    'volatility_trend': vol_trend,
                    'trend_strength': trend_strength,
                    'momentum_divergence': abs(momentum_short - momentum_medium)
                }
            }
            
        except Exception as e:
            print(f"Error predicting regime change: {e}")
            return {'current_regime': MarketRegime.STABLE.value, 'change_probability': 0.0}
    
    def reset_regime_history(self):
        """Reset regime detection history"""
        try:
            self.regime_history.clear()
            self.regime_confidence.clear()
            print("Regime history reset successfully")
        except Exception as e:
            print(f"Error resetting regime history: {e}")
