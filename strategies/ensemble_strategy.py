import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .base_strategy import TechnicalStrategy
from ai.ensemble import EnsembleStrategy as AIEnsemble
from utils.market_regime import MarketRegimeDetector
import pandas_ta as ta

class EnsembleStrategy(TechnicalStrategy):
    """Ensemble strategy combining multiple technical and AI approaches"""
    
    def __init__(self):
        super().__init__("Ensemble Strategy")
        self.ai_ensemble = AIEnsemble()
        self.regime_detector = MarketRegimeDetector()
        
        # Strategy weights based on market regime
        self.regime_weights = {
            'trending': {'technical': 0.4, 'ai': 0.6},
            'ranging': {'technical': 0.7, 'ai': 0.3},
            'volatile': {'technical': 0.5, 'ai': 0.5},
            'stable': {'technical': 0.3, 'ai': 0.7}
        }
        
        # Technical indicators configuration
        self.ta_config = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'bb_period': 20,
            'bb_std': 2,
            'volume_threshold': 1.5
        }
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            if not self.validate_data(data):
                return data
            
            indicators_df = data.copy()
            
            # Moving averages
            ma_data = self.calculate_moving_averages(data)
            indicators_df = pd.concat([indicators_df, ma_data], axis=1)
            
            # RSI
            indicators_df['RSI'] = self.calculate_rsi(data, self.ta_config['rsi_period'])
            
            # MACD
            macd_data = self.calculate_macd(data, 
                                          self.ta_config['macd_fast'],
                                          self.ta_config['macd_slow'])
            indicators_df = pd.concat([indicators_df, macd_data], axis=1)
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(data, 
                                                   self.ta_config['bb_period'],
                                                   self.ta_config['bb_std'])
            indicators_df = pd.concat([indicators_df, bb_data], axis=1)
            
            # Volume indicators
            vol_data = self.calculate_volume_indicators(data)
            indicators_df = pd.concat([indicators_df, vol_data], axis=1)
            
            # Momentum indicators
            momentum_data = self.calculate_momentum_indicators(data)
            indicators_df = pd.concat([indicators_df, momentum_data], axis=1)
            
            # Additional custom indicators
            indicators_df = self._calculate_custom_indicators(indicators_df)
            
            return indicators_df.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return data
    
    def _calculate_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom indicators for ensemble strategy"""
        try:
            # Price momentum
            data['Price_Momentum_5'] = data['close'].pct_change(5)
            data['Price_Momentum_10'] = data['close'].pct_change(10)
            
            # Volatility measures
            data['Volatility_5'] = data['close'].pct_change().rolling(5).std()
            data['Volatility_20'] = data['close'].pct_change().rolling(20).std()
            
            # Volume-price trend
            data['VPT'] = (data['volume'] * data['close'].pct_change()).cumsum()
            
            # Price channels
            data['High_Channel_20'] = data['high'].rolling(20).max()
            data['Low_Channel_20'] = data['low'].rolling(20).min()
            data['Channel_Position'] = ((data['close'] - data['Low_Channel_20']) / 
                                      (data['High_Channel_20'] - data['Low_Channel_20']))
            
            # Trend strength
            data['Trend_Strength'] = abs(data.get('SMA_20', data['close']) - 
                                       data.get('SMA_50', data['close'])) / data['close']
            
            return data
            
        except Exception as e:
            print(f"Error calculating custom indicators: {e}")
            return data
    
    def generate_technical_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signal based on technical analysis"""
        try:
            if len(data) < 50:
                return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0}
            
            signals = []
            strengths = []
            
            # RSI signal
            rsi = data['RSI'].iloc[-1]
            if rsi < self.ta_config['rsi_oversold']:
                signals.append('BUY')
                strengths.append((self.ta_config['rsi_oversold'] - rsi) / self.ta_config['rsi_oversold'])
            elif rsi > self.ta_config['rsi_overbought']:
                signals.append('SELL')
                strengths.append((rsi - self.ta_config['rsi_overbought']) / (100 - self.ta_config['rsi_overbought']))
            else:
                signals.append('HOLD')
                strengths.append(0.1)
            
            # MACD signal
            macd_col = f'MACD_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_9'
            signal_col = f'MACDs_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_9'
            
            if macd_col in data.columns and signal_col in data.columns:
                macd = data[macd_col].iloc[-1]
                macd_signal = data[signal_col].iloc[-1]
                macd_diff = macd - macd_signal
                
                if macd_diff > 0 and data[macd_col].iloc[-2] <= data[signal_col].iloc[-2]:
                    signals.append('BUY')
                    strengths.append(min(1.0, abs(macd_diff) * 100))
                elif macd_diff < 0 and data[macd_col].iloc[-2] >= data[signal_col].iloc[-2]:
                    signals.append('SELL')
                    strengths.append(min(1.0, abs(macd_diff) * 100))
                else:
                    signals.append('HOLD')
                    strengths.append(0.1)
            
            # Bollinger Bands signal
            bb_lower = data[f'BBL_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}'].iloc[-1]
            bb_upper = data[f'BBU_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}'].iloc[-1]
            current_price = data['close'].iloc[-1]
            
            if current_price <= bb_lower:
                signals.append('BUY')
                strengths.append((bb_lower - current_price) / bb_lower)
            elif current_price >= bb_upper:
                signals.append('SELL')
                strengths.append((current_price - bb_upper) / bb_upper)
            else:
                signals.append('HOLD')
                strengths.append(0.1)
            
            # Moving average crossover
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                sma_20 = data['SMA_20'].iloc[-1]
                sma_50 = data['SMA_50'].iloc[-1]
                prev_sma_20 = data['SMA_20'].iloc[-2]
                prev_sma_50 = data['SMA_50'].iloc[-2]
                
                if sma_20 > sma_50 and prev_sma_20 <= prev_sma_50:
                    signals.append('BUY')
                    strengths.append(min(1.0, (sma_20 - sma_50) / sma_50))
                elif sma_20 < sma_50 and prev_sma_20 >= prev_sma_50:
                    signals.append('SELL')
                    strengths.append(min(1.0, (sma_50 - sma_20) / sma_20))
                else:
                    signals.append('HOLD')
                    strengths.append(0.1)
            
            # Volume confirmation
            volume_signal = self._get_volume_signal(data)
            signals.append(volume_signal['signal'])
            strengths.append(volume_signal['strength'])
            
            # Aggregate signals
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            hold_count = signals.count('HOLD')
            
            total_signals = len(signals)
            
            if buy_count > sell_count and buy_count > hold_count:
                final_signal = 'BUY'
                confidence = buy_count / total_signals
            elif sell_count > buy_count and sell_count > hold_count:
                final_signal = 'SELL'
                confidence = sell_count / total_signals
            else:
                final_signal = 'HOLD'
                confidence = max(hold_count, buy_count, sell_count) / total_signals
            
            avg_strength = np.mean(strengths) if strengths else 0.0
            
            return {
                'signal': final_signal,
                'strength': avg_strength,
                'confidence': confidence,
                'component_signals': {
                    'rsi': signals[0] if len(signals) > 0 else 'HOLD',
                    'macd': signals[1] if len(signals) > 1 else 'HOLD',
                    'bollinger': signals[2] if len(signals) > 2 else 'HOLD',
                    'ma_cross': signals[3] if len(signals) > 3 else 'HOLD',
                    'volume': signals[4] if len(signals) > 4 else 'HOLD'
                }
            }
            
        except Exception as e:
            print(f"Error generating technical signal: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0}
    
    def _get_volume_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate volume-based signal"""
        try:
            if 'Volume_SMA_20' not in data.columns:
                return {'signal': 'HOLD', 'strength': 0.1}
            
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['Volume_SMA_20'].iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > self.ta_config['volume_threshold']:
                # High volume - confirm trend
                price_change = data['close'].pct_change().iloc[-1]
                if price_change > 0:
                    return {'signal': 'BUY', 'strength': min(1.0, volume_ratio - 1)}
                elif price_change < 0:
                    return {'signal': 'SELL', 'strength': min(1.0, volume_ratio - 1)}
            
            return {'signal': 'HOLD', 'strength': 0.1}
            
        except Exception as e:
            print(f"Error getting volume signal: {e}")
            return {'signal': 'HOLD', 'strength': 0.1}
    
    def generate_signal(self, data: pd.DataFrame, ai_predictions: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate ensemble signal combining technical and AI analysis"""
        try:
            # Calculate indicators
            indicators_data = self.calculate_indicators(data)
            
            # Detect market regime
            regime = self.regime_detector.detect_regime(indicators_data)
            regime_weights = self.regime_weights.get(regime, self.regime_weights['stable'])
            
            # Generate technical signal
            technical_signal = self.generate_technical_signal(indicators_data)
            
            # Get AI signal if predictions provided
            ai_signal = {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0}
            if ai_predictions:
                ai_result = self.ai_ensemble.calculate_ensemble_prediction(ai_predictions)
                ai_signal = self.ai_ensemble.generate_signal(ai_result)
            
            # Combine signals based on market regime
            tech_weight = regime_weights['technical']
            ai_weight = regime_weights['ai']
            
            # Convert signals to numeric scores
            signal_scores = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
            
            tech_score = signal_scores.get(technical_signal['signal'], 0)
            ai_score = signal_scores.get(ai_signal['signal'], 0)
            
            # Weighted combination
            combined_score = (tech_score * technical_signal['strength'] * tech_weight +
                            ai_score * ai_signal['strength'] * ai_weight)
            
            # Determine final signal
            if combined_score > 0.2:
                final_signal = 'BUY'
            elif combined_score < -0.2:
                final_signal = 'SELL'
            else:
                final_signal = 'HOLD'
            
            # Calculate combined confidence
            combined_confidence = (technical_signal['confidence'] * tech_weight +
                                 ai_signal['confidence'] * ai_weight)
            
            signal_result = {
                'signal': final_signal,
                'strength': abs(combined_score),
                'confidence': combined_confidence,
                'market_regime': regime,
                'technical_signal': technical_signal,
                'ai_signal': ai_signal,
                'regime_weights': regime_weights,
                'combined_score': combined_score
            }
            
            # Add to history
            self.add_signal_to_history(signal_result)
            
            return signal_result
            
        except Exception as e:
            print(f"Error generating ensemble signal: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'market_regime': 'unknown',
                'error': str(e)
            }
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get comprehensive strategy summary"""
        try:
            recent_signals = self.get_recent_signals(50)
            
            if not recent_signals:
                return {'status': 'No signals generated yet'}
            
            # Signal distribution
            signal_counts = {}
            for signal in recent_signals:
                sig = signal.get('signal', 'HOLD')
                signal_counts[sig] = signal_counts.get(sig, 0) + 1
            
            # Average confidence and strength
            confidences = [s.get('confidence', 0) for s in recent_signals]
            strengths = [s.get('strength', 0) for s in recent_signals]
            
            # Regime distribution
            regime_counts = {}
            for signal in recent_signals:
                regime = signal.get('market_regime', 'unknown')
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            return {
                'total_signals': len(recent_signals),
                'signal_distribution': signal_counts,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'avg_strength': np.mean(strengths) if strengths else 0,
                'regime_distribution': regime_counts,
                'latest_signal': recent_signals[-1] if recent_signals else None,
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            print(f"Error getting strategy summary: {e}")
            return {'error': str(e)}
