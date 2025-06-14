"""
Signal Filter - Market Context-Aware Signal Processing
Applies market context penalties and filtering logic to trading signals
"""

import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from .market_regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

class SignalFilter:
    """Apply market context penalty layer and advanced signal filtering"""
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.filtered_signals = []
        
    def apply_market_context_penalty(self, signal: Dict, market_data: pd.DataFrame) -> Tuple[Dict, bool]:
        """Apply market context penalty to signal based on current regime"""
        try:
            # Detect current market regime
            regime_data = self.regime_detector.detect_regime(market_data)
            
            # Apply context filtering
            adjusted_confidence, execute_signal = self.regime_detector.apply_market_context_filter(
                signal['confidence'], 
                signal['signal_type'], 
                regime_data
            )
            
            # Update signal with context information
            filtered_signal = signal.copy()
            filtered_signal['original_confidence'] = signal['confidence']
            filtered_signal['adjusted_confidence'] = adjusted_confidence
            filtered_signal['market_regime'] = regime_data['regime']
            filtered_signal['regime_confidence'] = regime_data['confidence']
            filtered_signal['context_applied'] = True
            
            # Add regime details for analysis
            filtered_signal['regime_details'] = {
                'bull_score': regime_data['bull_score'],
                'bear_score': regime_data['bear_score'],
                'sideways_score': regime_data['sideways_score']
            }
            
            return filtered_signal, execute_signal
            
        except Exception as e:
            logger.error(f"Market context penalty failed: {e}")
            return signal, True
    
    def apply_volume_filter(self, signal: Dict, market_data: pd.DataFrame) -> Tuple[Dict, bool]:
        """Filter signals based on volume conditions"""
        try:
            if 'volume' not in market_data.columns or len(market_data) < 20:
                return signal, True
            
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            
            if avg_volume == 0:
                return signal, True
            
            volume_ratio = current_volume / avg_volume
            
            # Volume-based confidence adjustment
            volume_penalty = 0
            execute_signal = True
            
            if volume_ratio < 0.5:  # Very low volume
                volume_penalty = -10
                if signal['confidence'] < 80:
                    execute_signal = False
            elif volume_ratio < 0.8:  # Low volume
                volume_penalty = -5
            elif volume_ratio > 2.0:  # High volume (positive)
                volume_penalty = 5
            
            # Update signal
            filtered_signal = signal.copy()
            filtered_signal['volume_ratio'] = volume_ratio
            filtered_signal['volume_penalty'] = volume_penalty
            filtered_signal['adjusted_confidence'] = max(0, 
                filtered_signal.get('adjusted_confidence', signal['confidence']) + volume_penalty)
            
            return filtered_signal, execute_signal
            
        except Exception as e:
            logger.error(f"Volume filter failed: {e}")
            return signal, True
    
    def apply_volatility_filter(self, signal: Dict, market_data: pd.DataFrame) -> Tuple[Dict, bool]:
        """Filter signals based on market volatility"""
        try:
            if len(market_data) < 20:
                return signal, True
            
            # Calculate volatility
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            
            volatility_penalty = 0
            execute_signal = True
            
            # High volatility penalty (risky conditions)
            if volatility > 0.05:  # Very high volatility
                volatility_penalty = -15
                if signal['confidence'] < 85:
                    execute_signal = False
            elif volatility > 0.03:  # High volatility
                volatility_penalty = -8
            elif volatility < 0.01:  # Very low volatility (sideways)
                if signal['signal_type'] == 'BUY' and signal['confidence'] < 75:
                    execute_signal = False
            
            # Update signal
            filtered_signal = signal.copy()
            filtered_signal['volatility'] = volatility
            filtered_signal['volatility_penalty'] = volatility_penalty
            filtered_signal['adjusted_confidence'] = max(0,
                filtered_signal.get('adjusted_confidence', signal['confidence']) + volatility_penalty)
            
            return filtered_signal, execute_signal
            
        except Exception as e:
            logger.error(f"Volatility filter failed: {e}")
            return signal, True
    
    def apply_time_based_filter(self, signal: Dict) -> Tuple[Dict, bool]:
        """Apply time-based filtering (market hours, etc.)"""
        try:
            import datetime
            
            current_time = datetime.datetime.now()
            hour = current_time.hour
            
            # Crypto markets are 24/7, but some times are less liquid
            time_penalty = 0
            
            # Reduce confidence during typically low-volume hours (3-7 AM UTC)
            if 3 <= hour <= 7:
                time_penalty = -5
            
            # Update signal
            filtered_signal = signal.copy()
            filtered_signal['hour'] = hour
            filtered_signal['time_penalty'] = time_penalty
            filtered_signal['adjusted_confidence'] = max(0,
                filtered_signal.get('adjusted_confidence', signal['confidence']) + time_penalty)
            
            return filtered_signal, True
            
        except Exception as e:
            logger.error(f"Time filter failed: {e}")
            return signal, True
    
    def comprehensive_signal_filter(self, signal: Dict, market_data: pd.DataFrame) -> Tuple[Optional[Dict], bool]:
        """Apply comprehensive filtering pipeline"""
        try:
            # Start with original signal
            filtered_signal = signal.copy()
            execute_signal = True
            
            # Apply market context penalty (most important)
            filtered_signal, context_execute = self.apply_market_context_penalty(filtered_signal, market_data)
            execute_signal = execute_signal and context_execute
            
            # Apply volume filter
            filtered_signal, volume_execute = self.apply_volume_filter(filtered_signal, market_data)
            execute_signal = execute_signal and volume_execute
            
            # Apply volatility filter
            filtered_signal, volatility_execute = self.apply_volatility_filter(filtered_signal, market_data)
            execute_signal = execute_signal and volatility_execute
            
            # Apply time-based filter
            filtered_signal, time_execute = self.apply_time_based_filter(filtered_signal)
            execute_signal = execute_signal and time_execute
            
            # Final confidence check
            final_confidence = filtered_signal.get('adjusted_confidence', signal['confidence'])
            
            # Additional bear market protection for BUY signals
            if (signal['signal_type'] == 'BUY' and 
                filtered_signal.get('market_regime') == 'bear' and 
                final_confidence < 80):
                execute_signal = False
            
            # Minimum confidence threshold
            if final_confidence < 60:
                execute_signal = False
            
            # Log filtering result
            if execute_signal:
                logger.info(f"Signal passed filters: {signal['symbol']} {signal['signal_type']} "
                           f"({signal['confidence']:.1f}% → {final_confidence:.1f}%) "
                           f"Regime: {filtered_signal.get('market_regime', 'unknown')}")
            else:
                logger.info(f"Signal filtered out: {signal['symbol']} {signal['signal_type']} "
                           f"({signal['confidence']:.1f}% → {final_confidence:.1f}%) "
                           f"Regime: {filtered_signal.get('market_regime', 'unknown')}")
            
            # Store filtered signal for analysis
            filtered_signal['filter_result'] = 'passed' if execute_signal else 'rejected'
            filtered_signal['final_confidence'] = final_confidence
            self.filtered_signals.append(filtered_signal)
            
            # Keep only recent signals
            if len(self.filtered_signals) > 200:
                self.filtered_signals = self.filtered_signals[-200:]
            
            return filtered_signal if execute_signal else None, execute_signal
            
        except Exception as e:
            logger.error(f"Comprehensive signal filtering failed: {e}")
            return signal, True
    
    def get_filter_statistics(self) -> Dict:
        """Get filtering statistics for analysis"""
        if not self.filtered_signals:
            return {}
        
        total_signals = len(self.filtered_signals)
        passed_signals = len([s for s in self.filtered_signals if s['filter_result'] == 'passed'])
        
        # Regime breakdown
        regimes = [s.get('market_regime', 'unknown') for s in self.filtered_signals]
        regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
        
        # Signal type breakdown
        signal_types = [s['signal_type'] for s in self.filtered_signals]
        type_counts = {stype: signal_types.count(stype) for stype in set(signal_types)}
        
        # Average confidence adjustment
        adjustments = [s.get('final_confidence', 0) - s['original_confidence'] 
                      for s in self.filtered_signals if 'original_confidence' in s]
        avg_adjustment = sum(adjustments) / len(adjustments) if adjustments else 0
        
        return {
            'total_signals': total_signals,
            'passed_signals': passed_signals,
            'pass_rate': passed_signals / total_signals if total_signals > 0 else 0,
            'regime_breakdown': regime_counts,
            'signal_type_breakdown': type_counts,
            'average_confidence_adjustment': avg_adjustment,
            'recent_regime': self.filtered_signals[-1].get('market_regime', 'unknown') if self.filtered_signals else 'unknown'
        }
    
    def get_recent_filtered_signals(self, limit: int = 20) -> list:
        """Get recent filtered signals for analysis"""
        return self.filtered_signals[-limit:] if self.filtered_signals else []