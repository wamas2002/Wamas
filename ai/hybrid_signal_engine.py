"""
Hybrid Signal Engine - Combines top performing models for enhanced decision making
Blends signals from multiple AI models with confidence-weighted voting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

@dataclass
class ModelSignal:
    """Individual model signal with metadata"""
    model_name: str
    decision: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    predicted_return: float
    reasoning: Dict[str, Any]
    timestamp: datetime
    execution_latency: float

@dataclass
class HybridSignal:
    """Combined hybrid signal from multiple models"""
    final_decision: str
    combined_confidence: float
    weighted_return: float
    participating_models: List[str]
    model_votes: Dict[str, ModelSignal]
    consensus_strength: float
    reasoning: Dict[str, Any]
    timestamp: datetime
    hybrid_mode: bool

class HybridSignalEngine:
    """
    Combines signals from top performing models to generate enhanced trading decisions
    Uses confidence-weighted voting and consensus analysis
    """
    
    def __init__(self, adaptive_model_selector, ai_performance_tracker):
        self.model_selector = adaptive_model_selector
        self.performance_tracker = ai_performance_tracker
        self.signal_history = {}  # symbol -> recent signals
        self.consensus_threshold = 0.6  # Minimum consensus for high-confidence trades
        self.min_models_required = 2  # Minimum models for hybrid decision
        self.max_signal_age = 300  # 5 minutes max signal age
        
        # Model weight factors
        self.weight_factors = {
            'performance': 0.4,  # Historical performance weight
            'confidence': 0.3,   # Current confidence weight
            'latency': 0.1,      # Execution speed weight
            'consistency': 0.2   # Decision consistency weight
        }
        
        logger.info("Hybrid Signal Engine initialized")
    
    def generate_hybrid_signal(self, symbol: str, individual_signals: List[ModelSignal]) -> HybridSignal:
        """Generate hybrid signal from individual model signals"""
        try:
            if len(individual_signals) < self.min_models_required:
                return self._fallback_signal(symbol, individual_signals)
            
            # Filter signals by age and quality
            valid_signals = self._filter_valid_signals(individual_signals)
            if len(valid_signals) < self.min_models_required:
                return self._fallback_signal(symbol, valid_signals)
            
            # Calculate model weights based on performance
            model_weights = self._calculate_model_weights(symbol, valid_signals)
            
            # Perform weighted voting
            voting_results = self._weighted_voting(valid_signals, model_weights)
            
            # Determine final decision
            final_decision = self._determine_final_decision(voting_results)
            
            # Calculate combined confidence
            combined_confidence = self._calculate_combined_confidence(
                valid_signals, model_weights, voting_results
            )
            
            # Calculate weighted return prediction
            weighted_return = self._calculate_weighted_return(valid_signals, model_weights)
            
            # Analyze consensus strength
            consensus_strength = self._analyze_consensus(valid_signals, final_decision)
            
            # Generate reasoning
            reasoning = self._generate_hybrid_reasoning(
                valid_signals, model_weights, voting_results, consensus_strength
            )
            
            # Create hybrid signal
            hybrid_signal = HybridSignal(
                final_decision=final_decision,
                combined_confidence=combined_confidence,
                weighted_return=weighted_return,
                participating_models=[s.model_name for s in valid_signals],
                model_votes={s.model_name: s for s in valid_signals},
                consensus_strength=consensus_strength,
                reasoning=reasoning,
                timestamp=datetime.now(),
                hybrid_mode=True
            )
            
            # Store signal in history
            self._store_signal_history(symbol, hybrid_signal)
            
            logger.debug(f"Generated hybrid signal for {symbol}: {final_decision} "
                        f"(confidence: {combined_confidence:.1f}%, consensus: {consensus_strength:.1f})")
            
            return hybrid_signal
            
        except Exception as e:
            logger.error(f"Error generating hybrid signal for {symbol}: {e}")
            return self._fallback_signal(symbol, individual_signals)
    
    def _filter_valid_signals(self, signals: List[ModelSignal]) -> List[ModelSignal]:
        """Filter signals by age and quality criteria"""
        valid_signals = []
        current_time = datetime.now()
        
        for signal in signals:
            # Check signal age
            age_seconds = (current_time - signal.timestamp).total_seconds()
            if age_seconds > self.max_signal_age:
                continue
            
            # Check minimum confidence threshold
            if signal.confidence < 40:  # Below 40% confidence is too low
                continue
            
            # Check for valid decision
            if signal.decision not in ['BUY', 'SELL', 'HOLD']:
                continue
            
            valid_signals.append(signal)
        
        return valid_signals
    
    def _calculate_model_weights(self, symbol: str, signals: List[ModelSignal]) -> Dict[str, float]:
        """Calculate weights for each model based on performance metrics"""
        weights = {}
        
        for signal in signals:
            model_name = signal.model_name
            
            # Get model performance data
            performance_data = self.performance_tracker.get_model_performance(
                symbol, model_name, days_back=7
            )
            
            if not performance_data:
                # Default weight for models without performance history
                weights[model_name] = 0.5
                continue
            
            # Calculate performance-based weight
            win_rate = performance_data.get('win_rate', 50) / 100.0
            avg_confidence = performance_data.get('avg_confidence', 60) / 100.0
            avg_latency = min(1.0, performance_data.get('avg_latency', 0.1))
            total_trades = performance_data.get('total_trades', 0)
            
            # Weight components
            performance_weight = win_rate * self.weight_factors['performance']
            confidence_weight = avg_confidence * self.weight_factors['confidence']
            latency_weight = (1.0 - avg_latency) * self.weight_factors['latency']
            
            # Consistency weight (based on trade count and recent performance)
            consistency_weight = min(1.0, total_trades / 20.0) * self.weight_factors['consistency']
            
            # Combine weights
            total_weight = performance_weight + confidence_weight + latency_weight + consistency_weight
            
            # Apply current signal confidence as modifier
            signal_confidence_modifier = signal.confidence / 100.0
            final_weight = total_weight * signal_confidence_modifier
            
            weights[model_name] = max(0.1, min(1.0, final_weight))  # Clamp between 0.1 and 1.0
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {model: weight / total_weight for model, weight in weights.items()}
        
        return weights
    
    def _weighted_voting(self, signals: List[ModelSignal], weights: Dict[str, float]) -> Dict[str, float]:
        """Perform weighted voting across model signals"""
        votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        for signal in signals:
            model_weight = weights.get(signal.model_name, 0.5)
            confidence_factor = signal.confidence / 100.0
            
            # Weighted vote strength
            vote_strength = model_weight * confidence_factor
            votes[signal.decision] += vote_strength
        
        # Normalize votes
        total_votes = sum(votes.values())
        if total_votes > 0:
            votes = {decision: vote / total_votes for decision, vote in votes.items()}
        
        return votes
    
    def _determine_final_decision(self, voting_results: Dict[str, float]) -> str:
        """Determine final decision from voting results"""
        # Find decision with highest vote
        max_vote = max(voting_results.values())
        winning_decisions = [decision for decision, vote in voting_results.items() if vote == max_vote]
        
        # Handle ties
        if len(winning_decisions) > 1:
            # If BUY and SELL tie, default to HOLD
            if 'BUY' in winning_decisions and 'SELL' in winning_decisions:
                return 'HOLD'
            # Otherwise, return the first winning decision
            return winning_decisions[0]
        
        # Check if winning vote meets minimum threshold
        if max_vote < 0.4:  # Less than 40% consensus
            return 'HOLD'
        
        return winning_decisions[0]
    
    def _calculate_combined_confidence(self, signals: List[ModelSignal], 
                                     weights: Dict[str, float], 
                                     voting_results: Dict[str, float]) -> float:
        """Calculate combined confidence score"""
        # Weighted average of individual confidences
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for signal in signals:
            model_weight = weights.get(signal.model_name, 0.5)
            weighted_confidence += signal.confidence * model_weight
            total_weight += model_weight
        
        base_confidence = weighted_confidence / total_weight if total_weight > 0 else 50.0
        
        # Consensus bonus: higher consensus increases confidence
        max_vote = max(voting_results.values())
        consensus_bonus = (max_vote - 0.5) * 20  # Up to 10% bonus for strong consensus
        
        # Model count bonus: more models increase confidence
        model_count_bonus = min(10, len(signals) * 2)  # Up to 10% bonus
        
        final_confidence = base_confidence + consensus_bonus + model_count_bonus
        return max(0, min(100, final_confidence))
    
    def _calculate_weighted_return(self, signals: List[ModelSignal], weights: Dict[str, float]) -> float:
        """Calculate weighted average of predicted returns"""
        weighted_return = 0.0
        total_weight = 0.0
        
        for signal in signals:
            model_weight = weights.get(signal.model_name, 0.5)
            weighted_return += signal.predicted_return * model_weight
            total_weight += model_weight
        
        return weighted_return / total_weight if total_weight > 0 else 0.0
    
    def _analyze_consensus(self, signals: List[ModelSignal], final_decision: str) -> float:
        """Analyze consensus strength among participating models"""
        if not signals:
            return 0.0
        
        # Count models that agree with final decision
        agreeing_models = sum(1 for signal in signals if signal.decision == final_decision)
        
        # Basic consensus percentage
        basic_consensus = agreeing_models / len(signals)
        
        # Weight by model confidence
        weighted_agreement = 0.0
        total_confidence = 0.0
        
        for signal in signals:
            if signal.decision == final_decision:
                weighted_agreement += signal.confidence
            total_confidence += signal.confidence
        
        confidence_weighted_consensus = weighted_agreement / total_confidence if total_confidence > 0 else 0
        
        # Combine both measures
        return (basic_consensus * 0.6) + (confidence_weighted_consensus * 0.4)
    
    def _generate_hybrid_reasoning(self, signals: List[ModelSignal], 
                                 weights: Dict[str, float], 
                                 voting_results: Dict[str, float],
                                 consensus_strength: float) -> Dict[str, Any]:
        """Generate detailed reasoning for hybrid decision"""
        # Model contributions
        model_contributions = []
        for signal in signals:
            model_weight = weights.get(signal.model_name, 0.5)
            contribution = {
                'model': signal.model_name,
                'decision': signal.decision,
                'confidence': signal.confidence,
                'weight': round(model_weight * 100, 1),
                'predicted_return': signal.predicted_return,
                'reasoning': signal.reasoning
            }
            model_contributions.append(contribution)
        
        # Sort by weight (influence)
        model_contributions.sort(key=lambda x: x['weight'], reverse=True)
        
        # Voting breakdown
        voting_breakdown = {
            decision: round(vote * 100, 1) 
            for decision, vote in voting_results.items()
        }
        
        # Decision factors
        decision_factors = []
        
        if consensus_strength > 0.8:
            decision_factors.append("Strong consensus among models")
        elif consensus_strength > 0.6:
            decision_factors.append("Moderate consensus among models")
        else:
            decision_factors.append("Weak consensus - conflicting signals")
        
        # Top contributing model
        if model_contributions:
            top_model = model_contributions[0]
            decision_factors.append(f"Primary influence: {top_model['model']} "
                                  f"({top_model['weight']}% weight)")
        
        return {
            'hybrid_analysis': True,
            'model_contributions': model_contributions,
            'voting_breakdown': voting_breakdown,
            'consensus_strength': round(consensus_strength * 100, 1),
            'decision_factors': decision_factors,
            'models_participating': len(signals),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _fallback_signal(self, symbol: str, signals: List[ModelSignal]) -> HybridSignal:
        """Generate fallback signal when hybrid analysis fails"""
        if not signals:
            # No signals available - generate neutral signal
            return HybridSignal(
                final_decision='HOLD',
                combined_confidence=50.0,
                weighted_return=0.0,
                participating_models=[],
                model_votes={},
                consensus_strength=0.0,
                reasoning={'fallback': 'No valid signals available'},
                timestamp=datetime.now(),
                hybrid_mode=False
            )
        
        # Use best single signal as fallback
        best_signal = max(signals, key=lambda s: s.confidence)
        
        return HybridSignal(
            final_decision=best_signal.decision,
            combined_confidence=best_signal.confidence,
            weighted_return=best_signal.predicted_return,
            participating_models=[best_signal.model_name],
            model_votes={best_signal.model_name: best_signal},
            consensus_strength=1.0,  # Single model = 100% consensus
            reasoning={
                'fallback': f'Using best single signal from {best_signal.model_name}',
                'original_reasoning': best_signal.reasoning
            },
            timestamp=datetime.now(),
            hybrid_mode=False
        )
    
    def _store_signal_history(self, symbol: str, signal: HybridSignal):
        """Store signal in history for analysis"""
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        self.signal_history[symbol].append(signal)
        
        # Keep only recent signals (last 100)
        if len(self.signal_history[symbol]) > 100:
            self.signal_history[symbol] = self.signal_history[symbol][-100:]
    
    def get_signal_history(self, symbol: str, limit: int = 20) -> List[HybridSignal]:
        """Get recent signal history for a symbol"""
        if symbol not in self.signal_history:
            return []
        
        return self.signal_history[symbol][-limit:]
    
    def get_consensus_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get consensus analysis for recent signals"""
        recent_signals = self.get_signal_history(symbol, 10)
        
        if not recent_signals:
            return {'error': 'No recent signals available'}
        
        # Analyze recent consensus trends
        hybrid_count = sum(1 for s in recent_signals if s.hybrid_mode)
        avg_consensus = np.mean([s.consensus_strength for s in recent_signals])
        avg_confidence = np.mean([s.combined_confidence for s in recent_signals])
        
        # Decision distribution
        decisions = [s.final_decision for s in recent_signals]
        decision_counts = {
            'BUY': decisions.count('BUY'),
            'SELL': decisions.count('SELL'),
            'HOLD': decisions.count('HOLD')
        }
        
        return {
            'symbol': symbol,
            'recent_signals': len(recent_signals),
            'hybrid_signals': hybrid_count,
            'avg_consensus_strength': round(avg_consensus * 100, 1),
            'avg_confidence': round(avg_confidence, 1),
            'decision_distribution': decision_counts,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def simulate_hybrid_signal(self, symbol: str) -> HybridSignal:
        """Simulate a hybrid signal for testing purposes"""
        # Create mock individual signals
        mock_signals = [
            ModelSignal(
                model_name='LSTM',
                decision='BUY',
                confidence=75.0,
                predicted_return=0.025,
                reasoning={'trend': 'bullish', 'strength': 'high'},
                timestamp=datetime.now(),
                execution_latency=0.12
            ),
            ModelSignal(
                model_name='GradientBoost',
                decision='BUY',
                confidence=68.0,
                predicted_return=0.018,
                reasoning={'pattern': 'breakout', 'probability': 'high'},
                timestamp=datetime.now(),
                execution_latency=0.08
            ),
            ModelSignal(
                model_name='Ensemble',
                decision='HOLD',
                confidence=55.0,
                predicted_return=0.005,
                reasoning={'mixed_signals': True, 'uncertainty': 'moderate'},
                timestamp=datetime.now(),
                execution_latency=0.15
            )
        ]
        
        return self.generate_hybrid_signal(symbol, mock_signals)