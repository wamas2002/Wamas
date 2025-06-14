"""
Online Confidence Tuner - Adaptive Signal Weighting Based on Performance
Continuously learns from trade outcomes to optimize signal confidence levels
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class OnlineConfidenceTuner:
    """Dynamically adjust signal confidence based on historical performance"""
    
    def __init__(self, db_path: str = 'confidence_tuning.db'):
        self.db_path = db_path
        self.performance_history = []
        self.confidence_adjustments = {}
        self.learning_rate = 0.02
        self.min_samples = 10
        
        # Performance tracking for different signal components
        self.component_weights = {
            'rsi': 1.0,
            'macd': 1.0,
            'ema': 1.0,
            'volume': 1.0,
            'regime': 1.0
        }
        
        self.setup_database()
    
    def setup_database(self):
        """Initialize confidence tuning database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    original_confidence REAL,
                    adjusted_confidence REAL,
                    
                    -- Signal components
                    rsi_score REAL,
                    macd_score REAL,
                    ema_score REAL,
                    volume_score REAL,
                    regime_score REAL,
                    
                    -- Outcome
                    executed BOOLEAN,
                    outcome_24h REAL,
                    outcome_72h REAL,
                    win_loss TEXT,
                    
                    -- Learning metrics
                    prediction_error REAL,
                    confidence_calibration REAL
                )
            ''')
            
            # Weight adjustments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weight_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    component TEXT NOT NULL,
                    old_weight REAL,
                    new_weight REAL,
                    performance_metric REAL,
                    sample_size INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Online confidence tuning database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def analyze_signal_components(self, signal: Dict) -> Dict[str, float]:
        """Extract and normalize signal component scores"""
        components = signal.get('components', {})
        
        scores = {}
        
        # RSI component score
        rsi = components.get('rsi')
        if rsi is not None:
            if signal['signal_type'] == 'BUY':
                scores['rsi'] = max(0, (50 - rsi) / 20) if rsi < 50 else 0
            else:  # SELL
                scores['rsi'] = max(0, (rsi - 50) / 30) if rsi > 50 else 0
        else:
            scores['rsi'] = 0.5
        
        # MACD component score
        macd = components.get('macd')
        if macd is not None:
            if signal['signal_type'] == 'BUY':
                scores['macd'] = 1.0 if macd > 0 else 0.3
            else:  # SELL
                scores['macd'] = 1.0 if macd < 0 else 0.3
        else:
            scores['macd'] = 0.5
        
        # EMA component score
        ema20 = components.get('ema20')
        ema50 = components.get('ema50')
        price = signal.get('price', 0)
        
        if ema20 and ema50 and price:
            if signal['signal_type'] == 'BUY':
                scores['ema'] = 1.0 if price > ema20 > ema50 else 0.4
            else:  # SELL
                scores['ema'] = 1.0 if price < ema20 < ema50 else 0.4
        else:
            scores['ema'] = 0.5
        
        # Volume component score
        volume_ratio = signal.get('volume_ratio', 1.0)
        scores['volume'] = min(1.0, max(0.2, volume_ratio / 2.0))
        
        # Regime component score
        regime = signal.get('market_regime', 'unknown')
        regime_conf = signal.get('regime_confidence', 0.5)
        
        if signal['signal_type'] == 'BUY':
            if regime == 'bull':
                scores['regime'] = regime_conf
            elif regime == 'bear':
                scores['regime'] = 0.2
            else:
                scores['regime'] = 0.5
        else:  # SELL
            if regime == 'bear':
                scores['regime'] = regime_conf
            elif regime == 'bull':
                scores['regime'] = 0.2
            else:
                scores['regime'] = 0.5
        
        return scores
    
    def calculate_weighted_confidence(self, signal: Dict) -> float:
        """Calculate confidence using learned component weights"""
        try:
            component_scores = self.analyze_signal_components(signal)
            original_confidence = signal['confidence']
            
            # Calculate weighted score
            weighted_score = 0
            total_weight = 0
            
            for component, score in component_scores.items():
                weight = self.component_weights.get(component, 1.0)
                weighted_score += score * weight
                total_weight += weight
            
            if total_weight == 0:
                return original_confidence
            
            # Normalize weighted score
            normalized_score = weighted_score / total_weight
            
            # Adjust original confidence based on weighted score
            adjustment_factor = (normalized_score - 0.5) * 0.3  # ±15% max adjustment
            adjusted_confidence = original_confidence * (1 + adjustment_factor)
            
            # Ensure bounds
            adjusted_confidence = max(10.0, min(95.0, adjusted_confidence))
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Weighted confidence calculation failed: {e}")
            return signal['confidence']
    
    def record_signal_performance(self, signal: Dict, executed: bool = False):
        """Record signal for future learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            component_scores = self.analyze_signal_components(signal)
            
            cursor.execute('''
                INSERT INTO signal_performance (
                    symbol, signal_type, original_confidence, adjusted_confidence,
                    rsi_score, macd_score, ema_score, volume_score, regime_score,
                    executed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'],
                signal['signal_type'],
                signal['confidence'],
                signal.get('adjusted_confidence', signal['confidence']),
                component_scores.get('rsi', 0.5),
                component_scores.get('macd', 0.5),
                component_scores.get('ema', 0.5),
                component_scores.get('volume', 0.5),
                component_scores.get('regime', 0.5),
                executed
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record signal performance: {e}")
    
    def update_trade_outcome(self, symbol: str, timestamp: str, outcome_24h: float, win_loss: str):
        """Update trade outcome for learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find matching signal
            cursor.execute('''
                UPDATE signal_performance 
                SET outcome_24h = ?, win_loss = ?
                WHERE symbol = ? AND timestamp >= datetime(?, '-1 day')
                AND executed = 1
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (outcome_24h, win_loss, symbol, timestamp))
            
            conn.commit()
            conn.close()
            
            # Trigger learning update
            self.update_component_weights()
            
        except Exception as e:
            logger.error(f"Failed to update trade outcome: {e}")
    
    def update_component_weights(self):
        """Update component weights based on performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent performance data
            df = pd.read_sql_query('''
                SELECT * FROM signal_performance 
                WHERE executed = 1 AND win_loss IS NOT NULL
                AND timestamp > datetime('now', '-30 days')
            ''', conn)
            
            conn.close()
            
            if len(df) < self.min_samples:
                return
            
            # Calculate performance for each component
            for component in self.component_weights.keys():
                component_col = f'{component}_score'
                if component_col not in df.columns:
                    continue
                
                # Split into high/low component score groups
                high_threshold = df[component_col].quantile(0.7)
                
                high_group = df[df[component_col] >= high_threshold]
                low_group = df[df[component_col] < high_threshold]
                
                if len(high_group) >= 3 and len(low_group) >= 3:
                    high_win_rate = (high_group['win_loss'] == 'win').mean()
                    low_win_rate = (low_group['win_loss'] == 'win').mean()
                    
                    # Calculate performance difference
                    performance_diff = high_win_rate - low_win_rate
                    
                    # Update weight using learning rate
                    old_weight = self.component_weights[component]
                    weight_adjustment = self.learning_rate * performance_diff
                    new_weight = max(0.2, min(2.0, old_weight + weight_adjustment))
                    
                    if abs(new_weight - old_weight) > 0.01:
                        self.component_weights[component] = new_weight
                        
                        # Log weight change
                        self.log_weight_adjustment(component, old_weight, new_weight, 
                                                 performance_diff, len(df))
                        
                        logger.info(f"Updated {component} weight: {old_weight:.3f} → {new_weight:.3f} "
                                   f"(Performance diff: {performance_diff:.3f})")
            
        except Exception as e:
            logger.error(f"Component weight update failed: {e}")
    
    def log_weight_adjustment(self, component: str, old_weight: float, 
                            new_weight: float, performance_metric: float, sample_size: int):
        """Log weight adjustments for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO weight_adjustments (
                    component, old_weight, new_weight, performance_metric, sample_size
                ) VALUES (?, ?, ?, ?, ?)
            ''', (component, old_weight, new_weight, performance_metric, sample_size))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log weight adjustment: {e}")
    
    def get_tuning_insights(self) -> List[str]:
        """Get insights about confidence tuning performance"""
        try:
            insights = []
            
            # Component performance insights
            for component, weight in self.component_weights.items():
                if weight > 1.2:
                    insights.append(f"{component.upper()} component performing well (weight: {weight:.2f})")
                elif weight < 0.8:
                    insights.append(f"{component.upper()} component underperforming (weight: {weight:.2f})")
            
            # Recent adjustment insights
            conn = sqlite3.connect(self.db_path)
            recent_adjustments = pd.read_sql_query('''
                SELECT component, new_weight, performance_metric 
                FROM weight_adjustments 
                WHERE timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
                LIMIT 5
            ''', conn)
            conn.close()
            
            if not recent_adjustments.empty:
                insights.append(f"Recent tuning: {len(recent_adjustments)} weight adjustments")
            
            return insights if insights else ["Collecting performance data for tuning..."]
            
        except Exception as e:
            logger.error(f"Tuning insights failed: {e}")
            return ["Tuning analysis temporarily unavailable"]
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current component weights"""
        return self.component_weights.copy()
    
    def apply_confidence_tuning(self, signal: Dict) -> Dict:
        """Apply learned confidence adjustments to signal"""
        try:
            original_confidence = signal['confidence']
            tuned_confidence = self.calculate_weighted_confidence(signal)
            
            # Create tuned signal
            tuned_signal = signal.copy()
            tuned_signal['original_confidence'] = original_confidence
            tuned_signal['adjusted_confidence'] = tuned_confidence
            tuned_signal['confidence'] = tuned_confidence
            tuned_signal['tuning_applied'] = True
            
            # Record for learning
            self.record_signal_performance(tuned_signal)
            
            return tuned_signal
            
        except Exception as e:
            logger.error(f"Confidence tuning application failed: {e}")
            return signal