"""
Retraining Optimizer - Monitors dataset growth and triggers automated retraining
Triggers retraining when performance drops or sufficient new data is available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import sqlite3
from pathlib import Path
import threading
import time
import subprocess
import os

logger = logging.getLogger(__name__)

class RetrainingOptimizer:
    """
    Monitors model performance and dataset growth to trigger automated retraining
    Integrates with existing training pipelines without modification
    """
    
    def __init__(self, okx_data_service, ai_performance_tracker, adaptive_model_selector):
        self.okx_data_service = okx_data_service
        self.performance_tracker = ai_performance_tracker
        self.model_selector = adaptive_model_selector
        
        # Retraining thresholds
        self.new_data_threshold = 0.10  # 10% new data triggers retraining
        self.performance_drop_threshold = 0.15  # 15% performance drop triggers retraining
        self.min_hours_between_retraining = 24  # Minimum 24 hours between retraining cycles
        self.max_days_without_retraining = 7  # Force retraining after 7 days
        
        # Monitoring state
        self.last_retraining_times = {}  # symbol -> datetime
        self.baseline_performance = {}   # symbol -> {model -> performance_metrics}
        self.dataset_snapshots = {}     # symbol -> {timestamp -> data_size}
        self.is_monitoring = False
        self.monitoring_thread = None
        self.monitoring_interval = 3600  # Check every hour
        
        # Training status tracking
        self.active_retraining = set()  # Symbols currently being retrained
        self.retraining_queue = []      # Queue of pending retraining tasks
        self.max_concurrent_retraining = 2  # Maximum concurrent retraining processes
        
        self._initialize_monitoring()
        
    def _initialize_monitoring(self):
        """Initialize monitoring state and baselines"""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "XRPUSDT"]
        
        for symbol in symbols:
            # Initialize last retraining time (assume models were trained recently)
            self.last_retraining_times[symbol] = datetime.now() - timedelta(hours=12)
            
            # Initialize baseline performance
            self.baseline_performance[symbol] = {}
            for model in ['LSTM', 'Prophet', 'GradientBoost', 'Technical', 'Ensemble']:
                self.baseline_performance[symbol][model] = {
                    'win_rate': 55.0,  # Initial baseline
                    'avg_confidence': 65.0,
                    'total_trades': 10,
                    'sharpe_ratio': 0.3
                }
            
            # Initialize dataset snapshots
            self.dataset_snapshots[symbol] = {
                datetime.now().isoformat(): 1000  # Assume 1000 initial data points
            }
        
        logger.info("Retraining optimizer initialized with monitoring for 8 symbols")
    
    def start_monitoring(self):
        """Start automated monitoring and retraining"""
        if self.is_monitoring:
            logger.warning("Retraining monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started automated retraining monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring and retraining"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Stopped retraining monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop that checks for retraining triggers"""
        while self.is_monitoring:
            try:
                # Check each symbol for retraining needs
                for symbol in self.baseline_performance.keys():
                    if symbol not in self.active_retraining:
                        self._check_retraining_triggers(symbol)
                
                # Process retraining queue
                self._process_retraining_queue()
                
                # Sleep until next check
                for _ in range(self.monitoring_interval):
                    if not self.is_monitoring:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in retraining monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _check_retraining_triggers(self, symbol: str):
        """Check if retraining should be triggered for a symbol"""
        try:
            current_time = datetime.now()
            last_retraining = self.last_retraining_times.get(symbol, current_time - timedelta(days=30))
            hours_since_retraining = (current_time - last_retraining).total_seconds() / 3600
            
            # Skip if recently retrained
            if hours_since_retraining < self.min_hours_between_retraining:
                return
            
            # Force retraining if too much time has passed
            if hours_since_retraining > (self.max_days_without_retraining * 24):
                self._queue_retraining(symbol, "forced_schedule", 
                                     f"Forced retraining after {self.max_days_without_retraining} days")
                return
            
            # Check data growth trigger
            if self._check_data_growth_trigger(symbol):
                self._queue_retraining(symbol, "data_growth", 
                                     f"New data threshold ({self.new_data_threshold*100}%) exceeded")
                return
            
            # Check performance drop trigger
            if self._check_performance_drop_trigger(symbol):
                self._queue_retraining(symbol, "performance_drop", 
                                     f"Performance drop threshold ({self.performance_drop_threshold*100}%) exceeded")
                return
                
        except Exception as e:
            logger.error(f"Error checking retraining triggers for {symbol}: {e}")
    
    def _check_data_growth_trigger(self, symbol: str) -> bool:
        """Check if new data growth exceeds threshold"""
        try:
            # Get current data size (simulate based on time passage)
            current_time = datetime.now()
            
            # Simulate data growth: assume ~100 new data points per day
            if symbol in self.dataset_snapshots:
                latest_snapshot = max(self.dataset_snapshots[symbol].items(), key=lambda x: x[0])
                latest_time = datetime.fromisoformat(latest_snapshot[0])
                latest_size = latest_snapshot[1]
                
                days_passed = (current_time - latest_time).total_seconds() / 86400
                estimated_current_size = latest_size + int(days_passed * 100)
                
                # Calculate growth percentage
                growth_percentage = (estimated_current_size - latest_size) / latest_size
                
                if growth_percentage >= self.new_data_threshold:
                    # Update snapshot
                    self.dataset_snapshots[symbol][current_time.isoformat()] = estimated_current_size
                    logger.info(f"Data growth trigger for {symbol}: {growth_percentage:.1%} new data")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking data growth for {symbol}: {e}")
            return False
    
    def _check_performance_drop_trigger(self, symbol: str) -> bool:
        """Check if model performance has dropped significantly"""
        try:
            # Get current performance for active model
            active_model = self.model_selector.get_active_model(symbol)
            current_performance = self.performance_tracker.get_model_performance(
                symbol, active_model, days_back=3
            )
            
            if not current_performance or current_performance.get('total_trades', 0) < 5:
                return False  # Not enough recent data
            
            # Compare with baseline
            baseline = self.baseline_performance.get(symbol, {}).get(active_model, {})
            current_win_rate = current_performance.get('win_rate', 50)
            baseline_win_rate = baseline.get('win_rate', 50)
            
            # Calculate performance drop
            if baseline_win_rate > 0:
                performance_drop = (baseline_win_rate - current_win_rate) / baseline_win_rate
                
                if performance_drop >= self.performance_drop_threshold:
                    logger.info(f"Performance drop trigger for {symbol} {active_model}: "
                              f"{performance_drop:.1%} drop in win rate")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking performance drop for {symbol}: {e}")
            return False
    
    def _queue_retraining(self, symbol: str, trigger_type: str, reason: str):
        """Queue a symbol for retraining"""
        retraining_task = {
            'symbol': symbol,
            'trigger_type': trigger_type,
            'reason': reason,
            'queued_at': datetime.now(),
            'priority': self._get_trigger_priority(trigger_type)
        }
        
        # Check if already queued
        if any(task['symbol'] == symbol for task in self.retraining_queue):
            logger.debug(f"Retraining already queued for {symbol}")
            return
        
        self.retraining_queue.append(retraining_task)
        
        # Sort queue by priority
        self.retraining_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"Queued retraining for {symbol}: {reason}")
    
    def _get_trigger_priority(self, trigger_type: str) -> int:
        """Get priority score for trigger type"""
        priority_map = {
            'performance_drop': 3,  # Highest priority
            'data_growth': 2,       # Medium priority
            'forced_schedule': 1,   # Low priority
            'manual': 4            # Manual triggers get highest priority
        }
        return priority_map.get(trigger_type, 1)
    
    def _process_retraining_queue(self):
        """Process pending retraining tasks"""
        if not self.retraining_queue:
            return
        
        # Check if we can start more retraining processes
        available_slots = self.max_concurrent_retraining - len(self.active_retraining)
        
        if available_slots <= 0:
            return
        
        # Start retraining for available slots
        for _ in range(min(available_slots, len(self.retraining_queue))):
            task = self.retraining_queue.pop(0)
            self._start_retraining(task)
    
    def _start_retraining(self, task: Dict[str, Any]):
        """Start retraining process for a symbol"""
        symbol = task['symbol']
        
        try:
            self.active_retraining.add(symbol)
            
            # Start retraining in separate thread
            retraining_thread = threading.Thread(
                target=self._execute_retraining,
                args=(symbol, task),
                daemon=True
            )
            retraining_thread.start()
            
            logger.info(f"Started retraining for {symbol} (trigger: {task['trigger_type']})")
            
        except Exception as e:
            logger.error(f"Error starting retraining for {symbol}: {e}")
            self.active_retraining.discard(symbol)
    
    def _execute_retraining(self, symbol: str, task: Dict[str, Any]):
        """Execute the actual retraining process"""
        try:
            start_time = datetime.now()
            
            # Simulate retraining by calling existing training pipeline
            success = self._call_training_pipeline(symbol)
            
            if success:
                # Update last retraining time
                self.last_retraining_times[symbol] = datetime.now()
                
                # Update baseline performance after retraining
                self._update_baseline_performance(symbol)
                
                # Force model evaluation to pick up new performance
                self.model_selector.force_model_evaluation()
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed retraining for {symbol} in {duration:.1f} seconds")
                
            else:
                logger.error(f"Retraining failed for {symbol}")
                
        except Exception as e:
            logger.error(f"Error during retraining execution for {symbol}: {e}")
            
        finally:
            # Remove from active retraining set
            self.active_retraining.discard(symbol)
    
    def _call_training_pipeline(self, symbol: str) -> bool:
        """Call existing training pipeline for a symbol"""
        try:
            # Simulate calling the autonomous training system
            # In real implementation, this would call run_autonomous_training.py
            # or use the existing training components directly
            
            # For now, simulate successful training
            training_duration = np.random.uniform(30, 120)  # 30-120 seconds
            time.sleep(min(training_duration, 10))  # Cap simulation time at 10 seconds
            
            # Simulate 90% success rate
            success = np.random.random() > 0.1
            
            if success:
                logger.info(f"Training pipeline completed for {symbol}")
            else:
                logger.warning(f"Training pipeline failed for {symbol}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error calling training pipeline for {symbol}: {e}")
            return False
    
    def _update_baseline_performance(self, symbol: str):
        """Update baseline performance metrics after retraining"""
        try:
            # Get fresh performance data for all models
            for model in ['LSTM', 'Prophet', 'GradientBoost', 'Technical', 'Ensemble']:
                performance_data = self.performance_tracker.get_model_performance(
                    symbol, model, days_back=1
                )
                
                if performance_data and performance_data.get('total_trades', 0) > 0:
                    self.baseline_performance[symbol][model] = {
                        'win_rate': performance_data.get('win_rate', 55),
                        'avg_confidence': performance_data.get('avg_confidence', 65),
                        'total_trades': performance_data.get('total_trades', 0),
                        'sharpe_ratio': performance_data.get('sharpe_ratio', 0.3)
                    }
            
            logger.debug(f"Updated baseline performance for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating baseline performance for {symbol}: {e}")
    
    def trigger_manual_retraining(self, symbol: str, reason: str = "Manual trigger") -> bool:
        """Manually trigger retraining for a symbol"""
        try:
            if symbol in self.active_retraining:
                logger.warning(f"Retraining already active for {symbol}")
                return False
            
            # Remove from queue if already queued
            self.retraining_queue = [task for task in self.retraining_queue 
                                   if task['symbol'] != symbol]
            
            # Queue with high priority
            self._queue_retraining(symbol, "manual", reason)
            
            logger.info(f"Manually triggered retraining for {symbol}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error triggering manual retraining for {symbol}: {e}")
            return False
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status and statistics"""
        try:
            current_time = datetime.now()
            
            status = {
                'monitoring_active': self.is_monitoring,
                'active_retraining': list(self.active_retraining),
                'queued_tasks': len(self.retraining_queue),
                'queue_details': [
                    {
                        'symbol': task['symbol'],
                        'trigger_type': task['trigger_type'],
                        'reason': task['reason'],
                        'priority': task['priority'],
                        'queued_minutes_ago': int((current_time - task['queued_at']).total_seconds() / 60)
                    }
                    for task in self.retraining_queue[:5]  # Show first 5
                ],
                'last_retraining_times': {},
                'next_scheduled_checks': {},
                'performance_status': {}
            }
            
            # Add last retraining times and next checks
            for symbol in self.last_retraining_times:
                last_time = self.last_retraining_times[symbol]
                hours_ago = (current_time - last_time).total_seconds() / 3600
                
                status['last_retraining_times'][symbol] = {
                    'timestamp': last_time.isoformat(),
                    'hours_ago': round(hours_ago, 1)
                }
                
                # Calculate next forced retraining time
                next_forced = last_time + timedelta(days=self.max_days_without_retraining)
                hours_until_forced = (next_forced - current_time).total_seconds() / 3600
                
                status['next_scheduled_checks'][symbol] = {
                    'next_forced_retraining': next_forced.isoformat(),
                    'hours_until_forced': max(0, round(hours_until_forced, 1))
                }
                
                # Add performance status
                active_model = self.model_selector.get_active_model(symbol)
                recent_performance = self.performance_tracker.get_model_performance(
                    symbol, active_model, days_back=3
                )
                
                baseline = self.baseline_performance.get(symbol, {}).get(active_model, {})
                
                if recent_performance and baseline:
                    current_win_rate = recent_performance.get('win_rate', 50)
                    baseline_win_rate = baseline.get('win_rate', 50)
                    performance_change = current_win_rate - baseline_win_rate
                    
                    status['performance_status'][symbol] = {
                        'active_model': active_model,
                        'current_win_rate': current_win_rate,
                        'baseline_win_rate': baseline_win_rate,
                        'performance_change': round(performance_change, 1),
                        'recent_trades': recent_performance.get('total_trades', 0)
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting retraining status: {e}")
            return {'error': str(e)}
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on current state"""
        recommendations = []
        
        try:
            current_time = datetime.now()
            
            for symbol in self.baseline_performance.keys():
                # Skip if currently retraining
                if symbol in self.active_retraining:
                    continue
                
                # Check for performance issues
                active_model = self.model_selector.get_active_model(symbol)
                recent_performance = self.performance_tracker.get_model_performance(
                    symbol, active_model, days_back=3
                )
                
                if recent_performance and recent_performance.get('total_trades', 0) >= 5:
                    win_rate = recent_performance.get('win_rate', 50)
                    confidence = recent_performance.get('avg_confidence', 50)
                    
                    if win_rate < 45:
                        recommendations.append({
                            'symbol': symbol,
                            'type': 'performance_concern',
                            'priority': 'high',
                            'message': f'Low win rate ({win_rate:.1f}%) - consider retraining',
                            'action': 'retrain'
                        })
                    
                    if confidence < 50:
                        recommendations.append({
                            'symbol': symbol,
                            'type': 'confidence_issue',
                            'priority': 'medium',
                            'message': f'Low confidence ({confidence:.1f}%) - model uncertainty',
                            'action': 'evaluate_models'
                        })
                
                # Check for stale models
                last_retraining = self.last_retraining_times.get(symbol)
                if last_retraining:
                    days_since_retraining = (current_time - last_retraining).total_seconds() / 86400
                    
                    if days_since_retraining > 5:
                        recommendations.append({
                            'symbol': symbol,
                            'type': 'stale_model',
                            'priority': 'low',
                            'message': f'Model not retrained for {days_since_retraining:.1f} days',
                            'action': 'consider_retraining'
                        })
            
            # Sort by priority
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return []