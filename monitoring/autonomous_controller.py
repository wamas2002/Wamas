#!/usr/bin/env python3
"""
Autonomous Trading System Controller
Manages long-term autonomous operation with self-optimization
"""

import os
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import schedule
import json
import pandas as pd
from dataclasses import dataclass

@dataclass
class AutoConfig:
    model_retrain_interval: int = 24  # hours
    performance_review_interval: int = 6  # hours
    memory_cleanup_interval: int = 1  # hours
    health_check_interval: int = 5  # minutes
    auto_rebalance_enabled: bool = True
    circuit_breaker_enabled: bool = True
    adaptive_position_sizing: bool = True

class AutonomousController:
    """Main controller for autonomous trading system operation"""
    
    def __init__(self):
        self.setup_logging()
        self.config = AutoConfig()
        self.is_running = False
        self.last_model_retrain = {}
        self.performance_history = []
        self.active_strategies = ['comprehensive_ml', 'freqai', 'lstm', 'prophet']
        self.strategy_performance = {}
        self.auto_optimization_enabled = True
        
    def setup_logging(self):
        """Setup logging for autonomous controller"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/autonomous_controller.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AutonomousController')
        
    def start_autonomous_operation(self):
        """Start autonomous trading operation"""
        self.logger.info("Starting autonomous trading system...")
        self.is_running = True
        
        # Schedule autonomous tasks
        schedule.every(self.config.model_retrain_interval).hours.do(self.autonomous_model_retraining)
        schedule.every(self.config.performance_review_interval).hours.do(self.performance_review_and_optimization)
        schedule.every(self.config.memory_cleanup_interval).hours.do(self.system_maintenance)
        schedule.every(self.config.health_check_interval).minutes.do(self.health_check_and_alerts)
        
        # Start controller thread
        controller_thread = threading.Thread(target=self._autonomous_loop, daemon=True)
        controller_thread.start()
        
        self.logger.info("Autonomous trading system started successfully")
        
    def _autonomous_loop(self):
        """Main autonomous operation loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in autonomous loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
                
    def autonomous_model_retraining(self):
        """Automatically retrain models based on performance degradation"""
        try:
            self.logger.info("Starting autonomous model retraining cycle...")
            
            from ai.comprehensive_ml_pipeline import ComprehensiveMLPipeline
            from ai.freqai_pipeline import FreqAILevelPipeline
            from ai.lstm_predictor import AdvancedLSTMPredictor
            from ai.prophet_predictor import AdvancedProphetPredictor
            from trading.okx_data_service import OKXDataService
            
            data_service = OKXDataService()
            symbols = ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'BNB-USDT']
            
            for symbol in symbols:
                try:
                    # Get fresh market data
                    df = data_service.get_historical_data(symbol, '1H', 500)
                    
                    if df is None or len(df) < 100:
                        self.logger.warning(f"Insufficient data for {symbol}, skipping retraining")
                        continue
                    
                    # Check if retraining is needed based on performance
                    if self._should_retrain_models(symbol):
                        self.logger.info(f"Retraining models for {symbol}")
                        
                        # Retrain comprehensive ML pipeline
                        ml_pipeline = ComprehensiveMLPipeline()
                        ml_result = ml_pipeline.train_all_models(df)
                        
                        # Retrain FreqAI pipeline
                        freqai_pipeline = FreqAILevelPipeline()
                        freqai_result = freqai_pipeline.train_all_models(df)
                        
                        # Retrain LSTM
                        lstm_predictor = AdvancedLSTMPredictor()
                        lstm_result = lstm_predictor.train(df)
                        
                        # Retrain Prophet
                        prophet_predictor = AdvancedProphetPredictor()
                        prophet_result = prophet_predictor.train(df)
                        
                        # Update last retrain timestamp
                        self.last_model_retrain[symbol] = datetime.now()
                        
                        self.logger.info(f"Model retraining completed for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error retraining models for {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in autonomous model retraining: {e}")
            
    def _should_retrain_models(self, symbol: str) -> bool:
        """Determine if models need retraining based on performance"""
        try:
            # Check if enough time has passed since last retrain
            last_retrain = self.last_model_retrain.get(symbol)
            if last_retrain:
                hours_since_retrain = (datetime.now() - last_retrain).total_seconds() / 3600
                if hours_since_retrain < self.config.model_retrain_interval:
                    return False
            
            # Check recent prediction accuracy
            recent_performance = self._get_recent_model_performance(symbol)
            
            # Retrain if accuracy has degraded significantly
            if recent_performance and recent_performance.get('accuracy', 0) < 0.6:
                return True
                
            # Retrain if it's been more than 48 hours
            if not last_retrain or hours_since_retrain > 48:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking retrain necessity for {symbol}: {e}")
            return True  # Default to retraining on error
            
    def _get_recent_model_performance(self, symbol: str) -> Dict[str, float]:
        """Get recent model performance metrics"""
        try:
            # This would integrate with your performance tracking system
            # For now, return sample data
            return {
                'accuracy': 0.72,
                'precision': 0.68,
                'recall': 0.75,
                'f1_score': 0.71
            }
        except Exception:
            return {}
            
    def performance_review_and_optimization(self):
        """Review performance and optimize strategy allocation"""
        try:
            self.logger.info("Starting performance review and optimization...")
            
            # Analyze strategy performance
            strategy_metrics = self._analyze_strategy_performance()
            
            # Optimize strategy weights based on performance
            if self.auto_optimization_enabled:
                new_weights = self._optimize_strategy_weights(strategy_metrics)
                self._update_strategy_allocation(new_weights)
                
            # Auto-adjust risk parameters
            self._auto_adjust_risk_parameters(strategy_metrics)
            
            self.logger.info("Performance optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error in performance review: {e}")
            
    def _analyze_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance of each trading strategy"""
        try:
            # This would integrate with your trading history
            # For now, return sample metrics
            return {
                'comprehensive_ml': {
                    'win_rate': 0.68,
                    'avg_return': 0.0234,
                    'sharpe_ratio': 1.82,
                    'max_drawdown': 0.045
                },
                'freqai': {
                    'win_rate': 0.71,
                    'avg_return': 0.0198,
                    'sharpe_ratio': 2.15,
                    'max_drawdown': 0.038
                },
                'lstm': {
                    'win_rate': 0.65,
                    'avg_return': 0.0167,
                    'sharpe_ratio': 1.67,
                    'max_drawdown': 0.052
                },
                'prophet': {
                    'win_rate': 0.62,
                    'avg_return': 0.0145,
                    'sharpe_ratio': 1.43,
                    'max_drawdown': 0.048
                }
            }
        except Exception as e:
            self.logger.error(f"Error analyzing strategy performance: {e}")
            return {}
            
    def _optimize_strategy_weights(self, strategy_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Optimize strategy allocation weights based on performance"""
        try:
            if not strategy_metrics:
                return {strategy: 0.25 for strategy in self.active_strategies}
            
            # Calculate performance scores
            scores = {}
            for strategy, metrics in strategy_metrics.items():
                # Weighted score based on multiple metrics
                score = (
                    metrics.get('win_rate', 0) * 0.3 +
                    metrics.get('sharpe_ratio', 0) / 3 * 0.4 +  # Normalize Sharpe
                    (1 - metrics.get('max_drawdown', 0.1)) * 0.3
                )
                scores[strategy] = max(0.1, score)  # Minimum 10% allocation
            
            # Normalize to sum to 1
            total_score = sum(scores.values())
            weights = {strategy: score / total_score for strategy, score in scores.items()}
            
            self.logger.info(f"Optimized strategy weights: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy weights: {e}")
            return {strategy: 0.25 for strategy in self.active_strategies}
            
    def _update_strategy_allocation(self, new_weights: Dict[str, float]):
        """Update strategy allocation weights"""
        try:
            # This would integrate with your strategy manager
            self.strategy_performance = new_weights
            self.logger.info(f"Updated strategy allocation: {new_weights}")
        except Exception as e:
            self.logger.error(f"Error updating strategy allocation: {e}")
            
    def _auto_adjust_risk_parameters(self, strategy_metrics: Dict[str, Dict[str, float]]):
        """Automatically adjust risk parameters based on performance"""
        try:
            if not strategy_metrics:
                return
                
            # Calculate average max drawdown
            avg_drawdown = sum(metrics.get('max_drawdown', 0.05) 
                             for metrics in strategy_metrics.values()) / len(strategy_metrics)
            
            # Adjust position sizing based on recent performance
            if avg_drawdown > 0.06:  # If drawdown > 6%, reduce position sizes
                position_size_multiplier = 0.8
                self.logger.info("Reducing position sizes due to high drawdown")
            elif avg_drawdown < 0.03:  # If drawdown < 3%, increase position sizes
                position_size_multiplier = 1.1
                self.logger.info("Increasing position sizes due to low drawdown")
            else:
                position_size_multiplier = 1.0
                
            # This would integrate with your risk management system
            self.logger.info(f"Position size multiplier adjusted to: {position_size_multiplier}")
            
        except Exception as e:
            self.logger.error(f"Error auto-adjusting risk parameters: {e}")
            
    def system_maintenance(self):
        """Perform system maintenance and optimization"""
        try:
            self.logger.info("Starting system maintenance...")
            
            # Memory cleanup
            self._cleanup_memory()
            
            # Log rotation
            self._rotate_logs()
            
            # Performance data export
            self._export_performance_data()
            
            # System health check
            self._comprehensive_health_check()
            
            self.logger.info("System maintenance completed")
            
        except Exception as e:
            self.logger.error(f"Error in system maintenance: {e}")
            
    def _cleanup_memory(self):
        """Clean up memory usage"""
        try:
            import gc
            import psutil
            
            # Get current memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Force garbage collection
            gc.collect()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after
            
            self.logger.info(f"Memory cleanup: freed {memory_freed:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"Error in memory cleanup: {e}")
            
    def _rotate_logs(self):
        """Rotate log files to prevent disk space issues"""
        try:
            import glob
            import shutil
            
            log_files = glob.glob('logs/*.log')
            
            for log_file in log_files:
                # Get file size in MB
                file_size = os.path.getsize(log_file) / 1024 / 1024
                
                if file_size > 50:  # If file > 50MB, rotate
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"{log_file}.{timestamp}.bak"
                    shutil.move(log_file, backup_name)
                    self.logger.info(f"Rotated log file: {log_file}")
                    
        except Exception as e:
            self.logger.error(f"Error rotating logs: {e}")
            
    def _export_performance_data(self):
        """Export performance data for analysis"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export strategy performance
            if self.strategy_performance:
                with open(f'logs/strategy_performance_{timestamp}.json', 'w') as f:
                    json.dump(self.strategy_performance, f, indent=2)
                    
            # Export system metrics
            system_metrics = {
                'timestamp': timestamp,
                'active_strategies': self.active_strategies,
                'auto_optimization': self.auto_optimization_enabled,
                'last_retrain': {k: v.isoformat() for k, v in self.last_model_retrain.items()}
            }
            
            with open(f'logs/system_metrics_{timestamp}.json', 'w') as f:
                json.dump(system_metrics, f, indent=2)
                
            self.logger.info(f"Performance data exported: {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Error exporting performance data: {e}")
            
    def _comprehensive_health_check(self):
        """Perform comprehensive system health check"""
        try:
            import psutil
            import requests
            
            health_status = {}
            
            # Check system resources
            health_status['cpu_percent'] = psutil.cpu_percent(interval=1)
            health_status['memory_percent'] = psutil.virtual_memory().percent
            health_status['disk_percent'] = psutil.disk_usage('/').percent
            
            # Check API connectivity
            try:
                response = requests.get("https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT", timeout=5)
                health_status['api_status'] = 'connected' if response.status_code == 200 else 'error'
                health_status['api_latency'] = response.elapsed.total_seconds() * 1000
            except:
                health_status['api_status'] = 'disconnected'
                health_status['api_latency'] = 9999
                
            # Log health status
            self.logger.info(f"System health: {health_status}")
            
            # Alert if critical issues
            if health_status['memory_percent'] > 90:
                self.logger.critical("Critical: Memory usage > 90%")
            if health_status['api_status'] != 'connected':
                self.logger.critical("Critical: API disconnected")
                
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            
    def health_check_and_alerts(self):
        """Quick health check and alert generation"""
        try:
            # Quick system check
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Generate alerts for critical conditions
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            if memory_percent > 85:
                self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
                
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get current autonomous system status"""
        return {
            'is_running': self.is_running,
            'active_strategies': self.active_strategies,
            'auto_optimization_enabled': self.auto_optimization_enabled,
            'last_model_retrain': {k: v.isoformat() for k, v in self.last_model_retrain.items()},
            'config': {
                'model_retrain_interval': self.config.model_retrain_interval,
                'performance_review_interval': self.config.performance_review_interval,
                'auto_rebalance_enabled': self.config.auto_rebalance_enabled,
                'circuit_breaker_enabled': self.config.circuit_breaker_enabled
            }
        }
        
    def stop_autonomous_operation(self):
        """Stop autonomous trading operation"""
        self.logger.info("Stopping autonomous trading system...")
        self.is_running = False

# Global controller instance
autonomous_controller = AutonomousController()

def start_autonomous_trading():
    """Start autonomous trading system"""
    autonomous_controller.start_autonomous_operation()
    
def get_autonomous_status():
    """Get autonomous system status"""
    return autonomous_controller.get_autonomous_status()

if __name__ == "__main__":
    start_autonomous_trading()
    
    try:
        while True:
            time.sleep(3600)  # Keep running
    except KeyboardInterrupt:
        autonomous_controller.stop_autonomous_operation()
        print("Autonomous system stopped")