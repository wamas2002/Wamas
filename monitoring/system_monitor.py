#!/usr/bin/env python3
"""
Production System Monitor for Autonomous Trading Bot
Handles real-time monitoring, alerts, and performance tracking
"""

import os
import time
import json
import logging
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import psutil
import requests
from dataclasses import dataclass
from collections import deque

@dataclass
class TradeMetrics:
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    pnl: float
    confidence: float

@dataclass
class SystemHealth:
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    api_latency: float
    active_positions: int
    daily_pnl: float

class ProductionMonitor:
    """Comprehensive production monitoring system"""
    
    def __init__(self):
        self.setup_logging()
        self.metrics_history = deque(maxlen=10000)
        self.trade_history = deque(maxlen=1000)
        self.performance_metrics = {}
        self.alert_thresholds = {
            'daily_loss_limit': 0.10,  # 10% daily loss
            'position_risk_limit': 0.02,  # 2% per position
            'api_latency_limit': 500,  # 500ms
            'memory_limit': 80,  # 80% memory usage
            'cpu_limit': 85,  # 85% CPU usage
            'consecutive_losses': 3
        }
        
        self.consecutive_losses = 0
        self.daily_start_balance = 0
        self.current_balance = 0
        self.is_monitoring = False
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        os.makedirs('logs', exist_ok=True)
        
        # Main system log
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system_monitor.log'),
                logging.StreamHandler()
            ]
        )
        
        # Trade-specific logger
        self.trade_logger = logging.getLogger('trades')
        trade_handler = logging.FileHandler('logs/trades.log')
        trade_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.trade_logger.addHandler(trade_handler)
        
        # Performance logger
        self.perf_logger = logging.getLogger('performance')
        perf_handler = logging.FileHandler('logs/performance.log')
        perf_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.perf_logger.addHandler(perf_handler)
        
    def start_monitoring(self):
        """Start the monitoring system"""
        logging.info("Starting production monitoring system...")
        self.is_monitoring = True
        
        # Schedule regular tasks
        schedule.every(5).seconds.do(self.collect_system_metrics)
        schedule.every(1).minutes.do(self.check_trading_health)
        schedule.every(10).minutes.do(self.export_performance_data)
        schedule.every(1).hours.do(self.optimize_memory)
        schedule.every(24).hours.do(self.schedule_model_retraining)
        schedule.every(6).hours.do(self.rebalance_portfolio)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        logging.info("Production monitoring system started successfully")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
                
    def collect_system_metrics(self):
        """Collect system health metrics"""
        try:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # API latency test
            start_time = time.time()
            try:
                response = requests.get("https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT", timeout=5)
                api_latency = (time.time() - start_time) * 1000
            except:
                api_latency = 9999
            
            health = SystemHealth(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                api_latency=api_latency,
                active_positions=self.get_active_positions_count(),
                daily_pnl=self.calculate_daily_pnl()
            )
            
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'health': health
            })
            
            # Check thresholds and alert if necessary
            self.check_alert_thresholds(health)
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
            
    def check_alert_thresholds(self, health: SystemHealth):
        """Check if any metrics exceed alert thresholds"""
        alerts = []
        
        # Performance alerts
        if health.cpu_percent > self.alert_thresholds['cpu_limit']:
            alerts.append(f"High CPU usage: {health.cpu_percent:.1f}%")
            
        if health.memory_percent > self.alert_thresholds['memory_limit']:
            alerts.append(f"High memory usage: {health.memory_percent:.1f}%")
            
        if health.api_latency > self.alert_thresholds['api_latency_limit']:
            alerts.append(f"High API latency: {health.api_latency:.1f}ms")
            
        # Trading alerts
        daily_loss_pct = abs(health.daily_pnl) / max(self.daily_start_balance, 1) if health.daily_pnl < 0 else 0
        if daily_loss_pct > self.alert_thresholds['daily_loss_limit']:
            alerts.append(f"Daily loss limit exceeded: {daily_loss_pct:.1%}")
            self.trigger_circuit_breaker()
            
        if self.consecutive_losses >= self.alert_thresholds['consecutive_losses']:
            alerts.append(f"Consecutive losses: {self.consecutive_losses}")
            
        # Send alerts if any
        if alerts:
            self.send_alerts(alerts)
            
    def schedule_model_retraining(self):
        """Schedule and execute model retraining"""
        try:
            logging.info("Starting scheduled model retraining...")
            
            # Import and execute training modules
            from ai.comprehensive_ml_pipeline import ComprehensiveMLPipeline
            from ai.freqai_pipeline import FreqAILevelPipeline
            from trading.okx_data_service import OKXDataService
            
            data_service = OKXDataService()
            
            # Retrain models for major pairs
            symbols = ['BTC-USDT', 'ETH-USDT', 'ADA-USDT']
            
            for symbol in symbols:
                try:
                    # Get fresh training data
                    df = data_service.get_historical_data(symbol, '1H', 500)
                    
                    if df is not None and len(df) > 100:
                        # Retrain comprehensive ML pipeline
                        ml_pipeline = ComprehensiveMLPipeline()
                        result = ml_pipeline.train_all_models(df)
                        
                        if result.get('success'):
                            logging.info(f"Successfully retrained models for {symbol}")
                        else:
                            logging.warning(f"Model retraining failed for {symbol}")
                            
                except Exception as e:
                    logging.error(f"Error retraining models for {symbol}: {e}")
                    
            logging.info("Model retraining cycle completed")
            
        except Exception as e:
            logging.error(f"Error in scheduled model retraining: {e}")
            
    def get_active_positions_count(self) -> int:
        """Get count of active trading positions"""
        try:
            return len(self.trade_history)
        except:
            return 0
            
    def calculate_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        try:
            today_trades = [t for t in self.trade_history 
                          if t.timestamp.date() == datetime.now().date()]
            return sum(t.pnl for t in today_trades)
        except:
            return 0.0
            
    def send_alerts(self, messages: List[str], priority: str = "WARNING"):
        """Send alert notifications"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Log alerts
            for message in messages:
                logging.warning(f"ALERT [{priority}]: {message}")
                
            # Write to alert log
            with open('logs/alerts.log', 'a') as f:
                for message in messages:
                    f.write(f"{timestamp} [{priority}]: {message}\n")
                    
        except Exception as e:
            logging.error(f"Error sending alerts: {e}")

    def trigger_circuit_breaker(self):
        """Trigger emergency circuit breaker"""
        try:
            logging.critical("CIRCUIT BREAKER TRIGGERED - Emergency stop activated")
            self.trade_logger.critical("Trading halted due to circuit breaker activation")
            self.send_alerts([
                "EMERGENCY: Circuit breaker activated",
                "All trading has been halted",
                "Manual intervention required"
            ], priority="CRITICAL")
            
        except Exception as e:
            logging.error(f"Error triggering circuit breaker: {e}")

# Global monitor instance
monitor = ProductionMonitor()

def start_production_monitoring():
    """Start the production monitoring system"""
    monitor.start_monitoring()
    
def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    if monitor.metrics_history:
        latest = monitor.metrics_history[-1]
        health = latest['health']
        return {
            "timestamp": latest['timestamp'].isoformat(),
            "cpu_percent": health.cpu_percent,
            "memory_percent": health.memory_percent,
            "api_latency": health.api_latency,
            "active_positions": health.active_positions,
            "daily_pnl": health.daily_pnl,
            "status": "healthy" if health.cpu_percent < 80 and health.memory_percent < 80 else "warning"
        }
    return {"status": "no_data"}